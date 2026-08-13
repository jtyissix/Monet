import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

from inference.vllm_hrbench_attention_analysis import (
    QUERY_ANSWER,
    QUERY_LATENT,
    SOURCE_GENERATED_TEXT,
    SOURCE_INPUT_TEXT,
    SOURCE_INPUT_VISUAL,
    SOURCE_LATENT,
    SOURCE_SPECIAL,
    assemble_sample_archive,
    build_category_attention_csv_rows,
    build_latent_topk_csv_rows,
    category_attention_csv_fieldnames,
    classify_source_positions,
    latent_topk_csv_fieldnames,
    normalize_attention_groups,
    select_final_answer_token_indices,
    select_sample_indices,
    write_csv,
)


class FakeTokenizer:
    texts = {
        7: "中文,\"候选\"\n下一行",
        8: "plain",
        30: "answer",
    }

    def decode(self, token_ids, skip_special_tokens=False):
        del skip_special_tokens
        return "".join(self.texts.get(int(token_id), f"t{token_id}")
                       for token_id in token_ids)


class AttentionAnalysisHelpersTest(unittest.TestCase):
    def test_selection_is_deterministic(self):
        first = select_sample_indices(20, "random", 0, 5, 7)
        second = select_sample_indices(20, "random", 0, 5, 7)
        self.assertEqual(first, second)
        self.assertEqual(
            select_sample_indices(20, "sequential", 3, 4, 99),
            [3, 4, 5, 6],
        )

    def test_source_classification_precedence(self):
        kinds, token_ids = classify_source_positions(
            np.arange(7, dtype=np.int32),
            prompt_length=3,
            image_positions={1},
            latent_positions={4},
            prompt_token_ids=[10, 11, 12],
            generated_token_ids=[20, 21, 22, 23],
            special_token_ids={11, 20},
        )
        self.assertEqual(token_ids.tolist(), [10, 11, 12, 20, 21, 22, 23])
        self.assertEqual(kinds.tolist(), [
            SOURCE_INPUT_TEXT,
            SOURCE_INPUT_VISUAL,
            SOURCE_INPUT_TEXT,
            SOURCE_SPECIAL,
            SOURCE_LATENT,
            SOURCE_GENERATED_TEXT,
            SOURCE_GENERATED_TEXT,
        ])

    def test_group_normalization_preserves_only_targets(self):
        kinds = np.asarray([
            SOURCE_INPUT_TEXT,
            SOURCE_INPUT_VISUAL,
            SOURCE_GENERATED_TEXT,
            SOURCE_LATENT,
        ])
        raw = np.asarray([[0.2, 0.3, 0.1, 0.4]], dtype=np.float32)
        normalized = normalize_attention_groups(
            raw, kinds, (SOURCE_INPUT_TEXT, SOURCE_INPUT_VISUAL))
        np.testing.assert_allclose(normalized, [[0.4, 0.6, 0.0, 0.0]])

    def test_answer_range_without_latent_uses_all_readable_output(self):
        indices, fallback = select_final_answer_token_indices(
            [5, 99, 6], 100, 101, {99, 100, 101})
        self.assertEqual(indices, [0, 2])
        self.assertTrue(fallback)

    def test_answer_range_uses_only_content_after_final_latent(self):
        indices, fallback = select_final_answer_token_indices(
            [5, 100, 50, 101, 6, 100, 51, 101, 7, 99],
            100,
            101,
            {99, 100, 101},
        )
        self.assertEqual(indices, [8])
        self.assertFalse(fallback)

    def test_answer_range_can_be_empty_after_unclosed_latent(self):
        indices, fallback = select_final_answer_token_indices(
            [5, 100, 50], 100, 101, {100, 101})
        self.assertEqual(indices, [])
        self.assertFalse(fallback)

    def _write_spool(self, directory, name, matrices):
        path = directory / name
        with path.open("wb") as handle:
            for matrix in matrices:
                np.asarray(matrix, dtype=np.float32).tofile(handle)
        return path

    def test_ragged_archive_with_latent_and_answer(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            latent = np.asarray([
                [0.1, 0.2, 0.3, 0.4],
                [0.4, 0.3, 0.2, 0.1],
            ], dtype=np.float32)
            answer = np.asarray([
                [0.1, 0.1, 0.2, 0.2, 0.2, 0.2],
                [0.2, 0.2, 0.1, 0.1, 0.2, 0.2],
            ], dtype=np.float32)
            self._write_spool(directory, "latent.bin", [latent])
            self._write_spool(directory, "answer.bin", [answer])
            manifest = {
                "storage_dtype": "float32",
                "layer_names": ["layers.0", "layers.1"],
                "prompt_length": 3,
                "prompt_token_ids": [1, 2, 3],
                "generated_token_ids": [100, 101, 102],
                "image_positions": [1],
                "latent_positions": [3],
                "special_token_ids": [100],
                "no_latent_fallback": False,
                "latent_spool": "latent.bin",
                "answer_spool": "answer.bin",
                "latent_records": [{
                    "query_sequence_position": 3,
                    "source_count": 4,
                    "layer_count": 2,
                    "offset": 0,
                    "latent_index": 0,
                }],
                "answer_records": [{
                    "query_sequence_position": 5,
                    "source_count": 6,
                    "layer_count": 2,
                    "offset": 0,
                    "output_index": 2,
                    "predicted_token_id": 102,
                }],
                "latent_topk": [{
                    "query_sequence_position": 3,
                    "latent_index": 0,
                    "token_ids": list(range(20)),
                    "logits": [float(value) for value in range(20)],
                }],
            }
            data = assemble_sample_archive(manifest, directory)
            self.assertEqual(data["raw_attention"].shape, (2, 10))
            self.assertEqual(data["query_source_offsets"].tolist(), [0, 4, 10])
            self.assertEqual(data["query_kind_codes"].tolist(), [
                QUERY_LATENT, QUERY_ANSWER,
            ])
            self.assertEqual(data["latent_topk_token_ids"].shape, (1, 20))
            np.testing.assert_allclose(
                data["category_attention_mass"].sum(axis=-1), 1.0)

    def test_empty_latent_stream_is_supported(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            (directory / "latent.bin").write_bytes(b"")
            answer = np.asarray([[0.5, 0.5]], dtype=np.float32)
            self._write_spool(directory, "answer.bin", [answer])
            manifest = {
                "storage_dtype": "float32",
                "layer_names": ["layers.0"],
                "prompt_length": 1,
                "prompt_token_ids": [1],
                "generated_token_ids": [5],
                "image_positions": [],
                "latent_positions": [],
                "special_token_ids": [],
                "no_latent_fallback": True,
                "latent_spool": "latent.bin",
                "answer_spool": "answer.bin",
                "latent_records": [],
                "answer_records": [{
                    "query_sequence_position": 1,
                    "source_count": 2,
                    "layer_count": 1,
                    "offset": 0,
                    "output_index": 0,
                    "predicted_token_id": 5,
                }],
                "latent_topk": [],
            }
            data = assemble_sample_archive(manifest, directory)
            self.assertEqual(data["query_kind_codes"].tolist(), [QUERY_ANSWER])
            self.assertEqual(data["latent_topk_token_ids"].shape, (0, 20))
            self.assertTrue(bool(data["no_latent_fallback"]))

    def test_category_csv_rows_match_all_query_layer_masses(self):
        data = {
            "category_attention_mass": np.asarray([
                [[0.1, 0.2, 0.3, 0.2, 0.2],
                 [0.2, 0.1, 0.2, 0.3, 0.2]],
                [[0.4, 0.1, 0.1, 0.2, 0.2],
                 [0.3, 0.2, 0.1, 0.1, 0.3]],
            ], dtype=np.float32),
            "layer_names": np.asarray(["layers.0", "layers.1"]),
            "query_kind_codes": np.asarray(
                [QUERY_LATENT, QUERY_ANSWER], dtype=np.uint8),
            "query_sequence_positions": np.asarray([4, 9]),
            "query_output_indices": np.asarray([-1, 3]),
            "query_predicted_token_ids": np.asarray([-1, 30]),
            "query_latent_indices": np.asarray([0, -1]),
        }
        rows = build_category_attention_csv_rows(
            data,
            FakeTokenizer(),
            sample_ordinal=2,
            dataset_ordinal=11,
            dataset_index="hr-11",
            request_id="req-1",
        )
        self.assertEqual(len(rows), 4)
        self.assertEqual(rows[0]["query_kind"], "latent")
        self.assertEqual(rows[2]["query_kind"], "answer")
        self.assertEqual(rows[2]["query_predicted_text"], "answer")
        self.assertEqual(rows[3]["layer_name"], "layers.1")
        for row in rows:
            self.assertAlmostEqual(
                sum(float(row[name]) for name in [
                    "input_text", "input_visual", "latent",
                    "generated_text", "special",
                ]),
                1.0,
                places=6,
            )

    def test_latent_topk_csv_uses_one_row_and_rank_columns(self):
        decoded = [{
            "latent_index": 4,
            "sequence_position": 22,
            "candidates": [{
                "rank": 1,
                "token_id": 7,
                "decoded_text": FakeTokenizer.texts[7],
                "raw_logit": 3.25,
            }, {
                "rank": 2,
                "token_id": 8,
                "decoded_text": FakeTokenizer.texts[8],
                "raw_logit": 2.5,
            }],
        }]
        rows = build_latent_topk_csv_rows(
            decoded,
            sample_ordinal=0,
            dataset_ordinal=9,
            dataset_index="hr-9",
            request_id="req-2",
            top_k=2,
        )
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["top1_text"], FakeTokenizer.texts[7])
        self.assertEqual(rows[0]["top2_token_id"], 8)
        self.assertEqual(rows[0]["top2_raw_logit"], 2.5)
        self.assertEqual(len(latent_topk_csv_fieldnames(2)), 7 + 2 * 3)

    def test_csv_writer_uses_bom_quoting_and_empty_header(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            populated = directory / "populated.csv"
            fields = ["text", "value"]
            write_csv(populated, fields, [{
                "text": FakeTokenizer.texts[7],
                "value": 1,
            }])
            raw = populated.read_bytes()
            self.assertTrue(raw.startswith(b"\xef\xbb\xbf"))
            with populated.open(
                "r", encoding="utf-8-sig", newline=""
            ) as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(rows, [{
                "text": FakeTokenizer.texts[7],
                "value": "1",
            }])

            empty = directory / "empty.csv"
            fields = category_attention_csv_fieldnames()
            write_csv(empty, fields, [])
            with empty.open("r", encoding="utf-8-sig", newline="") as handle:
                reader = csv.DictReader(handle)
                self.assertEqual(reader.fieldnames, fields)
                self.assertEqual(list(reader), [])

            empty_topk = directory / "empty_topk.csv"
            topk_fields = latent_topk_csv_fieldnames(2)
            write_csv(empty_topk, topk_fields, [])
            with empty_topk.open(
                "r", encoding="utf-8-sig", newline=""
            ) as handle:
                reader = csv.DictReader(handle)
                self.assertEqual(reader.fieldnames, topk_fields)
                self.assertEqual(list(reader), [])


if __name__ == "__main__":
    unittest.main()
