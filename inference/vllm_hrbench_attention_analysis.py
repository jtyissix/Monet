"""Run Monet on HR-Bench and export latent/answer attention distributions.

Edit the global variables below, then run:

    python -m inference.vllm_hrbench_attention_analysis

There is deliberately no command-line interface. Attention capture is an
analysis-only extension of the Monet vLLM runner and requires vLLM 0.10.0,
TP=1, PP=1, eager execution, and a floating-point KV cache.
"""

from __future__ import annotations

import csv
import json
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image

from inference.vllm_hrbench_pca_analysis import (
    build_question,
    decode_hrbench_image,
    replace_abs_vis_token_content,
)


# ---------------------------------------------------------------------------
# Global configuration -- edit values here; no command-line arguments are used
# ---------------------------------------------------------------------------

MODEL_PATH = "/home/fit/renjujty/WORK/jty/lmllms/monet/"
HRBENCH_DIR = "/home/fit/renjujty/WORK/jty/lmllms/hrbench/"
HRBENCH_FILE = "hr_bench_4k.parquet"
OUTPUT_DIR = "outputs/hrbench_attention"
RESULTS_FILE = "results.jsonl"
RUN_CONFIG_FILE = "run_config.json"
CATEGORY_ATTENTION_CSV_FILE = "category_attention.csv"
LATENT_TOPK_CSV_FILE = "latent_topk.csv"
ATTENTION_SUBDIR = "attention"
PLOT_SUBDIR = "plots"

# "sequential": START_INDEX ... START_INDEX + NUM_SAMPLES
# "random": deterministic sampling without replacement using RANDOM_SEED
SELECTION_MODE = "sequential"
START_INDEX = 199
NUM_SAMPLES = 1
RANDOM_SEED = 0

LATENT_SIZE = 10
LATENT_TOP_K = 20
ATTENTION_STORAGE_DTYPE = "float16"  # "float16" or "float32"
PLOT_LAYER = -1                     # Python-style layer index
PLOT_DPI = 180
PLOT_MAX_TOKEN_LABELS = 80
PLOT_FIGURE_WIDTH = 18.0
PLOT_ROW_HEIGHT = 0.38

TENSOR_PARALLEL_SIZE = 1
GPU_MEMORY_UTILIZATION = 0.80
MAX_MODEL_LEN = 16384
MAX_NUM_SEQS = 16
MAX_OUTPUT_TOKENS = 4096
SWAP_SPACE_GB = 7
DTYPE = "bfloat16"
ENABLE_CHUNKED_PREFILL = True
ENABLE_SLEEP_MODE = True
MIN_PIXELS = 256 * 28 * 28
MAX_PIXELS = 8192 * 28 * 28

# Keep the sampling behavior from inference/vllm_inference_example.py.
TEMPERATURE = 0.1
TOP_K = 50
TOP_P = 0.8
REPETITION_PENALTY = 1.01
BEST_OF = 1
STOP = None

CAPTURE_WAIT_SECONDS = 5
KEEP_TEMP_CAPTURE_ON_ERROR = False


REQUIRED_COLUMNS = {
    "index", "question", "answer", "category", "A", "B", "C", "D",
    "cycle_category", "image",
}
SOURCE_KIND_NAMES = np.asarray([
    "input_text", "input_visual", "latent", "generated_text", "special",
])
SOURCE_INPUT_TEXT = 0
SOURCE_INPUT_VISUAL = 1
SOURCE_LATENT = 2
SOURCE_GENERATED_TEXT = 3
SOURCE_SPECIAL = 4
QUERY_KIND_NAMES = np.asarray(["latent", "answer"])
QUERY_LATENT = 0
QUERY_ANSWER = 1


def select_sample_indices(
    total: int,
    mode: str,
    start_index: int,
    count: int,
    seed: int,
) -> list[int]:
    if total <= 0:
        raise ValueError("HR-Bench is empty.")
    if count <= 0 or count > total:
        raise ValueError(
            f"NUM_SAMPLES must be in [1, {total}], received {count}.")
    if mode == "sequential":
        if start_index < 0 or start_index + count > total:
            raise ValueError(
                f"Sequential range [{start_index}, {start_index + count}) "
                f"is outside dataset size {total}.")
        return list(range(start_index, start_index + count))
    if mode == "random":
        rng = np.random.default_rng(seed)
        return rng.choice(total, size=count, replace=False).tolist()
    raise ValueError("SELECTION_MODE must be 'sequential' or 'random'.")


def validate_configuration() -> tuple[Path, Path, Path]:
    if TENSOR_PARALLEL_SIZE != 1:
        raise ValueError("Attention capture requires TENSOR_PARALLEL_SIZE=1.")
    if LATENT_SIZE <= 0 or LATENT_TOP_K <= 0:
        raise ValueError("LATENT_SIZE and LATENT_TOP_K must be positive.")
    if ATTENTION_STORAGE_DTYPE not in {"float16", "float32"}:
        raise ValueError(
            "ATTENTION_STORAGE_DTYPE must be 'float16' or 'float32'.")
    if BEST_OF != 1:
        raise ValueError("Attention capture requires BEST_OF=1.")
    model_path = Path(MODEL_PATH).expanduser()
    dataset_path = Path(HRBENCH_DIR).expanduser() / HRBENCH_FILE
    output_path = Path(OUTPUT_DIR).expanduser()
    if not model_path.exists():
        raise FileNotFoundError(
            f"MODEL_PATH does not exist: {model_path}. Edit the global value.")
    if not dataset_path.is_file():
        raise FileNotFoundError(
            f"HR-Bench parquet does not exist: {dataset_path}.")
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / ATTENTION_SUBDIR).mkdir(exist_ok=True)
    (output_path / PLOT_SUBDIR).mkdir(exist_ok=True)
    return model_path.resolve(), dataset_path.resolve(), output_path.resolve()


def load_hrbench_rows(dataset_path: Path) -> tuple[list[dict[str, Any]], list[int]]:
    from datasets import load_dataset

    dataset = load_dataset(
        "parquet", data_files=str(dataset_path), split="train")
    missing = REQUIRED_COLUMNS.difference(dataset.column_names)
    if missing:
        raise ValueError(
            "Unexpected HR-Bench schema; missing: "
            + ", ".join(sorted(missing)))
    selected = select_sample_indices(
        len(dataset), SELECTION_MODE, START_INDEX, NUM_SAMPLES, RANDOM_SEED)
    return [dict(dataset[index]) for index in selected], selected


def build_conversations(
    rows: list[dict[str, Any]], dataset_dir: Path
) -> tuple[list[list[dict[str, Any]]], list[Image.Image]]:
    conversations: list[list[dict[str, Any]]] = []
    opened_images = []
    for row in rows:
        image = decode_hrbench_image(row["image"], dataset_dir)
        opened_images.append(image)
        conversations.append([{
            "role": "user",
            "content": [
                {"type": "text", "text": build_question(row)},
                {"type": "image", "image": image},
            ],
        }])
    return conversations, opened_images


def process_messages(conversations, processor) -> list[dict[str, Any]]:
    from qwen_vl_utils import process_vision_info

    inputs = []
    for messages in conversations:
        prompt = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        image_inputs, _ = process_vision_info(
            messages, return_video_kwargs=False)
        if image_inputs and "<image>" not in prompt and "<im_start>" not in prompt:
            prompt = "<image>\n" + prompt
        inputs.append({
            "prompt": prompt,
            "multi_modal_data": {"image": image_inputs},
        })
    return inputs


def inspect_attention_worker(worker) -> dict[str, Any]:
    runner = getattr(worker, "model_runner", None)
    if runner is None:
        return {"error": "worker has no model_runner"}
    return {
        "runner_type": f"{type(runner).__module__}.{type(runner).__name__}",
        "capture_enabled": bool(
            getattr(runner, "attention_capture_enabled", False)),
        "capture_dir": str(getattr(runner, "attention_capture_dir", None)),
        "layer_names": list(getattr(runner, "attention_layer_names", [])),
        "top_k": int(getattr(runner, "attention_top_k", -1)),
        "storage_dtype": str(
            getattr(runner, "attention_storage_dtype", "<missing>")),
        "has_flush_all": callable(
            getattr(runner, "_attention_flush_all_requests", None)),
        "pending_request_ids": sorted(
            getattr(runner, "attention_capture_state", {}).keys()),
    }


def flush_attention_worker(worker) -> dict[str, Any]:
    runner = getattr(worker, "model_runner", None)
    flush = getattr(runner, "_attention_flush_all_requests", None)
    if runner is None or not callable(flush):
        return {"error": "Monet attention flush method is unavailable"}
    pending_before = sorted(runner.attention_capture_state.keys())
    flushed = flush()
    return {
        "pending_before": pending_before,
        "flushed_request_ids": sorted(flushed),
        "pending_after": sorted(runner.attention_capture_state.keys()),
    }


def validate_worker_status(statuses: list[dict[str, Any]], capture_dir: Path) -> None:
    problems = []
    if len(statuses) != 1:
        problems.append(f"expected exactly one worker, received {len(statuses)}")
    for status in statuses:
        if status.get("error"):
            problems.append(status["error"])
        if not status.get("capture_enabled"):
            problems.append("attention capture is disabled")
        if not status.get("has_flush_all"):
            problems.append("attention flush method is missing")
        if not status.get("layer_names"):
            problems.append("no decoder attention hooks were installed")
        if status.get("top_k") != LATENT_TOP_K:
            problems.append("runner LATENT_TOP_K does not match script")
        try:
            if Path(status.get("capture_dir", "")).resolve() != capture_dir.resolve():
                problems.append("runner capture directory does not match")
        except OSError:
            problems.append("runner capture directory is invalid")
    if problems:
        raise RuntimeError(
            "Monet attention worker validation failed:\n- "
            + "\n- ".join(problems)
            + "\nDiagnostics:\n"
            + json.dumps(statuses, ensure_ascii=False, indent=2))


def initialize_vllm(model_path: Path, capture_dir: Path):
    # The processor is deliberately loaded first so all special token IDs can
    # be passed to the worker before the process-local runner patch is imported.
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(
        str(model_path), trust_remote_code=True)
    tokenizer = processor.tokenizer
    special_ids = sorted({int(value) for value in tokenizer.all_special_ids})
    os.environ["LATENT_SIZE"] = str(LATENT_SIZE)
    os.environ["MONET_ATTENTION_CAPTURE"] = "1"
    os.environ["MONET_ATTENTION_CAPTURE_DIR"] = str(capture_dir)
    os.environ["MONET_ATTENTION_TOP_K"] = str(LATENT_TOP_K)
    os.environ["MONET_ATTENTION_STORAGE_DTYPE"] = ATTENTION_STORAGE_DTYPE
    os.environ["MONET_ATTENTION_SPECIAL_TOKEN_IDS"] = ",".join(
        str(value) for value in special_ids)
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

    import inference.apply_vllm_monet  # noqa: F401
    from vllm import LLM, SamplingParams

    engine = LLM(
        model=str(model_path),
        max_model_len=MAX_MODEL_LEN,
        max_num_seqs=MAX_NUM_SEQS,
        tensor_parallel_size=TENSOR_PARALLEL_SIZE,
        trust_remote_code=True,
        seed=RANDOM_SEED,
        swap_space=SWAP_SPACE_GB,
        gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
        enforce_eager=True,
        distributed_executor_backend=None,
        dtype=DTYPE,
        mm_processor_kwargs={
            "min_pixels": MIN_PIXELS,
            "max_pixels": MAX_PIXELS,
        },
        enable_sleep_mode=ENABLE_SLEEP_MODE,
        enable_chunked_prefill=ENABLE_CHUNKED_PREFILL,
    )
    sampling_params = SamplingParams(
        temperature=TEMPERATURE,
        top_k=TOP_K,
        top_p=TOP_P,
        repetition_penalty=REPETITION_PENALTY,
        max_tokens=MAX_OUTPUT_TOKENS,
        n=BEST_OF,
        stop=STOP,
        skip_special_tokens=False,
        seed=RANDOM_SEED if TEMPERATURE == 0 else None,
    )
    return engine, sampling_params, processor


def run_inference(
    model_path: Path,
    rows: list[dict[str, Any]],
    dataset_dir: Path,
    capture_dir: Path,
):
    engine, sampling_params, processor = initialize_vllm(
        model_path, capture_dir)
    statuses = engine.collective_rpc(
        inspect_attention_worker, timeout=CAPTURE_WAIT_SECONDS)
    validate_worker_status(statuses, capture_dir)
    conversations, images = build_conversations(rows, dataset_dir)
    try:
        inputs = process_messages(conversations, processor)
        outputs = engine.generate(
            inputs, sampling_params=sampling_params, use_tqdm=True)
        flush_results = engine.collective_rpc(
            flush_attention_worker, timeout=None)
        errors = [result for result in flush_results if result.get("error")]
        if errors:
            raise RuntimeError(
                "Failed to flush attention captures:\n"
                + json.dumps(errors, ensure_ascii=False, indent=2))
    finally:
        for image in images:
            image.close()
    return outputs, processor.tokenizer, statuses, flush_results


def wait_for_capture_manifests(
    capture_dir: Path, request_ids: Iterable[str]
) -> dict[str, Path]:
    expected = set(request_ids)
    found: dict[str, Path] = {}
    for path in capture_dir.glob("attention_*.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            found[str(payload["request_id"])] = path
        except (OSError, ValueError, KeyError):
            continue
    missing = sorted(expected.difference(found))
    if missing:
        raise RuntimeError(
            "Missing attention manifests for request IDs: "
            + ", ".join(missing))
    return {request_id: found[request_id] for request_id in expected}


def classify_source_positions(
    source_positions: np.ndarray,
    prompt_length: int,
    image_positions: set[int],
    latent_positions: set[int],
    prompt_token_ids: list[int],
    generated_token_ids: list[int],
    special_token_ids: set[int],
) -> tuple[np.ndarray, np.ndarray]:
    kinds = np.empty(len(source_positions), dtype=np.uint8)
    token_ids = np.full(len(source_positions), -1, dtype=np.int32)
    for index, position_value in enumerate(source_positions):
        position = int(position_value)
        if position < prompt_length:
            token_id = int(prompt_token_ids[position])
        else:
            output_index = position - prompt_length
            token_id = (
                int(generated_token_ids[output_index])
                if 0 <= output_index < len(generated_token_ids) else -1
            )
        token_ids[index] = token_id
        if position in latent_positions:
            kinds[index] = SOURCE_LATENT
        elif position in image_positions:
            kinds[index] = SOURCE_INPUT_VISUAL
        elif position < prompt_length:
            kinds[index] = (
                SOURCE_SPECIAL if token_id in special_token_ids
                else SOURCE_INPUT_TEXT)
        elif token_id in special_token_ids:
            kinds[index] = SOURCE_SPECIAL
        else:
            kinds[index] = SOURCE_GENERATED_TEXT
    return kinds, token_ids


def normalize_attention_groups(
    raw: np.ndarray,
    source_kinds: np.ndarray,
    target_kind_codes: tuple[int, ...],
) -> np.ndarray:
    normalized = np.zeros_like(raw, dtype=np.float32)
    mask = np.isin(source_kinds, target_kind_codes)
    if not mask.any():
        return normalized
    denominator = raw[:, mask].sum(axis=1, keepdims=True, dtype=np.float32)
    valid = denominator[:, 0] > 0
    normalized[np.ix_(valid, mask)] = (
        raw[np.ix_(valid, mask)].astype(np.float32)
        / denominator[valid]
    )
    return normalized


def select_final_answer_token_indices(
    generated_token_ids: list[int],
    latent_start_id: int,
    latent_end_id: int,
    special_token_ids: set[int],
) -> tuple[list[int], bool]:
    """Return readable output indices after the final latent segment.

    Starting a later latent segment discards earlier answer candidates. With
    no latent segment, all non-special output tokens are returned.
    """
    answer_indices: list[int] = []
    saw_latent = False
    in_latent = False
    for output_index, token_id in enumerate(generated_token_ids):
        if token_id == latent_start_id:
            saw_latent = True
            in_latent = True
            answer_indices.clear()
        elif token_id == latent_end_id:
            in_latent = False
        elif not in_latent and token_id not in special_token_ids:
            answer_indices.append(output_index)
    return answer_indices, not saw_latent


def _read_record_matrix(
    spool: np.memmap, record: dict[str, Any], dtype: np.dtype
) -> np.ndarray:
    layer_count = int(record["layer_count"])
    source_count = int(record["source_count"])
    offset = int(record["offset"])
    count = layer_count * source_count
    return np.asarray(spool[offset:offset + count]).reshape(
        layer_count, source_count)


def assemble_sample_archive(
    manifest: dict[str, Any], capture_dir: Path
) -> dict[str, np.ndarray]:
    dtype = np.dtype(manifest["storage_dtype"])
    layer_names = list(manifest["layer_names"])
    layer_count = len(layer_names)
    if layer_count <= 0:
        raise RuntimeError("Capture manifest contains no decoder layers.")
    image_positions = set(map(int, manifest["image_positions"]))
    latent_positions = set(map(int, manifest["latent_positions"]))
    special_ids = set(map(int, manifest["special_token_ids"]))
    prompt_ids = list(map(int, manifest["prompt_token_ids"]))
    generated_ids = list(map(int, manifest["generated_token_ids"]))
    prompt_length = int(manifest["prompt_length"])

    query_source_offsets = [0]
    query_kinds = []
    query_positions = []
    query_output_indices = []
    query_predicted_ids = []
    query_latent_indices = []
    source_positions_all = []
    source_kinds_all = []
    source_token_ids_all = []
    raw_blocks = []
    normalized_blocks = []
    category_mass = []

    streams = [
        ("latent", QUERY_LATENT, (SOURCE_INPUT_TEXT, SOURCE_INPUT_VISUAL)),
        ("answer", QUERY_ANSWER,
         (SOURCE_INPUT_TEXT, SOURCE_INPUT_VISUAL, SOURCE_LATENT)),
    ]
    for stream, query_kind, normalization_kinds in streams:
        spool_path = capture_dir / manifest[f"{stream}_spool"]
        records = manifest[f"{stream}_records"]
        expected_values = sum(
            int(record["layer_count"]) * int(record["source_count"])
            for record in records)
        actual_values = spool_path.stat().st_size // dtype.itemsize
        if actual_values != expected_values:
            raise RuntimeError(
                f"{stream} spool size mismatch: {actual_values} != "
                f"{expected_values}")
        spool = (
            np.memmap(spool_path, mode="r", dtype=dtype)
            if expected_values else None
        )
        try:
            for record in records:
                if int(record["layer_count"]) != layer_count:
                    raise RuntimeError("Layer count changed within a capture.")
                source_count = int(record["source_count"])
                source_positions = np.arange(source_count, dtype=np.int32)
                source_kinds, source_token_ids = classify_source_positions(
                    source_positions,
                    prompt_length,
                    image_positions,
                    latent_positions,
                    prompt_ids,
                    generated_ids,
                    special_ids,
                )
                assert spool is not None
                raw = _read_record_matrix(spool, record, dtype)
                sums = raw.astype(np.float32).sum(axis=1)
                tolerance = 5e-3 if dtype == np.dtype("float16") else 1e-4
                if not np.allclose(sums, 1.0, atol=tolerance, rtol=tolerance):
                    raise RuntimeError(
                        "Raw attention does not sum to one for query at "
                        f"position {record['query_sequence_position']}.")
                normalized = normalize_attention_groups(
                    raw, source_kinds, normalization_kinds)
                masses = np.stack([
                    raw[:, source_kinds == kind].astype(np.float32).sum(axis=1)
                    for kind in range(len(SOURCE_KIND_NAMES))
                ], axis=-1)
                raw_blocks.append(raw)
                normalized_blocks.append(
                    normalized.astype(dtype, copy=False))
                category_mass.append(masses)
                source_positions_all.append(source_positions)
                source_kinds_all.append(source_kinds)
                source_token_ids_all.append(source_token_ids)
                query_source_offsets.append(
                    query_source_offsets[-1] + source_count)
                query_kinds.append(query_kind)
                query_positions.append(int(record["query_sequence_position"]))
                query_output_indices.append(int(record.get("output_index", -1)))
                query_predicted_ids.append(
                    int(record.get("predicted_token_id", -1)))
                query_latent_indices.append(int(record.get("latent_index", -1)))
        finally:
            if spool is not None:
                del spool

    total_sources = query_source_offsets[-1]
    if raw_blocks:
        raw_attention = np.concatenate(raw_blocks, axis=1)
        normalized_attention = np.concatenate(normalized_blocks, axis=1)
        mass_array = np.stack(category_mass, axis=0).astype(np.float32)
    else:
        raw_attention = np.empty((layer_count, 0), dtype=dtype)
        normalized_attention = np.empty((layer_count, 0), dtype=dtype)
        mass_array = np.empty(
            (0, layer_count, len(SOURCE_KIND_NAMES)), dtype=np.float32)
    assert raw_attention.shape == (layer_count, total_sources)

    topk_records = manifest["latent_topk"]
    if topk_records:
        topk_token_ids = np.asarray(
            [record["token_ids"] for record in topk_records], dtype=np.int32)
        topk_logits = np.asarray(
            [record["logits"] for record in topk_records], dtype=np.float32)
    else:
        topk_token_ids = np.empty((0, LATENT_TOP_K), dtype=np.int32)
        topk_logits = np.empty((0, LATENT_TOP_K), dtype=np.float32)
    if len(topk_records) != len(manifest["latent_records"]):
        raise RuntimeError(
            "Every latent query must have exactly one output-head top-k row.")

    def concatenate(values, output_dtype):
        return (
            np.concatenate(values).astype(output_dtype, copy=False)
            if values else np.empty(0, dtype=output_dtype)
        )
    return {
        "raw_attention": raw_attention,
        "group_normalized_attention": normalized_attention,
        "query_source_offsets": np.asarray(query_source_offsets, dtype=np.int64),
        "query_kind_codes": np.asarray(query_kinds, dtype=np.uint8),
        "query_sequence_positions": np.asarray(query_positions, dtype=np.int32),
        "query_output_indices": np.asarray(query_output_indices, dtype=np.int32),
        "query_predicted_token_ids": np.asarray(
            query_predicted_ids, dtype=np.int32),
        "query_latent_indices": np.asarray(query_latent_indices, dtype=np.int32),
        "source_sequence_positions": concatenate(
            source_positions_all, np.int32),
        "source_kind_codes": concatenate(source_kinds_all, np.uint8),
        "source_token_ids": concatenate(source_token_ids_all, np.int32),
        "category_attention_mass": mass_array,
        "source_kind_names": SOURCE_KIND_NAMES,
        "query_kind_names": QUERY_KIND_NAMES,
        "layer_names": np.asarray(layer_names, dtype=np.str_),
        "latent_topk_token_ids": topk_token_ids,
        "latent_topk_logits": topk_logits,
        "latent_topk_sequence_positions": np.asarray([
            record["query_sequence_position"] for record in topk_records
        ], dtype=np.int32),
        "latent_topk_indices": np.asarray([
            record["latent_index"] for record in topk_records
        ], dtype=np.int32),
        "no_latent_fallback": np.asarray(
            bool(manifest["no_latent_fallback"])),
    }


def token_piece(tokenizer, token_id: int) -> str:
    if token_id < 0:
        return "<unknown>"
    piece = tokenizer.convert_ids_to_tokens(token_id)
    return str(piece).replace("\n", "\\n")


def source_labels(data: dict[str, np.ndarray], tokenizer) -> list[str]:
    labels = []
    for position, kind, token_id in zip(
        data["source_sequence_positions"],
        data["source_kind_codes"],
        data["source_token_ids"],
    ):
        kind = int(kind)
        if kind == SOURCE_INPUT_VISUAL:
            label = f"visual@{int(position)}"
        elif kind == SOURCE_LATENT:
            label = f"latent@{int(position)}"
        else:
            label = f"{token_piece(tokenizer, int(token_id))}@{int(position)}"
        labels.append(label)
    return labels


def query_labels(data: dict[str, np.ndarray], tokenizer) -> list[str]:
    labels = []
    for kind, position, predicted, latent_index in zip(
        data["query_kind_codes"],
        data["query_sequence_positions"],
        data["query_predicted_token_ids"],
        data["query_latent_indices"],
    ):
        if int(kind) == QUERY_LATENT:
            labels.append(f"latent[{int(latent_index)}]@{int(position)}")
        else:
            labels.append(
                f"{token_piece(tokenizer, int(predicted))}@{int(position)}")
    return labels


def _ragged_heatmap(
    data: dict[str, np.ndarray],
    tokenizer,
    query_kind: int,
    layer_index: int,
) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    query_indices = np.flatnonzero(data["query_kind_codes"] == query_kind)
    if len(query_indices) == 0:
        return (
            np.empty((0, 0), dtype=np.float32),
            np.empty((0, 0), dtype=np.float32), [], [],
        )
    offsets = data["query_source_offsets"]
    width = max(int(offsets[index + 1] - offsets[index]) for index in query_indices)
    raw = np.full((len(query_indices), width), np.nan, dtype=np.float32)
    normalized = np.full_like(raw, np.nan)
    all_source_labels = source_labels(data, tokenizer)
    selected_source_labels = [""] * width
    all_query_labels = query_labels(data, tokenizer)
    selected_query_labels = []
    for output_row, query_index in enumerate(query_indices):
        start, end = map(int, offsets[query_index:query_index + 2])
        length = end - start
        raw[output_row, :length] = data["raw_attention"][
            layer_index, start:end]
        normalized[output_row, :length] = data[
            "group_normalized_attention"][layer_index, start:end]
        if length == width:
            selected_source_labels = all_source_labels[start:end]
        selected_query_labels.append(all_query_labels[query_index])
    return raw, normalized, selected_source_labels, selected_query_labels


def _tick_positions(count: int) -> np.ndarray:
    if count <= PLOT_MAX_TOKEN_LABELS:
        return np.arange(count)
    return np.unique(np.linspace(
        0, count - 1, PLOT_MAX_TOKEN_LABELS, dtype=int))


def plot_attention_heatmap(
    data: dict[str, np.ndarray],
    tokenizer,
    query_kind: int,
    plot_path: Path,
) -> None:
    import matplotlib.pyplot as plt

    layer_count = len(data["layer_names"])
    layer_index = PLOT_LAYER if PLOT_LAYER >= 0 else layer_count + PLOT_LAYER
    if layer_index < 0 or layer_index >= layer_count:
        raise ValueError(
            f"PLOT_LAYER={PLOT_LAYER} is invalid for {layer_count} layers.")
    raw, normalized, xlabels, ylabels = _ragged_heatmap(
        data, tokenizer, query_kind, layer_index)
    if raw.size == 0:
        return
    height = max(4.5, min(30.0, len(ylabels) * PLOT_ROW_HEIGHT + 2.5))
    fig, axes = plt.subplots(
        2, 1, figsize=(PLOT_FIGURE_WIDTH, height * 2), constrained_layout=True)
    titles = ["raw head-mean attention", "target-group normalized attention"]
    for axis, matrix, title in zip(axes, [raw, normalized], titles):
        image = axis.imshow(matrix, aspect="auto", interpolation="nearest")
        ticks = _tick_positions(len(xlabels))
        axis.set_xticks(ticks, [xlabels[i] for i in ticks], rotation=90, fontsize=6)
        axis.set_yticks(np.arange(len(ylabels)), ylabels, fontsize=7)
        axis.set_title(
            f"{QUERY_KIND_NAMES[query_kind]} — layer {layer_index} — {title}")
        axis.set_xlabel("causally visible source position")
        axis.set_ylabel("query")
        fig.colorbar(image, ax=axis, fraction=0.02, pad=0.01)
    fig.savefig(plot_path, dpi=PLOT_DPI)
    plt.close(fig)


def plot_category_attention(
    data: dict[str, np.ndarray], tokenizer, plot_path: Path
) -> None:
    import matplotlib.pyplot as plt

    masses = data["category_attention_mass"]
    if len(masses) == 0:
        return
    layer_count = masses.shape[1]
    layer_index = PLOT_LAYER if PLOT_LAYER >= 0 else layer_count + PLOT_LAYER
    labels = query_labels(data, tokenizer)
    fig, axes = plt.subplots(
        2, 2, figsize=(PLOT_FIGURE_WIDTH, 11), constrained_layout=True)
    for column, query_kind in enumerate((QUERY_LATENT, QUERY_ANSWER)):
        indices = np.flatnonzero(data["query_kind_codes"] == query_kind)
        title = str(QUERY_KIND_NAMES[query_kind])
        if len(indices) == 0:
            axes[0, column].axis("off")
            axes[1, column].axis("off")
            continue
        query_matrix = masses[indices, layer_index, :]
        image = axes[0, column].imshow(
            query_matrix, aspect="auto", vmin=0.0, vmax=1.0)
        axes[0, column].set_xticks(
            np.arange(len(SOURCE_KIND_NAMES)), SOURCE_KIND_NAMES, rotation=30)
        axes[0, column].set_yticks(
            np.arange(len(indices)), [labels[index] for index in indices], fontsize=7)
        axes[0, column].set_title(f"{title}: per query at layer {layer_index}")
        fig.colorbar(image, ax=axes[0, column], fraction=0.03, pad=0.02)
        layer_matrix = masses[indices].mean(axis=0)
        image = axes[1, column].imshow(
            layer_matrix, aspect="auto", vmin=0.0, vmax=1.0)
        axes[1, column].set_xticks(
            np.arange(len(SOURCE_KIND_NAMES)), SOURCE_KIND_NAMES, rotation=30)
        axes[1, column].set_yticks(np.arange(layer_count))
        axes[1, column].set_title(f"{title}: query-mean mass by layer")
        axes[1, column].set_ylabel("decoder layer")
        fig.colorbar(image, ax=axes[1, column], fraction=0.03, pad=0.02)
    fig.savefig(plot_path, dpi=PLOT_DPI)
    plt.close(fig)


def decode_topk(data: dict[str, np.ndarray], tokenizer) -> list[dict[str, Any]]:
    records = []
    for latent_index, position, token_ids, logits in zip(
        data["latent_topk_indices"],
        data["latent_topk_sequence_positions"],
        data["latent_topk_token_ids"],
        data["latent_topk_logits"],
    ):
        candidates = []
        for rank, (token_id, logit) in enumerate(zip(token_ids, logits), start=1):
            token_id = int(token_id)
            candidates.append({
                "rank": rank,
                "token_id": token_id,
                "token_piece": str(tokenizer.convert_ids_to_tokens(token_id)),
                "decoded_text": tokenizer.decode(
                    [token_id], skip_special_tokens=False),
                "raw_logit": float(logit),
            })
        records.append({
            "latent_index": int(latent_index),
            "sequence_position": int(position),
            "candidates": candidates,
        })
    return records


def category_attention_csv_fieldnames() -> list[str]:
    return [
        "sample_ordinal", "dataset_ordinal", "dataset_index", "request_id",
        "query_ordinal", "query_kind", "query_sequence_position",
        "query_output_index", "query_predicted_token_id",
        "query_predicted_text", "query_latent_index", "layer_index",
        "layer_name", *SOURCE_KIND_NAMES.tolist(),
    ]


def latent_topk_csv_fieldnames(top_k: int = LATENT_TOP_K) -> list[str]:
    fields = [
        "sample_ordinal", "dataset_ordinal", "dataset_index", "request_id",
        "latent_ordinal", "latent_index", "sequence_position",
    ]
    for rank in range(1, top_k + 1):
        fields.extend([
            f"top{rank}_text",
            f"top{rank}_token_id",
            f"top{rank}_raw_logit",
        ])
    return fields


def build_category_attention_csv_rows(
    data: dict[str, np.ndarray],
    tokenizer,
    *,
    sample_ordinal: int,
    dataset_ordinal: int,
    dataset_index: Any,
    request_id: str,
) -> list[dict[str, Any]]:
    masses = data["category_attention_mass"]
    layer_names = data["layer_names"]
    if masses.shape != (
        len(data["query_kind_codes"]),
        len(layer_names),
        len(SOURCE_KIND_NAMES),
    ):
        raise ValueError(
            "Unexpected category_attention_mass shape: "
            f"{masses.shape}")
    rows = []
    for query_ordinal in range(len(data["query_kind_codes"])):
        query_kind = int(data["query_kind_codes"][query_ordinal])
        predicted_token_id = int(
            data["query_predicted_token_ids"][query_ordinal])
        predicted_text = (
            tokenizer.decode([predicted_token_id], skip_special_tokens=False)
            if predicted_token_id >= 0 else ""
        )
        common = {
            "sample_ordinal": sample_ordinal,
            "dataset_ordinal": int(dataset_ordinal),
            "dataset_index": dataset_index,
            "request_id": request_id,
            "query_ordinal": query_ordinal,
            "query_kind": str(QUERY_KIND_NAMES[query_kind]),
            "query_sequence_position": int(
                data["query_sequence_positions"][query_ordinal]),
            "query_output_index": int(
                data["query_output_indices"][query_ordinal]),
            "query_predicted_token_id": predicted_token_id,
            "query_predicted_text": predicted_text,
            "query_latent_index": int(
                data["query_latent_indices"][query_ordinal]),
        }
        for layer_index, layer_name in enumerate(layer_names):
            row = {
                **common,
                "layer_index": layer_index,
                "layer_name": str(layer_name),
            }
            for kind_index, kind_name in enumerate(SOURCE_KIND_NAMES):
                row[str(kind_name)] = float(
                    masses[query_ordinal, layer_index, kind_index])
            rows.append(row)
    return rows


def build_latent_topk_csv_rows(
    decoded_topk: list[dict[str, Any]],
    *,
    sample_ordinal: int,
    dataset_ordinal: int,
    dataset_index: Any,
    request_id: str,
    top_k: int = LATENT_TOP_K,
) -> list[dict[str, Any]]:
    rows = []
    for latent_ordinal, latent_record in enumerate(decoded_topk):
        candidates = latent_record["candidates"]
        if len(candidates) != top_k:
            raise ValueError(
                f"Expected {top_k} top-k candidates, received "
                f"{len(candidates)}.")
        row = {
            "sample_ordinal": sample_ordinal,
            "dataset_ordinal": int(dataset_ordinal),
            "dataset_index": dataset_index,
            "request_id": request_id,
            "latent_ordinal": latent_ordinal,
            "latent_index": int(latent_record["latent_index"]),
            "sequence_position": int(latent_record["sequence_position"]),
        }
        for rank, candidate in enumerate(candidates, start=1):
            if int(candidate["rank"]) != rank:
                raise ValueError("Latent top-k ranks are not contiguous.")
            row[f"top{rank}_text"] = candidate["decoded_text"]
            row[f"top{rank}_token_id"] = int(candidate["token_id"])
            row[f"top{rank}_raw_logit"] = float(candidate["raw_logit"])
        rows.append(row)
    return rows


def write_csv(
    path: Path, fieldnames: list[str], rows: Iterable[dict[str, Any]]
) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=fieldnames, quoting=csv.QUOTE_MINIMAL)
        writer.writeheader()
        writer.writerows(rows)


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def global_config_snapshot() -> dict[str, Any]:
    names = [
        "MODEL_PATH", "HRBENCH_DIR", "HRBENCH_FILE", "OUTPUT_DIR",
        "RESULTS_FILE", "RUN_CONFIG_FILE", "CATEGORY_ATTENTION_CSV_FILE",
        "LATENT_TOPK_CSV_FILE", "ATTENTION_SUBDIR", "PLOT_SUBDIR",
        "SELECTION_MODE", "START_INDEX", "NUM_SAMPLES", "RANDOM_SEED",
        "LATENT_SIZE", "LATENT_TOP_K", "ATTENTION_STORAGE_DTYPE",
        "PLOT_LAYER", "PLOT_DPI", "PLOT_MAX_TOKEN_LABELS",
        "TENSOR_PARALLEL_SIZE", "GPU_MEMORY_UTILIZATION", "MAX_MODEL_LEN",
        "MAX_NUM_SEQS", "MAX_OUTPUT_TOKENS", "SWAP_SPACE_GB", "DTYPE",
        "ENABLE_CHUNKED_PREFILL", "ENABLE_SLEEP_MODE", "MIN_PIXELS",
        "MAX_PIXELS", "TEMPERATURE", "TOP_K", "TOP_P",
        "REPETITION_PENALTY", "BEST_OF", "STOP",
    ]
    return {name: globals()[name] for name in names}


def main() -> None:
    model_path, dataset_path, output_path = validate_configuration()
    rows, selected_indices = load_hrbench_rows(dataset_path)
    capture_dir = Path(tempfile.mkdtemp(
        prefix=".monet_attention_", dir=output_path))
    succeeded = False
    try:
        outputs, tokenizer, statuses, flush_results = run_inference(
            model_path, rows, dataset_path.parent, capture_dir)
        if len(outputs) != len(rows):
            raise RuntimeError(
                f"vLLM returned {len(outputs)} outputs for {len(rows)} rows.")
        request_ids = [str(output.request_id) for output in outputs]
        manifests = wait_for_capture_manifests(capture_dir, request_ids)
        results = []
        capture_statistics = []
        category_attention_csv_rows = []
        latent_topk_csv_rows = []
        for sample_ordinal, (row, dataset_ordinal, output) in enumerate(
            zip(rows, selected_indices, outputs)
        ):
            generated = output.outputs[0]
            request_id = str(output.request_id)
            manifest = json.loads(
                manifests[request_id].read_text(encoding="utf-8"))
            output_ids = [int(value) for value in generated.token_ids]
            captured_ids = list(map(int, manifest["generated_token_ids"]))
            if captured_ids != output_ids:
                raise RuntimeError(
                    f"Captured output IDs differ for request {request_id}.")
            expected_answer_indices, expected_fallback = (
                select_final_answer_token_indices(
                    captured_ids,
                    latent_start_id=int(os.environ["LATENT_START_ID"]),
                    latent_end_id=int(os.environ["LATENT_END_ID"]),
                    special_token_ids=set(map(
                        int, manifest["special_token_ids"])),
                )
            )
            captured_answer_indices = [
                int(record["output_index"])
                for record in manifest["answer_records"]
            ]
            if captured_answer_indices != expected_answer_indices:
                raise RuntimeError(
                    "Runner answer-range state differs from the declared "
                    f"final-latent rule for request {request_id}: "
                    f"{captured_answer_indices} != {expected_answer_indices}")
            if bool(manifest["no_latent_fallback"]) != expected_fallback:
                raise RuntimeError(
                    f"no_latent_fallback mismatch for request {request_id}.")
            data = assemble_sample_archive(manifest, capture_dir)
            stem = f"sample_{sample_ordinal:06d}"
            archive_rel = Path(ATTENTION_SUBDIR) / f"{stem}.npz"
            np.savez_compressed(output_path / archive_rel, **data)

            latent_plot_rel = Path(PLOT_SUBDIR) / f"{stem}_latent_attention.png"
            answer_plot_rel = Path(PLOT_SUBDIR) / f"{stem}_answer_attention.png"
            category_plot_rel = Path(PLOT_SUBDIR) / f"{stem}_category_attention.png"
            plot_attention_heatmap(
                data, tokenizer, QUERY_LATENT, output_path / latent_plot_rel)
            plot_attention_heatmap(
                data, tokenizer, QUERY_ANSWER, output_path / answer_plot_rel)
            plot_category_attention(
                data, tokenizer, output_path / category_plot_rel)

            answer_mask = data["query_kind_codes"] == QUERY_ANSWER
            answer_ids = data["query_predicted_token_ids"][answer_mask].tolist()
            answer_output_indices = data["query_output_indices"][answer_mask].tolist()
            decoded_topk = decode_topk(data, tokenizer)
            category_attention_csv_rows.extend(
                build_category_attention_csv_rows(
                    data,
                    tokenizer,
                    sample_ordinal=sample_ordinal,
                    dataset_ordinal=int(dataset_ordinal),
                    dataset_index=row["index"],
                    request_id=request_id,
                )
            )
            latent_topk_csv_rows.extend(
                build_latent_topk_csv_rows(
                    decoded_topk,
                    sample_ordinal=sample_ordinal,
                    dataset_ordinal=int(dataset_ordinal),
                    dataset_index=row["index"],
                    request_id=request_id,
                )
            )
            result = {
                "sample_ordinal": sample_ordinal,
                "dataset_ordinal": int(dataset_ordinal),
                "dataset_index": row["index"],
                "question": row["question"],
                "answer": row["answer"],
                "category": row["category"],
                "cycle_category": row["cycle_category"],
                "choices": {letter: row[letter] for letter in "ABCD"},
                "request_id": request_id,
                "raw_output_text": generated.text,
                "cleaned_output_text": replace_abs_vis_token_content(
                    generated.text),
                "output_token_ids": output_ids,
                "finish_reason": generated.finish_reason,
                "no_latent_fallback": bool(data["no_latent_fallback"]),
                "answer_output_indices": answer_output_indices,
                "answer_token_ids": answer_ids,
                "answer_text": tokenizer.decode(
                    answer_ids, skip_special_tokens=False),
                "latent_output_head_topk": decoded_topk,
                "attention_archive": str(archive_rel),
                "plots": {
                    "latent_attention": (
                        str(latent_plot_rel)
                        if (output_path / latent_plot_rel).exists() else None),
                    "answer_attention": (
                        str(answer_plot_rel)
                        if (output_path / answer_plot_rel).exists() else None),
                    "category_attention": (
                        str(category_plot_rel)
                        if (output_path / category_plot_rel).exists() else None),
                },
            }
            results.append(result)
            capture_statistics.append({
                "sample_ordinal": sample_ordinal,
                "latent_queries": int(np.count_nonzero(
                    data["query_kind_codes"] == QUERY_LATENT)),
                "answer_queries": int(np.count_nonzero(answer_mask)),
                "ragged_source_entries": int(
                    data["query_source_offsets"][-1]),
            })

        write_jsonl(output_path / RESULTS_FILE, results)
        write_csv(
            output_path / CATEGORY_ATTENTION_CSV_FILE,
            category_attention_csv_fieldnames(),
            category_attention_csv_rows,
        )
        write_csv(
            output_path / LATENT_TOPK_CSV_FILE,
            latent_topk_csv_fieldnames(),
            latent_topk_csv_rows,
        )
        run_config = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "config": global_config_snapshot(),
            "selected_dataset_ordinals": selected_indices,
            "validated_worker_statuses": statuses,
            "flush_results": flush_results,
            "capture_statistics": capture_statistics,
            "schema": {
                "source_kind_names": SOURCE_KIND_NAMES.tolist(),
                "query_kind_names": QUERY_KIND_NAMES.tolist(),
                "raw_attention": (
                    "[decoder_layer, concatenated_ragged_source_entry]"),
                "group_normalized_attention": (
                    "same layout; latent targets input_text+input_visual; "
                    "answer targets input_text+input_visual+latent"),
                "category_attention_mass": (
                    "[query, decoder_layer, source_kind]"),
                "query_source_offsets": (
                    "offsets into concatenated ragged source arrays"),
                "answer_alignment": "query that generated the recorded token",
            },
            "outputs": {
                "results": RESULTS_FILE,
                "category_attention_csv": CATEGORY_ATTENTION_CSV_FILE,
                "latent_topk_csv": LATENT_TOPK_CSV_FILE,
                "attention_directory": ATTENTION_SUBDIR,
                "plot_directory": PLOT_SUBDIR,
            },
        }
        (output_path / RUN_CONFIG_FILE).write_text(
            json.dumps(run_config, ensure_ascii=False, indent=2),
            encoding="utf-8")
        succeeded = True
        print(f"Attention analysis complete: {output_path}")
    finally:
        for name in [
            "MONET_ATTENTION_CAPTURE", "MONET_ATTENTION_CAPTURE_DIR",
            "MONET_ATTENTION_TOP_K", "MONET_ATTENTION_STORAGE_DTYPE",
            "MONET_ATTENTION_SPECIAL_TOKEN_IDS",
            "VLLM_ENABLE_V1_MULTIPROCESSING",
        ]:
            os.environ.pop(name, None)
        if succeeded or not KEEP_TEMP_CAPTURE_ON_ERROR:
            shutil.rmtree(capture_dir, ignore_errors=True)
        elif capture_dir.exists():
            print(f"Temporary attention captures kept: {capture_dir}")


if __name__ == "__main__":
    main()
