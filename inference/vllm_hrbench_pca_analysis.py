"""Run Monet on local HR-Bench 4K data and export internal vectors.

Edit the global variables in the configuration section below, then run:

    python -m inference.vllm_hrbench_pca_analysis

There is deliberately no command-line interface. The Monet runner writes
temporary float16 captures, this script performs a balanced joint PCA, and the
temporary high-dimensional vectors are removed after successful projection.
"""

from __future__ import annotations

import base64
import gc
import io
import inspect
import json
import os
import sys
import re
import shutil
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable
sys.path.insert(0, '/home/fit/renjujty/WORK/jty/Monet/')
import numpy as np
from PIL import Image
from sklearn.decomposition import PCA


# ---------------------------------------------------------------------------
# Global configuration -- edit values here; no command-line arguments are used
# ---------------------------------------------------------------------------

MODEL_PATH = "/home/fit/renjujty/WORK/jty/lmllms/monet/"
HRBENCH_DIR = "/home/fit/renjujty/WORK/jty/lmllms/hrbench/"
HRBENCH_FILE = "hr_bench_4k.parquet"
OUTPUT_DIR = "outputs/hrbench_pca"
POINT_CLOUD_VTP_FILE = "hrbench_point_cloud.vtp"
TRAJECTORY_VTP_FILE = "hrbench_trajectories.vtp"

# "sequential": START_INDEX ... START_INDEX + NUM_SAMPLES
# "random": deterministic sampling without replacement using RANDOM_SEED
SELECTION_MODE = "sequential"
START_INDEX = 0
NUM_SAMPLES = 10
RANDOM_SEED = 0

LATENT_SIZE = 10
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

# Every non-empty kind contributes exactly this many rows to PCA fitting.
# Kinds with fewer rows are deterministically sampled with replacement.
PCA_FIT_POINTS_PER_KIND = 2048
PCA_TRANSFORM_BATCH_SIZE = 8192

VTP_COMPRESSION_LEVEL = 6
CAPTURE_WAIT_SECONDS = 5
KEEP_TEMP_CAPTURE_ON_ERROR = False


KIND_NAMES = np.asarray(
    ["prompt_text", "image_feature", "latent", "response_text"]
)
REQUIRED_COLUMNS = {
    "index", "question", "answer", "category", "A", "B", "C", "D",
    "cycle_category", "image",
}
MAX_PATH_CANDIDATE_LENGTH = 4096


def replace_abs_vis_token_content(text: str) -> str:
    pattern = re.compile(
        r"(<abs_vis_token>)(.*?)(</abs_vis_token>)", flags=re.DOTALL
    )
    return pattern.sub(r"\1<latent>\3", text)


def inspect_analysis_worker(worker) -> dict[str, Any]:
    """Return recorder diagnostics from a vLLM worker via collective RPC."""
    runner = getattr(worker, "model_runner", None)
    if runner is None:
        return {
            "error": "worker has no model_runner attribute",
            "worker_type": f"{type(worker).__module__}.{type(worker).__name__}",
        }
    runner_type = type(runner)
    try:
        runner_file = str(Path(inspect.getfile(runner_type)).resolve())
    except (OSError, TypeError):
        runner_file = "<unknown>"
    capture_dir = getattr(runner, "analysis_capture_dir", None)
    return {
        "worker_type": f"{type(worker).__module__}.{type(worker).__name__}",
        "runner_type": f"{runner_type.__module__}.{runner_type.__name__}",
        "runner_file": runner_file,
        "capture_enabled": bool(
            getattr(runner, "analysis_capture_enabled", False)
        ),
        "capture_dir": str(capture_dir) if capture_dir is not None else None,
        "has_flush_all": callable(
            getattr(runner, "_analysis_flush_all_requests", None)
        ),
        "pending_request_ids": sorted(
            getattr(runner, "analysis_capture_state", {}).keys()
        ),
    }


def flush_analysis_worker(worker) -> dict[str, Any]:
    """Synchronously flush pending captures inside the vLLM worker."""
    runner = getattr(worker, "model_runner", None)
    flush_all = getattr(runner, "_analysis_flush_all_requests", None)
    if runner is None or not callable(flush_all):
        return {
            "error": "Monet analysis runner/flush method is unavailable",
            "status": inspect_analysis_worker(worker),
        }
    pending_before = sorted(runner.analysis_capture_state.keys())
    flushed = flush_all()
    return {
        "pending_before": pending_before,
        "flushed_request_ids": sorted(flushed),
        "pending_after": sorted(runner.analysis_capture_state.keys()),
    }


def validate_analysis_worker_statuses(
    statuses: list[dict[str, Any]], capture_dir: Path
) -> None:
    expected_suffix = "/inference/vllm/monet_gpu_model_runner.py"
    problems = []
    if len(statuses) != 1:
        problems.append(
            f"expected exactly one TP=1 worker, received {len(statuses)}"
        )
    expected_capture_dir = capture_dir.resolve()
    for index, status in enumerate(statuses):
        runner_file = str(status.get("runner_file", "")).replace("\\", "/")
        if status.get("error"):
            problems.append(f"worker {index}: {status['error']}")
        if not runner_file.endswith(expected_suffix):
            problems.append(
                f"worker {index}: unexpected runner file {runner_file!r}"
            )
        if not status.get("capture_enabled"):
            problems.append(f"worker {index}: recorder is disabled")
        if not status.get("has_flush_all"):
            problems.append(f"worker {index}: flush-all method is missing")
        actual_capture_dir = status.get("capture_dir")
        if actual_capture_dir is None:
            problems.append(f"worker {index}: capture directory is unset")
        else:
            try:
                if Path(actual_capture_dir).resolve() != expected_capture_dir:
                    problems.append(
                        f"worker {index}: capture directory mismatch "
                        f"({actual_capture_dir!r} != {str(expected_capture_dir)!r})"
                    )
            except OSError as exc:
                problems.append(
                    f"worker {index}: invalid capture directory: {exc}"
                )
    if problems:
        details = json.dumps(statuses, ensure_ascii=False, indent=2)
        raise RuntimeError(
            "Monet analysis worker validation failed:\n- "
            + "\n- ".join(problems)
            + "\nWorker diagnostics:\n"
            + details
            + "\nSync both vllm_hrbench_pca_analysis.py and "
            "inference/vllm/monet_gpu_model_runner.py on the GPU host."
        )


def select_sample_indices(
    total: int,
    mode: str = SELECTION_MODE,
    start_index: int = START_INDEX,
    count: int = NUM_SAMPLES,
    seed: int = RANDOM_SEED,
) -> list[int]:
    if total <= 0:
        raise ValueError("HR-Bench is empty.")
    if count <= 0:
        raise ValueError("NUM_SAMPLES must be positive.")
    if count > total:
        raise ValueError(f"NUM_SAMPLES={count} exceeds dataset size {total}.")
    if mode == "sequential":
        if start_index < 0 or start_index + count > total:
            raise ValueError(
                f"Sequential range [{start_index}, {start_index + count}) "
                f"is outside dataset size {total}."
            )
        return list(range(start_index, start_index + count))
    if mode == "random":
        rng = np.random.default_rng(seed)
        return rng.choice(total, size=count, replace=False).tolist()
    raise ValueError("SELECTION_MODE must be 'sequential' or 'random'.")


def validate_configuration() -> tuple[Path, Path, Path]:
    if TENSOR_PARALLEL_SIZE != 1:
        raise ValueError("Analysis capture currently requires TP=1.")
    if LATENT_SIZE <= 0:
        raise ValueError("LATENT_SIZE must be positive.")
    if PCA_FIT_POINTS_PER_KIND <= 0 or PCA_TRANSFORM_BATCH_SIZE <= 0:
        raise ValueError("PCA point and batch sizes must be positive.")
    if not 1 <= VTP_COMPRESSION_LEVEL <= 9:
        raise ValueError("VTP_COMPRESSION_LEVEL must be between 1 and 9.")
    try:
        from vtkmodules.vtkIOXML import vtkXMLPolyDataWriter  # noqa: F401
        from vtkmodules.util.numpy_support import numpy_to_vtk  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "ParaView VTP export requires the 'vtk' package. Install the "
            "updated requirements.txt before starting inference."
        ) from exc

    model_path = Path(MODEL_PATH).expanduser()
    dataset_path = Path(HRBENCH_DIR).expanduser() / HRBENCH_FILE
    output_path = Path(OUTPUT_DIR).expanduser()
    if not model_path.exists():
        raise FileNotFoundError(
            f"MODEL_PATH does not exist: {model_path}. Edit the global variable."
        )
    if not dataset_path.is_file():
        raise FileNotFoundError(
            f"HR-Bench parquet does not exist: {dataset_path}. "
            "Edit HRBENCH_DIR/HRBENCH_FILE."
        )
    output_path.mkdir(parents=True, exist_ok=True)
    return model_path.resolve(), dataset_path.resolve(), output_path.resolve()


def load_hrbench_rows(dataset_path: Path) -> tuple[list[dict[str, Any]], list[int]]:
    from datasets import load_dataset

    dataset = load_dataset(
        "parquet", data_files=str(dataset_path), split="train"
    )
    missing = REQUIRED_COLUMNS.difference(dataset.column_names)
    if missing:
        raise ValueError(
            "Unexpected HR-Bench schema; missing columns: "
            + ", ".join(sorted(missing))
        )
    selected = select_sample_indices(len(dataset))
    return [dict(dataset[index]) for index in selected], selected


def _open_image_path(path_value: str, dataset_dir: Path) -> Image.Image | None:
    """Open a plausible image path without leaking filesystem probe errors."""
    if not path_value or "\x00" in path_value:
        return None
    try:
        possible_path = Path(path_value).expanduser()
        if not possible_path.is_absolute():
            possible_path = dataset_dir / possible_path
        if not possible_path.is_file():
            return None
    except (OSError, RuntimeError):
        return None

    try:
        return Image.open(possible_path).convert("RGB")
    except Exception as exc:
        raise ValueError(f"Unable to open image file: {possible_path}") from exc


def _open_image_bytes(image_bytes: bytes, description: str) -> Image.Image:
    try:
        return Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as exc:
        raise ValueError(f"Unable to decode {description} as an image.") from exc


def _decode_base64_image(encoded: str) -> Image.Image:
    try:
        image_bytes = base64.b64decode(encoded, validate=False)
    except Exception as exc:
        raise ValueError("The HR-Bench image contains invalid base64 data.") from exc
    return _open_image_bytes(image_bytes, "HR-Bench base64 data")


def decode_hrbench_image(value: Any, dataset_dir: Path) -> Image.Image:
    if isinstance(value, Image.Image):
        return value.convert("RGB")
    if isinstance(value, dict):
        if value.get("bytes") is not None:
            return _open_image_bytes(value["bytes"], "HR-Bench byte data")
        if value.get("path"):
            image = _open_image_path(str(value["path"]), dataset_dir)
            if image is not None:
                return image
            raise ValueError(f"Image path does not exist: {value['path']}")
    if isinstance(value, (bytes, bytearray, memoryview)):
        return _open_image_bytes(bytes(value), "HR-Bench byte data")
    if not isinstance(value, str):
        raise TypeError(f"Unsupported HR-Bench image value: {type(value)!r}")

    value = value.strip()
    if value.startswith("data:image/"):
        if "," not in value:
            raise ValueError("Malformed image data URI: missing comma separator.")
        return _decode_base64_image(value.split(",", 1)[1])

    # HR-Bench stores large images as raw base64 strings. Never pass those
    # strings to stat(2): Linux raises ENAMETOOLONG before we can fall back.
    if len(value) <= MAX_PATH_CANDIDATE_LENGTH:
        image = _open_image_path(value, dataset_dir)
        if image is not None:
            return image
    return _decode_base64_image(value)


def build_question(row: dict[str, Any]) -> str:
    choices = "\n".join(f"({letter}) {row[letter]}" for letter in "ABCD")
    return (
        f"Question: {row['question']} The choices are listed below:\n"
        f"{choices}\nPut your final answer in \\boxed{{}}."
    )


def build_conversations(
    rows: list[dict[str, Any]], dataset_dir: Path
) -> tuple[list[list[dict[str, Any]]], list[Image.Image]]:
    conversations = []
    opened_images = []
    for row in rows:
        image = decode_hrbench_image(row["image"], dataset_dir)
        opened_images.append(image)
        conversations.append([
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": build_question(row)},
                    {"type": "image", "image": image},
                ],
            }
        ])
    return conversations, opened_images


def initialize_vllm(model_path: Path):
    # The patch must be imported only after capture-related environment
    # variables are set, and before importing vLLM itself.
    import inference.apply_vllm_monet  # noqa: F401
    from transformers import AutoProcessor
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
    processor = AutoProcessor.from_pretrained(
        str(model_path), trust_remote_code=True
    )
    return engine, sampling_params, processor


def process_messages(conversations, processor) -> list[dict[str, Any]]:
    from qwen_vl_utils import process_vision_info

    inputs = []
    for messages in conversations:
        prompt = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, _ = process_vision_info(
            messages, return_video_kwargs=False
        )
        if image_inputs and "<image>" not in prompt and "<im_start>" not in prompt:
            prompt = "<image>\n" + prompt
        inputs.append({
            "prompt": prompt,
            "multi_modal_data": {"image": image_inputs},
        })
    return inputs


def run_inference(
    model_path: Path,
    rows: list[dict[str, Any]],
    dataset_dir: Path,
    capture_dir: Path,
):
    os.environ["LATENT_SIZE"] = str(LATENT_SIZE)
    os.environ["MONET_ANALYSIS_CAPTURE_DIR"] = str(capture_dir)
    # vLLM V1 otherwise starts a separate EngineCore process. sys.modules
    # patches made by apply_vllm_monet are process-local and would be lost.
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    engine, sampling_params, processor = initialize_vllm(model_path)
    worker_statuses = engine.collective_rpc(
        inspect_analysis_worker, timeout=CAPTURE_WAIT_SECONDS
    )
    validate_analysis_worker_statuses(worker_statuses, capture_dir)
    print(
        "[Monet analysis] validated worker:\n"
        + json.dumps(worker_statuses, ensure_ascii=False, indent=2)
    )
    conversations, images = build_conversations(rows, dataset_dir)
    try:
        inputs = process_messages(conversations, processor)
        outputs = engine.generate(
            inputs, sampling_params=sampling_params, use_tqdm=True
        )
        flush_results = engine.collective_rpc(
            flush_analysis_worker, timeout=None
        )
        flush_errors = [
            result for result in flush_results if result.get("error")
        ]
        if flush_errors:
            raise RuntimeError(
                "Failed to flush Monet analysis captures:\n"
                + json.dumps(flush_errors, ensure_ascii=False, indent=2)
            )
        print(
            "[Monet analysis] capture flush complete:\n"
            + json.dumps(flush_results, ensure_ascii=False, indent=2)
        )
    finally:
        for image in images:
            image.close()
    return outputs, processor, worker_statuses


def wait_for_captures(
    capture_dir: Path,
    request_ids: Iterable[str],
    worker_statuses: list[dict[str, Any]] | None = None,
) -> dict[str, Path]:
    expected = set(request_ids)
    deadline = time.monotonic() + CAPTURE_WAIT_SECONDS
    found: dict[str, Path] = {}
    while time.monotonic() < deadline:
        for path in capture_dir.glob("capture_*.npz"):
            if path.name.endswith(".tmp.npz"):
                continue
            try:
                with np.load(path, allow_pickle=False) as data:
                    request_id = str(data["request_id"].item())
                found[request_id] = path
            except (OSError, ValueError, KeyError):
                continue
        if expected.issubset(found):
            return {request_id: found[request_id] for request_id in expected}
        time.sleep(0.25)
    missing = sorted(expected.difference(found))
    directory_entries = (
        sorted(path.name for path in capture_dir.iterdir())
        if capture_dir.is_dir() else []
    )
    raise RuntimeError(
        "Timed out waiting for runner captures. Missing request IDs: "
        + ", ".join(missing)
        + f"\nCapture directory: {capture_dir}"
        + f"\nDirectory entries: {directory_entries}"
        + "\nWorker diagnostics:\n"
        + json.dumps(worker_statuses or [], ensure_ascii=False, indent=2)
    )


def load_capture(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key].copy() for key in data.files}


def balanced_pca(
    captures: list[dict[str, Any]],
) -> tuple[PCA, list[np.ndarray], dict[str, int]]:
    rng = np.random.default_rng(RANDOM_SEED)
    fit_blocks = []
    available_counts = {}
    for kind_code, kind_name in enumerate(KIND_NAMES.tolist()):
        locations = []
        total = 0
        for capture_index, capture in enumerate(captures):
            indices = np.flatnonzero(capture["kind_codes"] == kind_code)
            if indices.size:
                locations.append((capture_index, indices))
                total += int(indices.size)
        available_counts[kind_name] = total
        if total == 0:
            continue

        chosen = rng.choice(
            total,
            size=PCA_FIT_POINTS_PER_KIND,
            replace=total < PCA_FIT_POINTS_PER_KIND,
        )
        cumulative = np.cumsum(
            [len(indices) for _, indices in locations], dtype=np.int64
        )
        location_indices = np.searchsorted(cumulative, chosen, side="right")
        previous_cumulative = np.concatenate((np.asarray([0]), cumulative[:-1]))
        hidden_size = int(captures[locations[0][0]]["vectors"].shape[1])
        fit_block = np.empty(
            (PCA_FIT_POINTS_PER_KIND, hidden_size), dtype=np.float32
        )
        for location_index, (capture_index, indices) in enumerate(locations):
            output_rows = np.flatnonzero(location_indices == location_index)
            if not output_rows.size:
                continue
            offsets = chosen[output_rows] - previous_cumulative[location_index]
            source_rows = indices[offsets]
            fit_block[output_rows] = captures[capture_index]["vectors"][
                source_rows
            ]
        fit_blocks.append(fit_block)

    if not fit_blocks or sum(block.shape[0] for block in fit_blocks) < 3:
        raise ValueError("Fewer than three captured vectors are available for PCA.")
    fit_matrix = np.concatenate(fit_blocks, axis=0)
    pca = PCA(n_components=3, svd_solver="randomized", random_state=RANDOM_SEED)
    pca.fit(fit_matrix)
    del fit_matrix, fit_blocks

    projected = []
    for capture in captures:
        vectors = capture["vectors"]
        chunks = []
        for start in range(0, len(vectors), PCA_TRANSFORM_BATCH_SIZE):
            batch = vectors[start:start + PCA_TRANSFORM_BATCH_SIZE]
            chunks.append(pca.transform(batch.astype(np.float32, copy=False)))
        projected.append(np.concatenate(chunks).astype(np.float32, copy=False))
    return pca, projected, available_counts


def decode_token(tokenizer, token_id: int) -> str:
    try:
        text = tokenizer.decode(
            [int(token_id)],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
    except TypeError:
        text = tokenizer.decode([int(token_id)], skip_special_tokens=False)
    if text:
        return text
    converted = tokenizer.convert_ids_to_tokens(int(token_id))
    return str(converted)


def assemble_points(
    captures: list[dict[str, Any]],
    projected: list[np.ndarray],
    sample_records: list[dict[str, Any]],
    tokenizer,
) -> dict[str, np.ndarray]:
    fields: dict[str, list[np.ndarray]] = {
        "coordinates": [],
        "sample_ordinal": [],
        "kind_codes": [],
        "token_ids": [],
        "sequence_positions": [],
        "generation_steps": [],
        "latent_indices": [],
        "image_feature_indices": [],
        "trajectory_steps": [],
        "token_labels": [],
    }
    for sample_ordinal, (capture, coordinates) in enumerate(
        zip(captures, projected)
    ):
        kind_codes = capture["kind_codes"].astype(np.uint8, copy=False)
        token_ids = capture["token_ids"].astype(np.int32, copy=False)
        labels = []
        image_feature_indices = np.full(
            len(coordinates), -1, dtype=np.int32
        )
        image_index = 0
        for point_index, (kind_code, token_id, latent_index) in enumerate(zip(
            kind_codes, token_ids, capture["latent_indices"]
        )):
            kind_name = KIND_NAMES[int(kind_code)]
            if kind_name == "image_feature":
                labels.append("")
                image_feature_indices[point_index] = image_index
                image_index += 1
            elif kind_name == "latent":
                labels.append(f"latent_{int(latent_index)}")
            else:
                labels.append(decode_token(tokenizer, int(token_id)))

        trajectory_steps = np.full(len(coordinates), -1, dtype=np.int32)
        trajectory_indices = np.flatnonzero(kind_codes != 1)
        trajectory_indices = trajectory_indices[np.argsort(
            capture["sequence_positions"][trajectory_indices], kind="stable"
        )]
        trajectory_steps[trajectory_indices] = np.arange(
            len(trajectory_indices), dtype=np.int32
        )

        fields["coordinates"].append(coordinates)
        fields["sample_ordinal"].append(
            np.full(len(coordinates), sample_ordinal, dtype=np.int32)
        )
        fields["kind_codes"].append(kind_codes)
        fields["token_ids"].append(token_ids)
        fields["sequence_positions"].append(capture["sequence_positions"])
        fields["generation_steps"].append(capture["generation_steps"])
        fields["latent_indices"].append(capture["latent_indices"])
        fields["image_feature_indices"].append(image_feature_indices)
        fields["trajectory_steps"].append(trajectory_steps)
        fields["token_labels"].append(np.asarray(labels, dtype=np.str_))

        counts = np.bincount(kind_codes, minlength=len(KIND_NAMES))
        sample_records[sample_ordinal]["capture_counts"] = {
            name: int(counts[index]) for index, name in enumerate(KIND_NAMES)
        }
        prompt_length = int(capture["prompt_length"].item())
        consumed_generation_steps = capture["generation_steps"][
            capture["generation_steps"] >= 0
        ]
        consumed_count = (
            int(consumed_generation_steps.max()) + 1
            if consumed_generation_steps.size else 0
        )
        sampled_ids = sample_records[sample_ordinal]["output_token_ids"]
        sample_records[sample_ordinal]["prompt_token_count"] = prompt_length
        sample_records[sample_ordinal]["consumed_output_token_count"] = consumed_count
        sample_records[sample_ordinal]["unconsumed_output_token_ids"] = sampled_ids[
            consumed_count:
        ]

    return {
        key: np.concatenate(values, axis=0) for key, values in fields.items()
    }


def save_pca_archive(
    output_path: Path,
    points: dict[str, np.ndarray],
    pca: PCA,
    sample_records: list[dict[str, Any]],
) -> None:
    np.savez_compressed(
        output_path / "joint_pca_3d.npz",
        **points,
        kind_names=KIND_NAMES,
        dataset_indices=np.asarray(
            [str(record["dataset_index"]) for record in sample_records],
            dtype=np.str_,
        ),
        request_ids=np.asarray(
            [record["request_id"] for record in sample_records], dtype=np.str_
        ),
        pca_components=pca.components_.astype(np.float32),
        pca_mean=pca.mean_.astype(np.float32),
        explained_variance=pca.explained_variance_.astype(np.float32),
        explained_variance_ratio=pca.explained_variance_ratio_.astype(np.float32),
    )


VTP_POINT_ARRAYS = {
    "sample_ordinal": "sample_ordinal",
    "kind_codes": "kind_code",
    "token_ids": "token_id",
    "sequence_positions": "sequence_position",
    "generation_steps": "generation_step",
    "latent_indices": "latent_index",
    "image_feature_indices": "image_feature_index",
    "trajectory_steps": "trajectory_step",
}


def _vtk_numeric_array(values: np.ndarray, name: str):
    from vtkmodules.util.numpy_support import numpy_to_vtk

    vtk_array = numpy_to_vtk(np.ascontiguousarray(values), deep=True)
    vtk_array.SetName(name)
    return vtk_array


def _vtk_string_array(values: Iterable[Any], name: str):
    from vtkmodules.vtkCommonCore import vtkStringArray

    vtk_array = vtkStringArray()
    vtk_array.SetName(name)
    for value in values:
        vtk_array.InsertNextValue(str(value))
    return vtk_array


def _vtk_cell_array(offsets: np.ndarray, connectivity: np.ndarray):
    from vtkmodules.vtkCommonDataModel import vtkCellArray
    from vtkmodules.util.numpy_support import numpy_to_vtkIdTypeArray

    cells = vtkCellArray()
    cells.SetData(
        numpy_to_vtkIdTypeArray(
            np.ascontiguousarray(offsets, dtype=np.int64), deep=True
        ),
        numpy_to_vtkIdTypeArray(
            np.ascontiguousarray(connectivity, dtype=np.int64), deep=True
        ),
    )
    return cells


def _add_vtp_point_data(polydata, points: dict[str, np.ndarray]) -> None:
    point_data = polydata.GetPointData()
    for source_name, output_name in VTP_POINT_ARRAYS.items():
        point_data.AddArray(_vtk_numeric_array(points[source_name], output_name))
    point_data.AddArray(_vtk_string_array(points["token_labels"], "token_label"))


def _add_vtp_field_data(
    polydata,
    sample_records: list[dict[str, Any]],
    explained_variance_ratio: np.ndarray,
) -> None:
    field_data = polydata.GetFieldData()
    field_data.AddArray(_vtk_string_array(KIND_NAMES, "kind_names"))
    field_data.AddArray(_vtk_string_array(
        (record["dataset_index"] for record in sample_records),
        "dataset_indices",
    ))
    field_data.AddArray(_vtk_string_array(
        (record["request_id"] for record in sample_records),
        "request_ids",
    ))
    field_data.AddArray(_vtk_numeric_array(
        np.asarray(explained_variance_ratio, dtype=np.float32),
        "pca_explained_variance_ratio",
    ))


def _new_vtp_polydata(coordinates: np.ndarray):
    from vtkmodules.vtkCommonCore import vtkPoints
    from vtkmodules.vtkCommonDataModel import vtkPolyData

    vtk_points = vtkPoints()
    vtk_points.SetData(_vtk_numeric_array(
        np.asarray(coordinates, dtype=np.float32), "PCA_coordinates"
    ))
    polydata = vtkPolyData()
    polydata.SetPoints(vtk_points)
    return polydata


def _write_compressed_vtp(path: Path, polydata) -> None:
    from vtkmodules.vtkIOXML import vtkXMLPolyDataWriter

    temporary_path = path.with_name(f".{path.stem}.tmp{path.suffix}")
    writer = vtkXMLPolyDataWriter()
    writer.SetFileName(str(temporary_path))
    writer.SetInputData(polydata)
    writer.SetDataModeToAppended()
    writer.EncodeAppendedDataOff()
    writer.SetCompressorTypeToZLib()
    writer.SetCompressionLevel(VTP_COMPRESSION_LEVEL)
    try:
        if writer.Write() != 1 or not temporary_path.is_file():
            raise RuntimeError(f"VTK failed to write PolyData: {path}")
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def build_vtp_outputs(
    output_path: Path,
    points: dict[str, np.ndarray],
    sample_records: list[dict[str, Any]],
    explained_variance_ratio: np.ndarray,
) -> dict[str, int]:
    coordinates = points["coordinates"]
    point_count = len(coordinates)

    point_cloud = _new_vtp_polydata(coordinates)
    point_cloud.SetVerts(_vtk_cell_array(
        np.arange(point_count + 1, dtype=np.int64),
        np.arange(point_count, dtype=np.int64),
    ))
    _add_vtp_point_data(point_cloud, points)
    _add_vtp_field_data(
        point_cloud, sample_records, explained_variance_ratio
    )

    trajectory_index_blocks = []
    line_connectivity_blocks = []
    line_sample_blocks = []
    line_step_blocks = []
    trajectory_point_offset = 0
    for sample_ordinal in range(len(sample_records)):
        indices = np.flatnonzero(
            (points["sample_ordinal"] == sample_ordinal)
            & (points["kind_codes"] != 1)
        )
        indices = indices[np.argsort(
            points["sequence_positions"][indices], kind="stable"
        )]
        trajectory_index_blocks.append(indices)
        if len(indices) > 1:
            local_indices = np.arange(
                trajectory_point_offset,
                trajectory_point_offset + len(indices),
                dtype=np.int64,
            )
            line_connectivity_blocks.append(np.column_stack((
                local_indices[:-1], local_indices[1:]
            )).reshape(-1))
            line_sample_blocks.append(np.full(
                len(indices) - 1, sample_ordinal, dtype=np.int32
            ))
            line_step_blocks.append(np.arange(
                1, len(indices), dtype=np.int32
            ))
        trajectory_point_offset += len(indices)

    trajectory_indices = (
        np.concatenate(trajectory_index_blocks)
        if trajectory_index_blocks else np.empty(0, dtype=np.int64)
    )
    trajectory_points = {
        name: values[trajectory_indices] for name, values in points.items()
    }
    line_connectivity = (
        np.concatenate(line_connectivity_blocks)
        if line_connectivity_blocks else np.empty(0, dtype=np.int64)
    )
    line_samples = (
        np.concatenate(line_sample_blocks)
        if line_sample_blocks else np.empty(0, dtype=np.int32)
    )
    line_steps = (
        np.concatenate(line_step_blocks)
        if line_step_blocks else np.empty(0, dtype=np.int32)
    )
    line_count = len(line_steps)

    trajectories = _new_vtp_polydata(trajectory_points["coordinates"])
    trajectories.SetLines(_vtk_cell_array(
        np.arange(0, 2 * line_count + 1, 2, dtype=np.int64),
        line_connectivity,
    ))
    _add_vtp_point_data(trajectories, trajectory_points)
    trajectories.GetCellData().AddArray(
        _vtk_numeric_array(line_samples, "sample_ordinal")
    )
    trajectories.GetCellData().AddArray(
        _vtk_numeric_array(line_steps, "trajectory_step")
    )
    _add_vtp_field_data(
        trajectories, sample_records, explained_variance_ratio
    )

    _write_compressed_vtp(output_path / POINT_CLOUD_VTP_FILE, point_cloud)
    _write_compressed_vtp(output_path / TRAJECTORY_VTP_FILE, trajectories)
    return {
        "point_cloud_points": point_count,
        "point_cloud_vertex_cells": point_count,
        "trajectory_points": len(trajectory_indices),
        "trajectory_line_segments": line_count,
    }


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def global_config_snapshot() -> dict[str, Any]:
    names = [
        "MODEL_PATH", "HRBENCH_DIR", "HRBENCH_FILE", "OUTPUT_DIR",
        "POINT_CLOUD_VTP_FILE", "TRAJECTORY_VTP_FILE",
        "SELECTION_MODE", "START_INDEX", "NUM_SAMPLES", "RANDOM_SEED",
        "LATENT_SIZE", "TENSOR_PARALLEL_SIZE", "GPU_MEMORY_UTILIZATION",
        "MAX_MODEL_LEN", "MAX_NUM_SEQS", "MAX_OUTPUT_TOKENS",
        "SWAP_SPACE_GB", "DTYPE", "ENABLE_CHUNKED_PREFILL",
        "ENABLE_SLEEP_MODE", "MIN_PIXELS", "MAX_PIXELS", "TEMPERATURE",
        "TOP_K", "TOP_P", "REPETITION_PENALTY", "BEST_OF", "STOP",
        "PCA_FIT_POINTS_PER_KIND", "PCA_TRANSFORM_BATCH_SIZE",
        "VTP_COMPRESSION_LEVEL",
    ]
    return {name: globals()[name] for name in names}


def main() -> None:
    model_path, dataset_path, output_path = validate_configuration()
    rows, selected_indices = load_hrbench_rows(dataset_path)
    capture_dir = Path(tempfile.mkdtemp(prefix=".monet_capture_", dir=output_path))
    succeeded = False
    try:
        outputs, processor, worker_statuses = run_inference(
            model_path, rows, dataset_path.parent, capture_dir
        )
        if len(outputs) != len(rows):
            raise RuntimeError(
                f"vLLM returned {len(outputs)} outputs for {len(rows)} inputs."
            )

        sample_records = []
        request_ids = []
        for sample_ordinal, (row, dataset_ordinal, output) in enumerate(
            zip(rows, selected_indices, outputs)
        ):
            generated = output.outputs[0]
            raw_text = generated.text
            request_id = str(output.request_id)
            request_ids.append(request_id)
            sample_records.append({
                "sample_ordinal": sample_ordinal,
                "dataset_ordinal": int(dataset_ordinal),
                "dataset_index": row["index"],
                "question": row["question"],
                "answer": row["answer"],
                "category": row["category"],
                "cycle_category": row["cycle_category"],
                "choices": {letter: row[letter] for letter in "ABCD"},
                "request_id": request_id,
                "raw_output_text": raw_text,
                "cleaned_output_text": replace_abs_vis_token_content(raw_text),
                "output_token_ids": [int(value) for value in generated.token_ids],
                "finish_reason": generated.finish_reason,
            })

        capture_paths = wait_for_captures(
            capture_dir, request_ids, worker_statuses
        )
        captures = [load_capture(capture_paths[request_id])
                    for request_id in request_ids]
        pca, projected, available_counts = balanced_pca(captures)
        points = assemble_points(
            captures, projected, sample_records, processor.tokenizer
        )
        del captures, projected
        gc.collect()
        save_pca_archive(output_path, points, pca, sample_records)
        write_jsonl(output_path / "results.jsonl", sample_records)
        vtp_statistics = build_vtp_outputs(
            output_path, points, sample_records, pca.explained_variance_ratio_
        )

        run_config = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "config": global_config_snapshot(),
            "vllm_enable_v1_multiprocessing": False,
            "validated_worker_statuses": worker_statuses,
            "selected_dataset_ordinals": selected_indices,
            "pca_available_counts": available_counts,
            "pca_balanced_fit_points_per_nonempty_kind": PCA_FIT_POINTS_PER_KIND,
            "pca_explained_variance_ratio":
                pca.explained_variance_ratio_.tolist(),
            "total_projected_points": int(len(points["coordinates"])),
            "vtp_outputs": {
                "point_cloud": POINT_CLOUD_VTP_FILE,
                "trajectories": TRAJECTORY_VTP_FILE,
                "compression": "appended binary with zlib",
                "compression_level": VTP_COMPRESSION_LEVEL,
                **vtp_statistics,
            },
            "paraview_filters": {
                "sample_selection": (
                    "Threshold Cell Data sample_ordinal to the selected sample"
                ),
                "trajectory_prefix": (
                    "Threshold Cell Data trajectory_step from 0 through k"
                ),
                "current_token": (
                    "Threshold Point Data sample_ordinal, then Point Data "
                    "trajectory_step exactly to k on the point cloud"
                ),
            },
            "note": (
                "Trajectory lines connect actual consumed input vectors in "
                "sequence order; image features are excluded from lines, "
                "and the lines are not attention paths."
            ),
        }
        (output_path / "run_config.json").write_text(
            json.dumps(run_config, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        succeeded = True
        print(f"Analysis complete: {output_path}")
        print(f"ParaView point cloud: {output_path / POINT_CLOUD_VTP_FILE}")
        print(f"ParaView trajectories: {output_path / TRAJECTORY_VTP_FILE}")
    finally:
        os.environ.pop("MONET_ANALYSIS_CAPTURE_DIR", None)
        os.environ.pop("VLLM_ENABLE_V1_MULTIPROCESSING", None)
        if succeeded or not KEEP_TEMP_CAPTURE_ON_ERROR:
            shutil.rmtree(capture_dir, ignore_errors=True)
        elif capture_dir.exists():
            print(f"Temporary captures kept for debugging: {capture_dir}")
        gc.collect()


if __name__ == "__main__":
    main()
