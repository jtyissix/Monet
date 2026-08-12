"""Run Monet on local MME-RealWorld-Lite data and export internal vectors.

Edit the global variables in the configuration section below, then run:

    python -m inference.vllm_mme_realworld_lite_pca_analysis

There is deliberately no command-line interface. The Monet runner writes
temporary float16 image/latent captures and the complete vocabulary input
embedding table. This script projects all requested vectors with one PCA and
removes the temporary high-dimensional data after successful projection.
"""

from __future__ import annotations

import gc
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

import numpy as np
from PIL import Image
from sklearn.decomposition import PCA


# ---------------------------------------------------------------------------
# Global configuration -- edit values here; no command-line arguments are used
# ---------------------------------------------------------------------------

MONET_REPO_DIR = "/home/fit/renjujty/WORK/jty/Monet/"
MODEL_PATH = "/home/fit/renjujty/WORK/jty/lmllms/monet/"
MME_REALWORLD_DIR = (
    "/home/fit/renjujty/WORK/jty/lmllms/mmereal/extracted/data"
)
QUESTION_FILE = "MME-RealWorld-Lite.json"
IMAGE_DIR = "./imgs/"
OUTPUT_DIR = "outputs/mme_realworld_lite_pca"
JOINT_PCA_FILE = "joint_pca_3d.npz"
LATENT_TRAJECTORY_FILE = "latent_trajectories.npz"

# "sequential": START_INDEX ... START_INDEX + NUM_SAMPLES
# "random": deterministic sampling without replacement using RANDOM_SEED
SELECTION_MODE = "random"
START_INDEX = 0
NUM_SAMPLES = 1000
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

VOCAB_EMBEDDING_BATCH_SIZE = 8192
PCA_TRANSFORM_BATCH_SIZE = 8192

CAPTURE_WAIT_SECONDS = 5
KEEP_TEMP_CAPTURE_ON_ERROR = False


if MONET_REPO_DIR not in sys.path:
    sys.path.insert(0, MONET_REPO_DIR)


KIND_NAMES = np.asarray(["vocabulary_embedding", "image_feature", "latent"])
CHOICE_LABELS = "ABCDE"
REQUIRED_FIELDS = {
    "Question_id", "Image", "Text", "Answer choices", "Ground truth",
    "Task", "Subtask", "Category",
}
OFFICIAL_PROMPT_SUFFIX = (
    "Select the best answer to the above multiple-choice question based on "
    "the image. Respond with only the letter (A, B, C, D, or E) of the "
    "correct option.\nThe best answer is:"
)
CHOICE_REPAIRS_KEY = "_choice_normalization_repairs"


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
        "has_export_vocabulary": callable(
            getattr(runner, "_analysis_export_vocabulary_embeddings", None)
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


def export_vocabulary_embeddings_worker(worker) -> dict[str, Any]:
    """Export the complete input embedding table inside the GPU worker."""
    runner = getattr(worker, "model_runner", None)
    exporter = getattr(
        runner, "_analysis_export_vocabulary_embeddings", None
    )
    if runner is None or not callable(exporter):
        return {
            "error": "Monet vocabulary exporter is unavailable",
            "status": inspect_analysis_worker(worker),
        }
    try:
        return exporter(VOCAB_EMBEDDING_BATCH_SIZE)
    except Exception as exc:
        return {
            "error": f"{type(exc).__name__}: {exc}",
            "status": inspect_analysis_worker(worker),
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
        if not status.get("has_export_vocabulary"):
            problems.append(
                f"worker {index}: vocabulary exporter method is missing"
            )
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
            + "\nSync both vllm_mme_realworld_lite_pca_analysis.py and "
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
        raise ValueError("MME-RealWorld-Lite is empty.")
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


def validate_configuration() -> tuple[Path, Path, Path, Path]:
    if TENSOR_PARALLEL_SIZE != 1:
        raise ValueError("Analysis capture currently requires TP=1.")
    if LATENT_SIZE <= 0:
        raise ValueError("LATENT_SIZE must be positive.")
    if VOCAB_EMBEDDING_BATCH_SIZE <= 0 or PCA_TRANSFORM_BATCH_SIZE <= 0:
        raise ValueError("Vocabulary and PCA batch sizes must be positive.")

    monet_repo_path = Path(MONET_REPO_DIR).expanduser()
    model_path = Path(MODEL_PATH).expanduser()
    dataset_root = Path(MME_REALWORLD_DIR).expanduser()
    question_path = dataset_root / QUESTION_FILE
    image_dir = dataset_root / IMAGE_DIR
    output_path = Path(OUTPUT_DIR).expanduser()
    if not monet_repo_path.is_dir():
        raise FileNotFoundError(
            "MONET_REPO_DIR does not exist or is not a directory: "
            f"{monet_repo_path}. Edit the global variable."
        )
    if not model_path.exists():
        raise FileNotFoundError(
            f"MODEL_PATH does not exist: {model_path}. Edit the global variable."
        )
    if not dataset_root.is_dir():
        raise FileNotFoundError(
            "MME_REALWORLD_DIR does not exist or is not a directory: "
            f"{dataset_root}. Edit the global variable."
        )
    if not question_path.is_file():
        raise FileNotFoundError(
            f"MME-RealWorld question JSON does not exist: {question_path}. "
            "Edit MME_REALWORLD_DIR/QUESTION_FILE."
        )
    if not image_dir.is_dir():
        raise FileNotFoundError(
            f"MME-RealWorld image directory does not exist: {image_dir}. "
            "Edit MME_REALWORLD_DIR/IMAGE_DIR."
        )
    output_path.mkdir(parents=True, exist_ok=True)
    return (
        model_path.resolve(),
        question_path.resolve(),
        image_dir.resolve(),
        output_path.resolve(),
    )


def _normalize_answer_choices(
    choices: Any,
    dataset_ordinal: int,
    question_id: str,
) -> tuple[list[str], list[dict[str, Any]]]:
    location = f"dataset row {dataset_ordinal} ({question_id})"
    if not isinstance(choices, list):
        raise TypeError(f"{location} Answer choices must be a list.")

    parsed = []
    repairs: list[dict[str, Any]] = []
    for choice_index, choice in enumerate(choices):
        if not isinstance(choice, str) or not choice.strip():
            raise ValueError(
                f"{location} choice {choice_index} must be a non-empty string."
            )
        stripped = choice.strip()
        match = re.fullmatch(r"\(([A-E])\)\s*(.+)", stripped, flags=re.DOTALL)
        if match is None:
            raise ValueError(
                f"{location} choice {choice_index} must start with an "
                f"(A)-(E) label, got {choice!r}."
            )
        label, body = match.groups()
        body = body.strip()
        if not body:
            raise ValueError(f"{location} choice {choice_index} has an empty body.")
        parsed.append((label, body))
        if not re.match(r"^\([A-E]\)\s", stripped):
            repairs.append({
                "type": "inserted_missing_space_after_choice_label",
                "choice_index": choice_index,
                "label": label,
            })

    actual_labels = "".join(label for label, _ in parsed)
    if actual_labels == CHOICE_LABELS:
        pass
    elif (
        actual_labels == "ABCDDE"
        and parsed[4] == ("D", "None of the above")
    ):
        dropped_label, dropped_body = parsed.pop(4)
        repairs.append({
            "type": "removed_duplicate_none_of_the_above_choice",
            "choice_index": 4,
            "label": dropped_label,
            "body": dropped_body,
        })
    else:
        raise ValueError(
            f"{location} has invalid Answer choice labels: "
            f"{actual_labels!r}; expected {CHOICE_LABELS!r}, or the known "
            "repairable 'ABCDDE' sequence with the second D equal to "
            "'None of the above'."
        )

    normalized = [f"({label}) {body}" for label, body in parsed]
    return normalized, repairs


def _validate_mme_row(row: Any, dataset_ordinal: int) -> dict[str, Any]:
    location = f"dataset row {dataset_ordinal}"
    if not isinstance(row, dict):
        raise TypeError(f"{location} must be a JSON object, got {type(row)!r}.")

    missing = REQUIRED_FIELDS.difference(row)
    if missing:
        raise ValueError(
            f"{location} is missing fields: " + ", ".join(sorted(missing))
        )

    for field in ("Question_id", "Image", "Text", "Ground truth"):
        if not isinstance(row[field], str) or not row[field].strip():
            raise ValueError(f"{location} field {field!r} must be a non-empty string.")

    question_id = row["Question_id"].strip()
    choices, choice_repairs = _normalize_answer_choices(
        row["Answer choices"], dataset_ordinal, question_id
    )

    answer = row["Ground truth"].strip().upper()
    if answer not in CHOICE_LABELS:
        raise ValueError(
            f"{location} Ground truth must be one of {CHOICE_LABELS}, "
            f"got {row['Ground truth']!r}."
        )

    normalized = dict(row)
    normalized["Question_id"] = question_id
    normalized["Image"] = row["Image"].strip()
    normalized["Text"] = row["Text"].strip()
    normalized["Ground truth"] = answer
    normalized["Answer choices"] = choices
    if choice_repairs:
        normalized[CHOICE_REPAIRS_KEY] = choice_repairs
    return normalized


def load_mme_realworld_rows(
    question_path: Path,
) -> tuple[list[dict[str, Any]], list[int], list[dict[str, Any]]]:
    try:
        with question_path.open("r", encoding="utf-8") as handle:
            dataset = json.load(handle)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Invalid MME-RealWorld JSON: {question_path}"
        ) from exc

    if not isinstance(dataset, list):
        raise TypeError(
            "MME-RealWorld question JSON must contain a top-level list."
        )
    validated = [
        _validate_mme_row(row, index) for index, row in enumerate(dataset)
    ]
    normalization_repairs = [
        {
            "dataset_ordinal": dataset_ordinal,
            "question_id": row["Question_id"],
            **repair,
        }
        for dataset_ordinal, row in enumerate(validated)
        for repair in row.get(CHOICE_REPAIRS_KEY, [])
    ]
    selected = select_sample_indices(len(validated))
    return (
        [validated[index] for index in selected],
        selected,
        normalization_repairs,
    )


def resolve_image_path(path_value: str, image_dir: Path) -> Path:
    if not isinstance(path_value, str) or not path_value.strip():
        raise TypeError("MME-RealWorld Image must be a non-empty path string.")
    if "\x00" in path_value:
        raise ValueError("MME-RealWorld Image path contains a null byte.")

    possible_path = Path(path_value.strip()).expanduser()
    if not possible_path.is_absolute():
        possible_path = image_dir / possible_path
    if not possible_path.is_file():
        raise FileNotFoundError(
            f"MME-RealWorld image does not exist: {possible_path}"
        )
    return possible_path.resolve()


def decode_mme_realworld_image(
    path_value: str, image_dir: Path
) -> Image.Image:
    image_path = resolve_image_path(path_value, image_dir)
    try:
        return Image.open(image_path).convert("RGB")
    except Exception as exc:
        raise ValueError(f"Unable to open image file: {image_path}") from exc


def choice_mapping(row: dict[str, Any]) -> dict[str, str]:
    return {
        label: re.sub(rf"^\({label}\)\s+", "", choice, count=1)
        for label, choice in zip(CHOICE_LABELS, row["Answer choices"])
    }


def build_question(row: dict[str, Any]) -> str:
    choices = "\n".join(row["Answer choices"])
    return (
        f"{row['Text']} The choices are listed below:\n"
        f"{choices}\n{OFFICIAL_PROMPT_SUFFIX}"
    )


def build_result_record(
    row: dict[str, Any],
    sample_ordinal: int,
    dataset_ordinal: int,
    output: Any,
) -> dict[str, Any]:
    generated = output.outputs[0]
    raw_text = generated.text
    return {
        "sample_ordinal": sample_ordinal,
        "dataset_ordinal": int(dataset_ordinal),
        "dataset_index": row["Question_id"],
        "question": row["Text"],
        "answer": row["Ground truth"],
        "task": row["Task"],
        "subtask": row["Subtask"],
        "category": row["Category"],
        "source_dataset": row.get("Dataset"),
        "image_path": row["Image"],
        "choices": choice_mapping(row),
        "request_id": str(output.request_id),
        "raw_output_text": raw_text,
        "cleaned_output_text": replace_abs_vis_token_content(raw_text),
        "output_token_ids": [int(value) for value in generated.token_ids],
        "finish_reason": generated.finish_reason,
    }


def build_conversations(
    rows: list[dict[str, Any]], image_dir: Path
) -> tuple[list[list[dict[str, Any]]], list[Image.Image]]:
    conversations = []
    opened_images = []
    for row in rows:
        image = decode_mme_realworld_image(row["Image"], image_dir)
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
    vocabulary_exports = engine.collective_rpc(
        export_vocabulary_embeddings_worker, timeout=None
    )
    if len(vocabulary_exports) != 1 or vocabulary_exports[0].get("error"):
        raise RuntimeError(
            "Failed to export vocabulary input embeddings:\n"
            + json.dumps(vocabulary_exports, ensure_ascii=False, indent=2)
        )
    vocabulary_export = vocabulary_exports[0]
    vocabulary_path = Path(vocabulary_export["path"]).resolve()
    if (
        vocabulary_path.parent != capture_dir.resolve()
        or not vocabulary_path.is_file()
    ):
        raise RuntimeError(
            "Vocabulary exporter returned an invalid path: "
            f"{vocabulary_path}"
        )
    vocabulary_shape = np.load(vocabulary_path, mmap_mode="r").shape
    expected_shape = (
        int(vocabulary_export["vocab_size"]),
        int(vocabulary_export["hidden_size"]),
    )
    if vocabulary_shape != expected_shape:
        raise RuntimeError(
            "Vocabulary embedding shape mismatch: "
            f"{vocabulary_shape} != {expected_shape}"
        )
    print(
        "[Monet analysis] vocabulary export complete:\n"
        + json.dumps(vocabulary_export, ensure_ascii=False, indent=2)
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
    return outputs, worker_statuses, vocabulary_export


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


def scan_capture_counts(
    capture_paths: list[Path],
) -> tuple[np.ndarray, np.ndarray]:
    image_counts = np.zeros(len(capture_paths), dtype=np.int64)
    latent_counts = np.zeros(len(capture_paths), dtype=np.int64)
    for sample_ordinal, path in enumerate(capture_paths):
        with np.load(path, allow_pickle=False) as data:
            kind_codes = data["kind_codes"]
            image_counts[sample_ordinal] = np.count_nonzero(kind_codes == 1)
            latent_counts[sample_ordinal] = np.count_nonzero(kind_codes == 2)
    return image_counts, latent_counts


def sample_image_positions(
    available_count: int, target_count: int
) -> tuple[np.ndarray, bool]:
    if available_count <= 0:
        raise ValueError("No image features were captured for PCA.")
    if target_count <= 0:
        raise ValueError("The image feature sample size must be positive.")
    replace = available_count < target_count
    rng = np.random.default_rng(RANDOM_SEED)
    positions = rng.choice(
        available_count, size=target_count, replace=replace
    )
    return np.sort(positions.astype(np.int64, copy=False)), replace


def extract_image_and_latent_vectors(
    capture_paths: list[Path],
    sample_records: list[dict[str, Any]],
    target_image_count: int,
    hidden_size: int,
    temporary_dir: Path,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray], dict[str, Any]]:
    image_counts, latent_counts = scan_capture_counts(capture_paths)
    total_images = int(image_counts.sum())
    total_latents = int(latent_counts.sum())
    chosen_images, used_replacement = sample_image_positions(
        total_images, target_image_count
    )

    image_path = temporary_dir / "sampled_image_embeddings.npy"
    image_vectors = np.lib.format.open_memmap(
        image_path,
        mode="w+",
        dtype=np.float16,
        shape=(target_image_count, hidden_size),
    )
    latent_vectors = np.empty((total_latents, hidden_size), dtype=np.float16)
    image_sample_ordinals = np.empty(target_image_count, dtype=np.int32)
    image_sequence_positions = np.empty(target_image_count, dtype=np.int32)
    image_generation_steps = np.empty(target_image_count, dtype=np.int32)
    latent_sample_ordinals = np.empty(total_latents, dtype=np.int32)
    latent_sequence_positions = np.empty(total_latents, dtype=np.int32)
    latent_generation_steps = np.empty(total_latents, dtype=np.int32)
    latent_indices = np.empty(total_latents, dtype=np.int32)
    latent_trajectory_steps = np.empty(total_latents, dtype=np.int32)

    image_global_start = 0
    latent_output_start = 0
    for sample_ordinal, path in enumerate(capture_paths):
        with np.load(path, allow_pickle=False) as data:
            vectors = data["vectors"]
            if vectors.ndim != 2 or vectors.shape[1] != hidden_size:
                raise RuntimeError(
                    f"Capture hidden size mismatch in {path}: "
                    f"{vectors.shape} versus (*, {hidden_size})"
                )
            kind_codes = data["kind_codes"]
            image_rows = np.flatnonzero(kind_codes == 1)
            latent_rows = np.flatnonzero(kind_codes == 2)

            image_global_end = image_global_start + len(image_rows)
            chosen_start = np.searchsorted(
                chosen_images, image_global_start, side="left"
            )
            chosen_end = np.searchsorted(
                chosen_images, image_global_end, side="left"
            )
            if chosen_end > chosen_start:
                local_image_ordinals = (
                    chosen_images[chosen_start:chosen_end] - image_global_start
                )
                selected_rows = image_rows[local_image_ordinals]
                image_vectors[chosen_start:chosen_end] = vectors[selected_rows]
                image_sample_ordinals[chosen_start:chosen_end] = sample_ordinal
                image_sequence_positions[chosen_start:chosen_end] = data[
                    "sequence_positions"
                ][selected_rows]
                image_generation_steps[chosen_start:chosen_end] = data[
                    "generation_steps"
                ][selected_rows]
            image_global_start = image_global_end

            latent_output_end = latent_output_start + len(latent_rows)
            if len(latent_rows):
                order = np.argsort(
                    data["sequence_positions"][latent_rows], kind="stable"
                )
                latent_rows = latent_rows[order]
                output_slice = slice(latent_output_start, latent_output_end)
                latent_vectors[output_slice] = vectors[latent_rows]
                latent_sample_ordinals[output_slice] = sample_ordinal
                latent_sequence_positions[output_slice] = data[
                    "sequence_positions"
                ][latent_rows]
                latent_generation_steps[output_slice] = data[
                    "generation_steps"
                ][latent_rows]
                latent_indices[output_slice] = data["latent_indices"][latent_rows]
                latent_trajectory_steps[output_slice] = np.arange(
                    len(latent_rows), dtype=np.int32
                )
            latent_output_start = latent_output_end

            prompt_length = int(data["prompt_length"].item())
            consumed_count = int(data["consumed_output_token_count"].item())
            sampled_ids = sample_records[sample_ordinal]["output_token_ids"]
            sample_records[sample_ordinal]["capture_counts"] = {
                "image_feature": int(len(image_rows)),
                "latent": int(len(latent_rows)),
            }
            sample_records[sample_ordinal]["prompt_token_count"] = prompt_length
            sample_records[sample_ordinal][
                "consumed_output_token_count"
            ] = consumed_count
            sample_records[sample_ordinal][
                "unconsumed_output_token_ids"
            ] = sampled_ids[consumed_count:]

    image_vectors.flush()
    latent_offsets = np.concatenate((
        np.asarray([0], dtype=np.int64),
        np.cumsum(latent_counts, dtype=np.int64),
    ))
    metadata = {
        "image_sample_ordinal": image_sample_ordinals,
        "image_sequence_positions": image_sequence_positions,
        "image_generation_steps": image_generation_steps,
        "latent_sample_ordinal": latent_sample_ordinals,
        "latent_sequence_positions": latent_sequence_positions,
        "latent_generation_steps": latent_generation_steps,
        "latent_indices": latent_indices,
        "latent_trajectory_steps": latent_trajectory_steps,
        "latent_sample_offsets": latent_offsets,
    }
    statistics = {
        "available_image_features": total_images,
        "sampled_image_features": target_image_count,
        "image_sampling_with_replacement": used_replacement,
        "latent_vectors": total_latents,
        "per_sample_image_counts": image_counts.tolist(),
        "per_sample_latent_counts": latent_counts.tolist(),
    }
    return image_vectors, latent_vectors, metadata, statistics


def _copy_float32_blocks(destination, start: int, source: np.ndarray) -> int:
    for source_start in range(0, len(source), PCA_TRANSFORM_BATCH_SIZE):
        source_end = min(
            source_start + PCA_TRANSFORM_BATCH_SIZE, len(source)
        )
        count = source_end - source_start
        destination[start:start + count] = source[source_start:source_end]
        start += count
    return start


def _project_vectors(pca: PCA, vectors: np.ndarray) -> np.ndarray:
    coordinates = np.empty((len(vectors), 3), dtype=np.float32)
    for start in range(0, len(vectors), PCA_TRANSFORM_BATCH_SIZE):
        end = min(start + PCA_TRANSFORM_BATCH_SIZE, len(vectors))
        coordinates[start:end] = pca.transform(
            vectors[start:end].astype(np.float32, copy=False)
        )
    return coordinates


def fit_and_project_joint_pca(
    vocabulary_vectors: np.ndarray,
    image_vectors: np.ndarray,
    latent_vectors: np.ndarray,
    temporary_dir: Path,
) -> tuple[PCA, list[np.ndarray]]:
    vector_sources = [vocabulary_vectors, image_vectors, latent_vectors]
    hidden_sizes = {vectors.shape[1] for vectors in vector_sources}
    if len(hidden_sizes) != 1:
        raise ValueError(f"Embedding hidden sizes do not match: {hidden_sizes}")
    total_points = sum(len(vectors) for vectors in vector_sources)
    if total_points < 3:
        raise ValueError("Fewer than three vectors are available for PCA.")

    fit_path = temporary_dir / "joint_pca_fit.float32.mmap"
    fit_matrix = np.memmap(
        fit_path,
        mode="w+",
        dtype=np.float32,
        shape=(total_points, hidden_sizes.pop()),
    )
    try:
        destination_start = 0
        for vectors in vector_sources:
            destination_start = _copy_float32_blocks(
                fit_matrix, destination_start, vectors
            )
        fit_matrix.flush()
        pca = PCA(
            n_components=3,
            svd_solver="randomized",
            random_state=RANDOM_SEED,
            copy=False,
        )
        pca.fit(fit_matrix)
    finally:
        del fit_matrix
        gc.collect()
        if fit_path.exists():
            fit_path.unlink()

    projected = [
        _project_vectors(pca, vectors) for vectors in vector_sources
    ]
    return pca, projected


def assemble_joint_points(
    projected: list[np.ndarray], metadata: dict[str, np.ndarray]
) -> dict[str, np.ndarray]:
    vocabulary_coordinates, image_coordinates, latent_coordinates = projected
    vocabulary_count = len(vocabulary_coordinates)
    image_count = len(image_coordinates)
    latent_count = len(latent_coordinates)
    return {
        "coordinates": np.concatenate(projected, axis=0),
        "kind_codes": np.concatenate((
            np.zeros(vocabulary_count, dtype=np.uint8),
            np.ones(image_count, dtype=np.uint8),
            np.full(latent_count, 2, dtype=np.uint8),
        )),
        "token_ids": np.concatenate((
            np.arange(vocabulary_count, dtype=np.int32),
            np.full(image_count + latent_count, -1, dtype=np.int32),
        )),
        "sample_ordinal": np.concatenate((
            np.full(vocabulary_count, -1, dtype=np.int32),
            metadata["image_sample_ordinal"],
            metadata["latent_sample_ordinal"],
        )),
        "sequence_positions": np.concatenate((
            np.full(vocabulary_count, -1, dtype=np.int32),
            metadata["image_sequence_positions"],
            metadata["latent_sequence_positions"],
        )),
        "generation_steps": np.concatenate((
            np.full(vocabulary_count, -1, dtype=np.int32),
            metadata["image_generation_steps"],
            metadata["latent_generation_steps"],
        )),
        "latent_indices": np.concatenate((
            np.full(vocabulary_count + image_count, -1, dtype=np.int32),
            metadata["latent_indices"],
        )),
        "trajectory_steps": np.concatenate((
            np.full(vocabulary_count + image_count, -1, dtype=np.int32),
            metadata["latent_trajectory_steps"],
        )),
    }


def save_pca_archives(
    output_path: Path,
    points: dict[str, np.ndarray],
    latent_coordinates: np.ndarray,
    metadata: dict[str, np.ndarray],
    pca: PCA,
    sample_records: list[dict[str, Any]],
) -> None:
    dataset_indices = np.asarray(
        [str(record["dataset_index"]) for record in sample_records],
        dtype=np.str_,
    )
    request_ids = np.asarray(
        [record["request_id"] for record in sample_records], dtype=np.str_
    )
    np.savez_compressed(
        output_path / JOINT_PCA_FILE,
        **points,
        kind_names=KIND_NAMES,
        dataset_indices=dataset_indices,
        request_ids=request_ids,
        pca_components=pca.components_.astype(np.float32),
        pca_mean=pca.mean_.astype(np.float32),
        explained_variance=pca.explained_variance_.astype(np.float32),
        explained_variance_ratio=pca.explained_variance_ratio_.astype(np.float32),
    )
    np.savez_compressed(
        output_path / LATENT_TRAJECTORY_FILE,
        coordinates=latent_coordinates,
        sample_ordinal=metadata["latent_sample_ordinal"],
        sequence_positions=metadata["latent_sequence_positions"],
        generation_steps=metadata["latent_generation_steps"],
        latent_indices=metadata["latent_indices"],
        trajectory_steps=metadata["latent_trajectory_steps"],
        sample_offsets=metadata["latent_sample_offsets"],
        dataset_indices=dataset_indices,
        request_ids=request_ids,
    )


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def global_config_snapshot() -> dict[str, Any]:
    names = [
        "MONET_REPO_DIR", "MODEL_PATH", "MME_REALWORLD_DIR", "QUESTION_FILE",
        "IMAGE_DIR", "OUTPUT_DIR",
        "JOINT_PCA_FILE", "LATENT_TRAJECTORY_FILE",
        "SELECTION_MODE", "START_INDEX", "NUM_SAMPLES", "RANDOM_SEED",
        "LATENT_SIZE", "TENSOR_PARALLEL_SIZE", "GPU_MEMORY_UTILIZATION",
        "MAX_MODEL_LEN", "MAX_NUM_SEQS", "MAX_OUTPUT_TOKENS",
        "SWAP_SPACE_GB", "DTYPE", "ENABLE_CHUNKED_PREFILL",
        "ENABLE_SLEEP_MODE", "MIN_PIXELS", "MAX_PIXELS", "TEMPERATURE",
        "TOP_K", "TOP_P", "REPETITION_PENALTY", "BEST_OF", "STOP",
        "VOCAB_EMBEDDING_BATCH_SIZE", "PCA_TRANSFORM_BATCH_SIZE",
    ]
    return {name: globals()[name] for name in names}


def main() -> None:
    model_path, question_path, image_dir, output_path = validate_configuration()
    rows, selected_indices, normalization_repairs = load_mme_realworld_rows(
        question_path
    )
    capture_dir = Path(tempfile.mkdtemp(prefix=".monet_capture_", dir=output_path))
    succeeded = False
    try:
        outputs, worker_statuses, vocabulary_export = run_inference(
            model_path, rows, image_dir, capture_dir
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
            record = build_result_record(
                row, sample_ordinal, dataset_ordinal, output
            )
            request_id = record["request_id"]
            request_ids.append(request_id)
            sample_records.append(record)

        capture_paths = wait_for_captures(
            capture_dir, request_ids, worker_statuses
        )
        ordered_capture_paths = [
            capture_paths[request_id] for request_id in request_ids
        ]
        vocabulary_vectors = np.load(
            Path(vocabulary_export["path"]), mmap_mode="r"
        )
        image_vectors, latent_vectors, metadata, capture_statistics = (
            extract_image_and_latent_vectors(
                ordered_capture_paths,
                sample_records,
                target_image_count=len(vocabulary_vectors),
                hidden_size=int(vocabulary_vectors.shape[1]),
                temporary_dir=capture_dir,
            )
        )
        pca, projected = fit_and_project_joint_pca(
            vocabulary_vectors,
            image_vectors,
            latent_vectors,
            capture_dir,
        )
        points = assemble_joint_points(projected, metadata)
        save_pca_archives(
            output_path,
            points,
            projected[2],
            metadata,
            pca,
            sample_records,
        )
        del vocabulary_vectors, image_vectors, latent_vectors, projected
        gc.collect()
        write_jsonl(output_path / "results.jsonl", sample_records)

        run_config = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "config": global_config_snapshot(),
            "vllm_enable_v1_multiprocessing": False,
            "validated_worker_statuses": worker_statuses,
            "vocabulary_export": {
                key: value
                for key, value in vocabulary_export.items()
                if key != "path"
            },
            "selected_dataset_ordinals": selected_indices,
            "dataset_normalization_repairs": normalization_repairs,
            "capture_statistics": capture_statistics,
            "pca_input_counts": {
                "vocabulary_embedding": int(vocabulary_export["vocab_size"]),
                "image_feature": capture_statistics["sampled_image_features"],
                "latent": capture_statistics["latent_vectors"],
            },
            "pca_explained_variance_ratio":
                pca.explained_variance_ratio_.tolist(),
            "total_projected_points": int(len(points["coordinates"])),
            "outputs": {
                "joint_pca": JOINT_PCA_FILE,
                "latent_trajectories": LATENT_TRAJECTORY_FILE,
                "results": "results.jsonl",
            },
            "note": (
                "Each sample's latent vectors are ordered by actual consumed "
                "sequence position and form one trajectory, including across "
                "intervening text tokens; trajectories are not attention paths."
            ),
        }
        (output_path / "run_config.json").write_text(
            json.dumps(run_config, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        succeeded = True
        print(f"Analysis complete: {output_path}")
        print(f"Joint PCA: {output_path / JOINT_PCA_FILE}")
        print(f"Latent trajectories: {output_path / LATENT_TRAJECTORY_FILE}")
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
