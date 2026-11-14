"""Pinned native-environment workers for non-SmolVLA SO101 policies.

This module is stdlib-only until AdapterRunner starts its isolated worker.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import selectors
import stat
import struct
import subprocess
import sys
import tempfile
import threading
import time
from typing import Any

SCHEMA = "so101.adapter-deployment.v1"
SEMANTICS = "so101.absolute-position.degrees.gripper-0-100.rgb.v1"
MODELS = ("rdvla", "anchormix", "pi05")
JOINTS = (
    "shoulder_pan.pos",
    "shoulder_lift.pos",
    "elbow_flex.pos",
    "wrist_flex.pos",
    "wrist_roll.pos",
    "gripper.pos",
)
CAMERAS = ("top", "wrist")
PREDICTION_STEPS = 50
GPU_RESERVE_BYTES = 4 * 1024**3
IMAGE_SHAPE = (480, 640, 3)
STATE_BYTES = 6 * 4
IMAGE_BYTES = 480 * 640 * 3
INPUT_BYTES = STATE_BYTES + 2 * IMAGE_BYTES
MAX_HEADER_BYTES = 4 * 1024 * 1024
MAX_TASK_CHARS = 4096
START_TIMEOUT = 900.0
REQUEST_TIMEOUT = 600.0
CLOSE_TIMEOUT = 5.0
_REQUIRED_DESCRIPTOR_KEYS = frozenset(
    {
        "schema",
        "model",
        "deployment",
        "artifact",
        "semantics",
        "checkpoint",
        "training_config",
        "python",
        "source_root",
        "pythonpath",
        "gpu_uuid",
        "progress",
        "checkpoint_manifest_sha256",
        "training_config_sha256",
        "source_sha256",
    }
)
_RUNTIME_KEYS = frozenset({"root", "python", "gpu_uuid"})


def validate_memory_limit(memory_limit_gib: float | None) -> float | None:
    if memory_limit_gib is None:
        return None
    if type(memory_limit_gib) not in (int, float) or not math.isfinite(memory_limit_gib) or memory_limit_gib <= 0:
        raise ValueError("memory-limit-gib must be finite and positive")
    return float(memory_limit_gib)


def apply_cuda_memory_limit(torch, memory_limit_gib: float | None) -> None:
    limit = validate_memory_limit(memory_limit_gib)
    if limit is None:
        return
    free, total = torch.cuda.mem_get_info(0)
    if limit > total / 1024**3:
        raise RuntimeError("Requested inference memory limit exceeds physical GPU capacity")
    budget = int(limit * 1024**3)
    if budget < 1 or free < budget + GPU_RESERVE_BYTES:
        raise RuntimeError("Inference needs its memory limit plus 4 GiB free GPU reserve; nothing was stopped")
    torch.cuda.set_per_process_memory_fraction(budget / total, device=0)


def _canonical_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _bad_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant is forbidden: {value}")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _json_bytes(data: bytes) -> Any:
    return json.loads(data, parse_constant=_bad_constant, object_pairs_hook=_unique_object)


def _read_json(path: Path) -> Any:
    return _json_bytes(path.read_bytes())


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _path(value: Any, label: str, *, strict: bool = True) -> Path:
    if not isinstance(value, str) or not value or not Path(value).is_absolute():
        raise ValueError(f"{label} must be an absolute path")
    return Path(value).resolve(strict=strict)


def _executable(value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value or not Path(value).is_absolute():
        raise ValueError(f"{label} must be an absolute executable path")
    path = Path(os.path.abspath(value))
    if not path.is_file() or not os.access(path, os.X_OK):
        raise ValueError(f"{label} is missing or not executable: {path}")
    return path


def _hash_value(value: Any, label: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _deployment_identity(spec: dict[str, Any]) -> str:
    return "sha256:" + _canonical_hash({key: value for key, value in spec.items() if key != "artifact"})


def _validate_progress_shape(model: str, progress: Any) -> None:
    if not isinstance(progress, dict):
        raise ValueError("deployment progress must be a JSON object")
    if model == "rdvla":
        identity = progress.get("identity")
        if (
            progress.get("schema") != 1
            or progress.get("status") != "checkpoint_committed"
            or not isinstance(identity, dict)
            or identity.get("model") != model
            or not isinstance(identity.get("run_id"), str)
            or not isinstance(identity.get("source_files"), list)
            or not isinstance(progress.get("wandb"), dict)
        ):
            raise ValueError("RD-VLA progress lacks its committed training identity")
    elif (
        progress.get("schema") != "so101.training.v1"
        or progress.get("model") != model
        or not isinstance(progress.get("wandb_run_id"), str)
        or not isinstance(progress.get("wandb_entity"), str)
        or not isinstance(progress.get("wandb_project"), str)
    ):
        raise ValueError(f"{model} progress lacks its committed training identity")


def _validate_descriptor(spec: Any, model: str | None = None) -> dict[str, Any]:
    """Validate descriptor structure and self-binding without resolving server paths."""
    if not isinstance(spec, dict) or set(spec) not in (
        _REQUIRED_DESCRIPTOR_KEYS,
        _REQUIRED_DESCRIPTOR_KEYS | {"runtime"},
    ):
        raise ValueError("deployment descriptor keys do not match the supported schema")
    if spec["schema"] != SCHEMA or spec["semantics"] != SEMANTICS:
        raise ValueError("unsupported SO101 adapter descriptor schema or semantics")
    name = spec["model"]
    if name not in MODELS or (model is not None and name != model):
        raise ValueError(f"deployment model mismatch: expected {model!r}, got {name!r}")
    deployment = spec["deployment"]
    if not isinstance(deployment, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,255}", deployment):
        raise ValueError("deployment must be a nonempty safe identifier")
    for key in ("checkpoint", "training_config", "source_root"):
        _path(spec[key], key, strict=False)
    if not isinstance(spec["python"], str) or not Path(spec["python"]).is_absolute():
        raise ValueError("python must be an absolute path")
    if not isinstance(spec["pythonpath"], list) or not spec["pythonpath"]:
        raise ValueError("pythonpath must be a nonempty list of absolute paths")
    paths = [_path(value, "pythonpath entry", strict=False) for value in spec["pythonpath"]]
    if len(set(paths)) != len(paths):
        raise ValueError("pythonpath entries must be unique")
    if not isinstance(spec["gpu_uuid"], str) or not re.fullmatch(r"GPU-[A-Za-z0-9-]+", spec["gpu_uuid"]):
        raise ValueError("gpu_uuid must identify a physical NVIDIA GPU")
    if "runtime" in spec:
        runtime = spec["runtime"]
        if name != "rdvla" or not isinstance(runtime, dict) or set(runtime) != _RUNTIME_KEYS:
            raise ValueError("runtime relocation is supported only for RD-VLA with root, python, and gpu_uuid")
        runtime_root = _path(runtime["root"], "runtime root", strict=False)
        if not isinstance(runtime["python"], str) or not Path(runtime["python"]).is_absolute():
            raise ValueError("runtime python must be an absolute path")
        if not isinstance(runtime["gpu_uuid"], str) or not re.fullmatch(r"GPU-[A-Za-z0-9-]+", runtime["gpu_uuid"]):
            raise ValueError("runtime gpu_uuid must identify a physical NVIDIA GPU")
        runtime_source = str(runtime_root / "source")
        if (
            spec["python"] != runtime["python"]
            or spec["gpu_uuid"] != runtime["gpu_uuid"]
            or spec["source_root"] != runtime_source
            or spec["pythonpath"] != [runtime_source]
            or Path(spec["checkpoint"]).parent != runtime_root / "runs" / "rdvla" / "checkpoints"
        ):
            raise ValueError("runtime paths or hardware differ from the descriptor")
    _hash_value(spec["checkpoint_manifest_sha256"], "checkpoint_manifest_sha256")
    _hash_value(spec["training_config_sha256"], "training_config_sha256")
    _hash_value(spec["source_sha256"], "source_sha256")
    _validate_progress_shape(name, spec["progress"])
    if not isinstance(spec["artifact"], str) or not re.fullmatch(r"sha256:[0-9a-f]{64}", spec["artifact"]):
        raise ValueError("artifact must be a sha256 identity")
    if _deployment_identity(spec) != spec["artifact"]:
        raise ValueError("deployment descriptor self-binding does not match its contents")
    return spec


def load_deployment(path: Path, *, model: str | None = None) -> dict[str, Any]:
    """Load a copied descriptor; server-only paths are intentionally not resolved here."""
    input_path = Path(path).expanduser()
    if input_path.is_symlink():
        raise ValueError("deployment descriptor must not be a symlink")
    descriptor = input_path.resolve(strict=True)
    if not stat.S_ISREG(descriptor.lstat().st_mode):
        raise ValueError("deployment descriptor must be a regular file")
    return dict(_validate_descriptor(_read_json(descriptor), model=model))


def _model_config(config: dict[str, Any], model: str) -> dict[str, Any]:
    models = config.get("models")
    if not isinstance(models, dict) or not isinstance(models.get(model), dict):
        raise ValueError(f"training config has no model entry for {model}")
    return models[model]


def _safe_relative(value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a nonempty relative path")
    result = Path(value)
    if result.is_absolute() or ".." in result.parts or result.as_posix() in {"", "."}:
        raise ValueError(f"{label} is unsafe: {value!r}")
    return result




def _active_rd_records(config: dict[str, Any], model: str) -> list[dict[str, str]]:
    records = config.get("source_files")
    if not isinstance(records, list):
        raise ValueError("RD-VLA config source_files manifest is missing")
    native_tree = "RD-VLA/"
    relevant = {"train.py", "training_state.py", "so101_data.py", f"{model}_adapter.py"}
    selected = [
        row
        for row in records
        if isinstance(row, dict)
        and isinstance(row.get("path"), str)
        and (row["path"] in relevant or row["path"].startswith(native_tree))
    ]
    return _validate_source_records_from_config(selected, "RD-VLA")


def _validate_source_records_from_config(records: list[dict[str, Any]], label: str) -> list[dict[str, str]]:
    if not records:
        raise ValueError(f"{label} active source manifest is empty")
    output: list[dict[str, str]] = []
    seen: set[str] = set()
    for row in records:
        if set(row) != {"path", "sha256"}:
            raise ValueError(f"{label} active source file entry is invalid")
        rel = _safe_relative(row["path"], f"{label} source path").as_posix()
        digest = _hash_value(row["sha256"], f"{label} source hash")
        if rel in seen:
            raise ValueError(f"duplicate {label} source file: {rel}")
        seen.add(rel)
        output.append({"path": rel, "sha256": digest})
    return output


def _actual_source_records(source_root: Path, records: Any, label: str) -> list[dict[str, str]]:
    validated = _validate_source_records_from_config(records, label)
    actual = []
    for item in validated:
        path = source_root
        for part in Path(item["path"]).parts:
            path = path / part
            if path.is_symlink():
                raise ValueError(f"{label} source tree contains a symlink: {item['path']}")
        if not path.is_file():
            raise ValueError(f"{label} source file is missing: {item['path']}")
        actual.append({"path": item["path"], "sha256": _file_hash(path)})
    return actual


def _verify_rd_source_tree(source_root: Path, records: Any) -> list[dict[str, str]]:
    expected = _validate_source_records_from_config(records, "RD-VLA")
    actual = _actual_source_records(source_root, expected, "RD-VLA")
    if any(
        saved["sha256"] != current["sha256"] and saved["path"] != "train.py"
        for saved, current in zip(expected, actual)
    ):
        raise ValueError("RD-VLA model/source files differ from the pinned config")
    return actual


def _source_tree_records(source_root: Path) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    for path in sorted(source_root.rglob("*")):
        if "__pycache__" in path.parts or path.suffix not in {".py", ".yaml", ".toml"}:
            continue
        if path.is_symlink():
            raise ValueError(f"training source tree contains a symlink: {path}")
        if path.is_file():
            records.append({"path": path.relative_to(source_root).as_posix(), "sha256": _file_hash(path)})
    if not records:
        raise ValueError("training source tree has no pinned Python/YAML/TOML files")
    return records


def _source_tree_identity(source_root: Path) -> str:
    return _canonical_hash(_source_tree_records(source_root))


def _rd_config_identity(config: dict[str, Any], model_cfg: dict[str, Any], model: str) -> str:
    identity = {key: value for key, value in config.items() if key not in {"models", "source_files"}}
    identity["model"] = model_cfg
    return _canonical_hash(identity)


def _semantic_config(config: dict[str, Any]) -> str:
    ignored = {"source_sha256", "source_files", "resume_compatible_code"}
    return _canonical_hash({key: value for key, value in config.items() if key not in ignored})


def _compat_configs(config: dict[str, Any], config_root: Path, model: str) -> list[tuple[dict[str, Any], str]]:
    records = config.get("resume_compatible_code", [])
    if not isinstance(records, list):
        raise ValueError("resume_compatible_code must be a list")
    compatible: list[tuple[dict[str, Any], str]] = []
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("resume-compatible config entry must be an object")
        name = record.get("config_file")
        if not isinstance(name, str) or Path(name).name != name:
            raise ValueError("resume-compatible config filename must be a basename")
        expected_hash = _hash_value(record.get("config_sha256"), "resume-compatible config hash")
        _hash_value(record.get("source_sha256"), "resume-compatible source hash")
        original_path = config_root / name
        if original_path.is_symlink() or not original_path.is_file() or _file_hash(original_path) != expected_hash:
            raise ValueError(f"declared resume-compatible config is missing or changed: {name}")
        original = _read_json(original_path)
        if not isinstance(original, dict) or _semantic_config(original) != _semantic_config(config):
            raise ValueError(f"resume-compatible config changes the trained model/data identity: {name}")
        compatible.append((original, expected_hash))
    return compatible


def _normalize_runtime(runtime: Any) -> dict[str, str]:
    if not isinstance(runtime, dict) or set(runtime) != _RUNTIME_KEYS:
        raise ValueError("runtime must contain exactly root, python, and gpu_uuid")
    root_value = runtime["root"]
    if not isinstance(root_value, str) or not Path(root_value).is_absolute():
        raise ValueError("runtime root must be an absolute path")
    root_input = Path(root_value)
    if root_input.is_symlink():
        raise ValueError("runtime root must not be a symlink")
    root = _path(root_value, "runtime root")
    if not root.is_dir():
        raise ValueError(f"runtime root is not a directory: {root}")
    python = _executable(runtime["python"], "runtime python")
    gpu_uuid = runtime["gpu_uuid"]
    if not isinstance(gpu_uuid, str) or not re.fullmatch(r"GPU-[A-Za-z0-9-]+", gpu_uuid):
        raise ValueError("runtime gpu_uuid must identify a physical NVIDIA GPU")
    return {"root": str(root), "python": str(python), "gpu_uuid": gpu_uuid}


def _verify_runtime_checkpoint(runtime: dict[str, str], checkpoint: Path) -> None:
    runtime_root = Path(runtime["root"])
    checkpoint_root = runtime_root / "runs" / "rdvla" / "checkpoints"
    for directory in (runtime_root / "runs", runtime_root / "runs" / "rdvla", checkpoint_root):
        if directory.is_symlink() or not directory.is_dir():
            raise ValueError(f"runtime checkpoint directory is missing or a symlink: {directory}")
    if checkpoint.parent != checkpoint_root:
        raise ValueError("RD-VLA checkpoint must be directly inside the runtime checkpoint root")


def _original_rd_checkpoint(config: dict[str, Any], checkpoint: Path) -> Path:
    root = config.get("root")
    if not isinstance(root, str) or not Path(root).is_absolute():
        raise ValueError("training root must be an absolute path")
    return Path(root) / "runs" / "rdvla" / "checkpoints" / checkpoint.name


def _verify_checkpoint_root(config: dict[str, Any], model: str, checkpoint: Path) -> tuple[Path, Path]:
    root = _path(config.get("root"), "training root")
    checkpoint_root = (root / "runs" / model / "checkpoints").resolve(strict=True)
    if checkpoint == checkpoint_root or not checkpoint.is_relative_to(checkpoint_root):
        raise ValueError(f"checkpoint must be inside the configured {model} checkpoint root")
    return root, checkpoint_root


def _verify_model_settings(
    config: dict[str, Any], model: str, *, runtime: dict[str, str] | None = None
) -> tuple[dict[str, Any], Path, Path, list[str]]:
    model_cfg = _model_config(config, model)
    if config.get("action_horizon") != PREDICTION_STEPS or config.get("execution_horizon") != PREDICTION_STEPS:
        raise ValueError("training config must retain the trained SO101 50-step prediction/execution horizons")
    if runtime is None:
        root = _path(config.get("root"), "training root")
        if "source_root" in config:
            source_root = _path(config["source_root"], "source_root")
        else:
            source_root = (root / "source").resolve(strict=True)
        _executable(model_cfg.get("python"), "training python")
    else:
        if model != "rdvla":
            raise ValueError("runtime relocation is supported only for RD-VLA")
        root = _path(config.get("root"), "training root", strict=False)
        if "source_root" in config:
            _path(config["source_root"], "source_root", strict=False)
        if not isinstance(model_cfg.get("python"), str) or not Path(model_cfg["python"]).is_absolute():
            raise ValueError("original training python must be an absolute path")
        original_gpu = config.get("gpu_uuid")
        if not isinstance(original_gpu, str) or not re.fullmatch(r"GPU-[A-Za-z0-9-]+", original_gpu):
            raise ValueError("training config must pin a physical GPU UUID")
        runtime_root = _path(runtime["root"], "runtime root")
        source_root = runtime_root / "source"
        if source_root.is_symlink() or not source_root.is_dir():
            raise ValueError(f"runtime source tree is missing or a symlink: {source_root}")
        native_root = source_root / "RD-VLA"
        if native_root.is_symlink() or not native_root.is_dir():
            raise ValueError(f"runtime source tree lacks RD-VLA: {native_root}")
        _executable(runtime["python"], "runtime python")
    adapter_file = source_root / f"{model}_adapter.py"
    if not adapter_file.is_file() or adapter_file.is_symlink():
        raise ValueError(f"native {model} adapter is missing from the configured source tree")
    if not (source_root / "training_state.py").is_file():
        raise ValueError("configured source tree lacks training_state.py")
    pythonpath: list[Path] = []
    if model == "pi05":
        lerobot_root = _path(config.get("lerobot_root"), "PI05 lerobot_root")
        native_lerobot = lerobot_root / "src"
        if not (native_lerobot / "lerobot" / "__init__.py").is_file():
            raise ValueError(f"configured native LeRobot source is missing: {native_lerobot}")
        pythonpath.extend((native_lerobot, source_root))
    else:
        pythonpath.append(source_root)
    if any("/inference/runtime/" in str(path) for path in pythonpath):
        raise ValueError("training PYTHONPATH must not use the LeRobot 0.7 runtime overlays")
    return model_cfg, root, source_root, [str(path.resolve(strict=True)) for path in pythonpath]


def _verify_source_lineage(config: dict[str, Any], model: str, source_root: Path) -> str:
    if model == "rdvla":
        current = _verify_rd_source_tree(source_root, _active_rd_records(config, model))
        return _canonical_hash(current)
    identity = _source_tree_identity(source_root)
    expected = _hash_value(config.get("source_sha256"), "training source_sha256")
    if identity != expected:
        raise ValueError("current native training source differs from config source_sha256")
    if model == "pi05":
        # This run uses an external native LeRobot checkout, not a copy in source_root.
        # Bind that serving dependency too; progress still validates the original training hash.
        lerobot_root = _path(config.get("lerobot_root"), "PI05 lerobot_root")
        return _canonical_hash({"adapter_source": identity, "native_lerobot_source": _source_tree_identity(lerobot_root)})
    return identity


def _check_hf_progress(
    config: dict[str, Any], config_path: Path, model: str, model_cfg: dict[str, Any], progress: dict[str, Any]
) -> None:
    if progress.get("schema") != "so101.training.v1" or progress.get("model") != model:
        raise ValueError("checkpoint progress schema/model does not match the requested model")
    required = {
        "config_sha256", "source_sha256", "data_sha256", "normalization_sha256",
        "wandb_run_id", "wandb_entity", "wandb_project", "steps", "training_samples",
        "epoch_index", "offset", "completed_epochs", "milestone_epoch",
    }
    if required - progress.keys():
        raise ValueError("checkpoint progress is missing required lineage or cursor fields")
    current_pair = (_file_hash(config_path), _hash_value(config.get("source_sha256"), "source_sha256"))
    saved_pair = (progress.get("config_sha256"), progress.get("source_sha256"))
    compatible = saved_pair == current_pair
    if not compatible:
        for original, original_hash in _compat_configs(config, Path(config["root"]), model):
            pair = (original_hash, original.get("source_sha256"))
            if saved_pair == pair:
                compatible = True
                break
    if not compatible:
        raise ValueError("checkpoint config/source lineage is not current or explicitly resume-compatible")
    for key in ("data_sha256", "normalization_sha256"):
        expected = _hash_value(config.get(key), key)
        if progress.get(key) != expected:
            raise ValueError(f"checkpoint {key} differs from training config")
    expected_wandb_id = model_cfg.get("wandb_run_id")
    if not isinstance(expected_wandb_id, str) or not expected_wandb_id or progress.get("wandb_run_id") != expected_wandb_id:
        raise ValueError("checkpoint W&B run ID differs from the selected model config")
    for key in ("wandb_entity", "wandb_project"):
        if progress.get(key) != config.get(key):
            raise ValueError(f"checkpoint {key} differs from training config")
    epochs, frames, batch_size, total_steps = (config.get(key) for key in ("epochs", "frames", "batch_size", "total_steps"))
    steps, samples = progress.get("steps"), progress.get("training_samples")
    epoch, offset = progress.get("epoch_index"), progress.get("offset")
    completed = progress.get("completed_epochs")
    if (
        type(epochs) is not int or epochs <= 0
        or type(frames) is not int or frames <= 0
        or type(batch_size) is not int or batch_size <= 0
        or type(total_steps) is not int or total_steps <= 0
        or type(steps) is not int or not 0 < steps <= total_steps
        or type(samples) is not int or samples <= 0
        or type(epoch) is not int or not 0 <= epoch < epochs
        or type(offset) is not int or not 0 < offset <= frames
        or type(completed) is not int or completed != epoch + int(offset == frames)
        or progress.get("milestone_epoch") not in (None, completed)
        or samples != epoch * frames + offset
        or steps != epoch * math.ceil(frames / batch_size) + math.ceil(offset / batch_size)
    ):
        raise ValueError("checkpoint progress counters are absent or inconsistent")


def _rd_source_candidates(
    config: dict[str, Any], config_root: Path, model: str
) -> list[tuple[str, str, list[dict[str, str]]]]:
    model_cfg = _model_config(config, model)
    candidates = [(_rd_config_identity(config, model_cfg, model), "current", _active_rd_records(config, model))]
    records = config.get("resume_compatible_code", [])
    if not isinstance(records, list):
        raise ValueError("resume_compatible_code must be a list")
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("resume-compatible config entry must be an object")
        name = record.get("config_file")
        if not isinstance(name, str) or Path(name).name != name:
            raise ValueError("resume-compatible config filename must be a basename")
        expected_hash = _hash_value(record.get("config_sha256"), "resume-compatible config hash")
        original_path = config_root / name
        if original_path.is_symlink() or not original_path.is_file() or _file_hash(original_path) != expected_hash:
            raise ValueError(f"declared resume-compatible config is missing or changed: {name}")
        original = _read_json(original_path)
        if not isinstance(original, dict) or _semantic_config(original) != _semantic_config(config):
            raise ValueError(f"resume-compatible config changes the trained model/data identity: {name}")
        original_cfg = _model_config(original, model)
        candidates.append((_rd_config_identity(original, original_cfg, model), name, _active_rd_records(original, model)))
    return candidates


def _check_rd_progress(
    config: dict[str, Any],
    config_root: Path,
    source_root: Path,
    model: str,
    progress: dict[str, Any],
    checkpoint: Path,
    *,
    identity_checkpoint: Path | None = None,
) -> None:
    identity = progress["identity"]
    if identity.get("run_id") != config.get("run_id") or identity.get("model") != model:
        raise ValueError("RD-VLA checkpoint run/model lineage differs from training config")
    if identity.get("manifest_sha256") != _hash_value(config.get("manifest_sha256"), "manifest_sha256"):
        raise ValueError("RD-VLA checkpoint dataset manifest differs from training config")
    expected_checkpoint = checkpoint if identity_checkpoint is None else identity_checkpoint
    if progress.get("checkpoint") != str(expected_checkpoint):
        raise ValueError("RD-VLA progress points to a different checkpoint directory")
    candidates = _rd_source_candidates(config, config_root, model)
    saved_files = identity.get("source_files")
    matched = False
    for config_hash, _, source_files in candidates:
        if identity.get("config_sha256") == config_hash and saved_files == source_files:
            matched = True
            break
    if not matched:
        raise ValueError("RD-VLA checkpoint config/source lineage is not current or explicitly compatible")
    current_files = _actual_source_records(source_root, _active_rd_records(config, model), "RD-VLA")
    saved_by_path = {row["path"]: row["sha256"] for row in saved_files}
    current_by_path = {row["path"]: row["sha256"] for row in current_files}
    if set(saved_by_path) != set(current_by_path) or any(
        saved_by_path[name] != current_by_path[name] and name != "train.py" for name in saved_by_path
    ):
        raise ValueError("RD-VLA model/source lineage changed outside the declared launcher compatibility")
    wandb = progress.get("wandb")
    if not isinstance(wandb, dict) or any(wandb.get(key) != config.get(expected) for key, expected in (
        ("entity", "wandb_entity"), ("project", "wandb_project")
    )) or not isinstance(wandb.get("id"), str) or not wandb["id"]:
        raise ValueError("RD-VLA checkpoint W&B identity differs from training config")
    model_cfg = _model_config(config, model)
    expected_run = model_cfg.get("wandb_run_id")
    if expected_run is not None and wandb["id"] != expected_run:
        raise ValueError("RD-VLA checkpoint W&B run ID differs from training config")
    step = progress.get("global_step")
    total_steps = progress.get("total_steps")
    completed = progress.get("completed_epochs")
    samples = progress.get("training_samples")
    epoch_offset = progress.get("epoch_offset")
    frames_per_epoch = progress.get("frames_per_epoch")
    epochs = config.get("epochs")
    if (
        type(step) is not int or step <= 0
        or type(total_steps) is not int or total_steps != identity.get("total_steps") or step > total_steps
        or type(completed) is not int or type(epochs) is not int or not 0 <= completed <= epochs
        or type(samples) is not int or samples <= 0
        or type(epoch_offset) is not int or epoch_offset < 0
        or type(frames_per_epoch) is not int or frames_per_epoch <= 0
    ):
        raise ValueError("RD-VLA checkpoint progress counters are absent or invalid")


def _validate_checkpoint_name(model: str, checkpoint: Path, progress: dict[str, Any], config: dict[str, Any]) -> None:
    if model == "rdvla":
        step = re.fullmatch(r"step-(\d+)", checkpoint.name)
        if step and int(step.group(1)) == progress["global_step"]:
            return
        milestone = re.fullmatch(r"epoch-(\d{2})", checkpoint.name)
        if milestone:
            epoch = int(milestone.group(1))
            if (
                1 <= epoch <= config.get("epochs", 0)
                and progress["completed_epochs"] == epoch
                and progress.get("epoch_offset") == 0
            ):
                return
        raise ValueError("RD-VLA checkpoint directory differs from its committed step/milestone")
    step = re.fullmatch(r"step-(\d+)-[0-9a-f]{32}", checkpoint.name)
    if step:
        if int(step.group(1)) != progress["steps"] or progress["milestone_epoch"] is not None:
            raise ValueError("checkpoint step directory differs from its recorded optimizer step")
        return
    milestone = re.fullmatch(r"epoch-(10|20)", checkpoint.name)
    if milestone:
        epoch = int(milestone.group(1))
        if (
            progress["milestone_epoch"] == epoch
            and progress["completed_epochs"] == epoch
            and progress["epoch_index"] == epoch - 1
            and progress["offset"] == config["frames"]
        ):
            return
    raise ValueError("checkpoint directory differs from its committed step/milestone")


def _progress_and_manifest(
    config: dict[str, Any],
    config_path: Path,
    model: str,
    model_cfg: dict[str, Any],
    checkpoint: Path,
    source_root: Path,
    *,
    identity_checkpoint: Path | None = None,
) -> tuple[dict[str, Any], str]:
    manifest_helper = source_root / "training_state.py"
    if manifest_helper.is_symlink() or not manifest_helper.is_file():
        raise ValueError("native source tree lacks its checkpoint manifest verifier")
    namespace = {"__file__": str(manifest_helper), "__name__": "_so101_training_state"}
    exec(compile(manifest_helper.read_bytes(), str(manifest_helper), "exec"), namespace)
    manifest_hash = namespace["verify_checkpoint_manifest"](checkpoint)
    progress = _read_json(checkpoint / "progress.json")
    _validate_progress_shape(model, progress)
    if model == "rdvla":
        _check_rd_progress(
            config,
            _path(config.get("root"), "training root", strict=False),
            source_root,
            model,
            progress,
            checkpoint,
            identity_checkpoint=identity_checkpoint,
        )
    else:
        _check_hf_progress(config, config_path, model, model_cfg, progress)
    _validate_checkpoint_name(model, checkpoint, progress, config)
    return progress, manifest_hash


def _json_object(path: Path) -> dict[str, Any]:
    value = _read_json(path)
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return value


def _check_policy_contract(model: str, policy_dir: Path) -> None:
    if model == "rdvla":
        value = _json_object(policy_dir / "model_config.json")
        expected = {
            "robot_platform": "SO101",
            "camera_order": list(CAMERAS),
            "image_sequence_len": 2,
            "action_dim": 6,
            "action_chunk_len": PREDICTION_STEPS,
            "execution_horizon": PREDICTION_STEPS,
        }
        if any(value.get(key) != expected_value for key, expected_value in expected.items()):
            raise ValueError("RD-VLA checkpoint export differs from the SO101 50x6/two-camera contract")
    elif model == "anchormix":
        observation = _json_object(policy_dir / "observation_schema.json")
        action = _json_object(policy_dir / "action_schema.json")
        images = observation.get("images")
        state = observation.get("state")
        if (
            observation.get("benchmark") != "so101"
            or observation.get("camera_order") != list(CAMERAS)
            or not isinstance(images, dict)
            or set(images) != set(CAMERAS)
            or any(images[name].get("color_space") != "RGB" for name in CAMERAS)
            or not isinstance(state, dict)
            or state.get("dimension") != 6
            or state.get("channel_order") != list(JOINTS)
            or action.get("benchmark") != "so101"
            or action.get("action_dim") != 6
            or action.get("action_horizon") != PREDICTION_STEPS
            or action.get("execution_horizon") != PREDICTION_STEPS
            or action.get("channel_order") != list(JOINTS)
        ):
            raise ValueError("AnchorMix checkpoint export differs from the SO101 50x6/RGB contract")
    else:
        processors = _json_object(policy_dir / "processors.json")
        config = _json_object(policy_dir / "config.json")
        expected = {
            "robot_type": "SO101",
            "camera_names": list(CAMERAS),
            "joint_names": list(JOINTS),
            "state_dim": 6,
            "action_dim": 6,
            "chunk_size": PREDICTION_STEPS,
            "execution_horizon": PREDICTION_STEPS,
        }
        if any(processors.get(key) != expected_value for key, expected_value in expected.items()):
            raise ValueError("PI05 processor export differs from the SO101 50x6 contract")
        input_features = config.get("input_features")
        output_features = config.get("output_features")
        if (
            config.get("chunk_size") != PREDICTION_STEPS
            or config.get("n_action_steps") != PREDICTION_STEPS
            or config.get("action_feature_names") != list(JOINTS)
            or not isinstance(input_features, dict)
            or set(input_features) != {"observation.state", "observation.images.top", "observation.images.wrist"}
            or input_features["observation.state"].get("shape") != [6]
            or not isinstance(output_features, dict)
            or output_features.get("action", {}).get("shape") != [6]
        ):
            raise ValueError("PI05 checkpoint config differs from the SO101 state/action feature contract")


def _deployment_name(model: str, config: dict[str, Any], checkpoint: Path) -> str:
    run_id = config.get("run_id")
    if not isinstance(run_id, str) or not run_id:
        raise ValueError("training config run_id is required")
    value = re.sub(r"[^A-Za-z0-9._-]+", "-", f"so101-{model}-{run_id}-{checkpoint.name}").strip("-.")
    if not value or len(value) > 256:
        value = f"so101-{model}-{hashlib.sha256(value.encode()).hexdigest()[:24]}"
    return value


def _build_descriptor(
    model: str,
    config_path: Path,
    checkpoint: Path,
    *,
    runtime: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if model not in MODELS:
        raise ValueError(f"model must be one of {MODELS}")
    config_path = Path(config_path).expanduser()
    if config_path.is_symlink() or not config_path.is_file():
        raise ValueError("training config must be a regular non-symlink file")
    config_path = config_path.resolve(strict=True)
    config = _json_object(config_path)
    if runtime is not None:
        if model != "rdvla":
            raise ValueError("runtime relocation is supported only for RD-VLA")
        runtime = _normalize_runtime(runtime)
    model_cfg, root, source_root, pythonpath = _verify_model_settings(config, model, runtime=runtime)
    if config.get("action_horizon") != PREDICTION_STEPS or config.get("execution_horizon") != PREDICTION_STEPS:
        raise ValueError("training config must keep both action horizons at 50")
    checkpoint_input = Path(checkpoint).expanduser()
    if checkpoint_input.is_symlink() or not checkpoint_input.is_dir():
        raise ValueError("checkpoint must be an explicit committed directory, not a symlink/latest pointer")
    checkpoint = checkpoint_input.resolve(strict=True)
    identity_checkpoint = None
    if runtime is None:
        configured_root, _ = _verify_checkpoint_root(config, model, checkpoint)
        if configured_root != root:
            raise ValueError("checkpoint training root changed during descriptor preparation")
    else:
        _verify_runtime_checkpoint(runtime, checkpoint)
        identity_checkpoint = _original_rd_checkpoint(config, checkpoint)
    source_sha256 = _verify_source_lineage(config, model, source_root)
    progress, manifest_hash = _progress_and_manifest(
        config,
        config_path,
        model,
        model_cfg,
        checkpoint,
        source_root,
        identity_checkpoint=identity_checkpoint,
    )
    _check_policy_contract(model, checkpoint / "policy")
    gpu_uuid = config.get("gpu_uuid") if runtime is None else runtime["gpu_uuid"]
    if not isinstance(gpu_uuid, str) or not re.fullmatch(r"GPU-[A-Za-z0-9-]+", gpu_uuid):
        raise ValueError("training config must pin a physical GPU UUID")
    training_config_sha256 = _file_hash(config_path)
    spec: dict[str, Any] = {
        "schema": SCHEMA,
        "model": model,
        "deployment": _deployment_name(model, config, checkpoint),
        "artifact": "",
        "semantics": SEMANTICS,
        "checkpoint": str(checkpoint),
        "training_config": str(config_path),
        "python": str(_executable(model_cfg.get("python"), "training python"))
        if runtime is None else runtime["python"],
        "source_root": str(source_root),
        "pythonpath": pythonpath,
        "gpu_uuid": gpu_uuid,
        "progress": progress,
        "checkpoint_manifest_sha256": manifest_hash,
        "training_config_sha256": training_config_sha256,
        "source_sha256": source_sha256,
    }
    if runtime is not None:
        spec["runtime"] = runtime
    spec["artifact"] = _deployment_identity(spec)
    _validate_descriptor(spec, model=model)
    return spec


def _write_descriptor_no_replace(path: Path, value: dict[str, Any]) -> None:
    path = Path(os.path.abspath(Path(path).expanduser()))
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False).encode("utf-8") + b"\n"
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def prepare_deployment(
    model: str,
    config_path: Path,
    checkpoint: Path,
    output: Path,
    *,
    runtime: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Verify one committed checkpoint and write a no-overwrite deployment descriptor."""
    destination = Path(os.path.abspath(Path(output).expanduser()))
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"refusing to overwrite deployment descriptor: {destination}")
    spec = _build_descriptor(model, Path(config_path), Path(checkpoint), runtime=runtime)
    _write_descriptor_no_replace(destination, spec)
    print(
        f"Prepared {model} deployment {spec['deployment']} ({spec['artifact']}) at {destination}",
        file=sys.stderr,
    )
    return spec


def _read_exact(stream, size: int) -> bytes:
    chunks = bytearray()
    while len(chunks) < size:
        block = stream.read(size - len(chunks))
        if not block:
            raise EOFError("worker input pipe closed mid-frame")
        chunks.extend(block)
    return bytes(chunks)


def _read_input(stream) -> tuple[dict[str, Any], bytearray] | None:
    prefix = stream.read(4)
    if not prefix:
        return None
    if len(prefix) != 4:
        prefix += _read_exact(stream, 4 - len(prefix))
    header_length = struct.unpack("!I", prefix)[0]
    if not 1 <= header_length <= MAX_HEADER_BYTES:
        raise ValueError("worker request header exceeds the bounded protocol size")
    header = _json_bytes(_read_exact(stream, header_length))
    if not isinstance(header, dict):
        raise ValueError("worker request header must be an object")
    payload_length = header.get("payload_bytes")
    if type(payload_length) is not int or not 0 <= payload_length <= INPUT_BYTES:
        raise ValueError("worker request payload length is invalid")
    return header, bytearray(_read_exact(stream, payload_length))


def _write_reply(fd: int, header: dict[str, Any], payload: bytes = b"") -> None:
    header = dict(header)
    header["payload_bytes"] = len(payload)
    encoded = json.dumps(header, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    if not 1 <= len(encoded) <= MAX_HEADER_BYTES or len(payload) > 1200:
        raise ValueError("worker reply exceeds the bounded protocol size")
    buffers = (memoryview(struct.pack("!I", len(encoded))), memoryview(encoded), memoryview(payload))
    for buffer in buffers:
        while buffer:
            try:
                written = os.write(fd, buffer)
            except InterruptedError:
                continue
            if written <= 0:
                raise BrokenPipeError("worker reply pipe closed")
            buffer = buffer[written:]


def _response_read(fd: int, size: int, deadline: float) -> bytearray:
    result = bytearray()
    with selectors.DefaultSelector() as selector:
        selector.register(fd, selectors.EVENT_READ)
        while len(result) < size:
            remaining = deadline - time.monotonic()
            if remaining <= 0 or not selector.select(remaining):
                raise TimeoutError("native adapter worker response timed out")
            try:
                block = os.read(fd, size - len(result))
            except BlockingIOError:
                continue
            if not block:
                raise EOFError("native adapter worker closed its reply pipe")
            result.extend(block)
    return result


def _receive_reply(fd: int, timeout: float) -> tuple[dict[str, Any], bytearray]:
    deadline = time.monotonic() + timeout
    prefix = _response_read(fd, 4, deadline)
    header_length = struct.unpack("!I", prefix)[0]
    if not 1 <= header_length <= MAX_HEADER_BYTES:
        raise ValueError("native adapter reply header exceeds the bounded protocol size")
    header = _json_bytes(_response_read(fd, header_length, deadline))
    if not isinstance(header, dict):
        raise ValueError("native adapter reply header must be an object")
    payload_length = header.get("payload_bytes")
    if type(payload_length) is not int or not 0 <= payload_length <= 1200:
        raise ValueError("native adapter reply payload length is invalid")
    return header, _response_read(fd, payload_length, deadline)


def _send_request(proc: subprocess.Popen, header: dict[str, Any], payloads: tuple[bytes | memoryview, ...], timeout: float) -> None:
    header = dict(header)
    header["payload_bytes"] = sum(len(item) for item in payloads)
    if not 0 <= header["payload_bytes"] <= INPUT_BYTES:
        raise ValueError("native adapter request exceeds its bounded payload size")
    encoded = json.dumps(header, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")
    if not 1 <= len(encoded) <= MAX_HEADER_BYTES:
        raise ValueError("native adapter request header exceeds the bounded protocol size")
    chunks = [memoryview(struct.pack("!I", len(encoded))), memoryview(encoded), *(memoryview(item) for item in payloads)]
    fd = proc.stdin.fileno()
    os.set_blocking(fd, False)
    deadline = time.monotonic() + timeout
    with selectors.DefaultSelector() as selector:
        selector.register(fd, selectors.EVENT_WRITE)
        for chunk in chunks:
            while chunk:
                remaining = deadline - time.monotonic()
                if remaining <= 0 or not selector.select(remaining):
                    raise TimeoutError("native adapter worker request timed out")
                try:
                    written = os.write(fd, chunk)
                except BlockingIOError:
                    continue
                if written <= 0:
                    raise BrokenPipeError("native adapter worker closed its request pipe")
                chunk = chunk[written:]


def _verify_full_descriptor(spec: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    _validate_descriptor(spec, model=spec.get("model"))
    model = spec["model"]
    config_value = Path(spec["training_config"])
    if config_value.is_symlink():
        raise ValueError("training config path must not be a symlink")
    config_path = _path(spec["training_config"], "training_config")
    if not config_path.is_file() or _file_hash(config_path) != spec["training_config_sha256"]:
        raise ValueError("training config changed since descriptor preparation")
    config = _json_object(config_path)
    runtime = spec.get("runtime")
    if runtime is not None:
        normalized_runtime = _normalize_runtime(runtime)
        if normalized_runtime != runtime:
            raise ValueError("runtime settings changed since descriptor preparation")
        runtime = normalized_runtime
    model_cfg, root, source_root, pythonpath = _verify_model_settings(config, model, runtime=runtime)
    checkpoint_value = Path(spec["checkpoint"])
    if checkpoint_value.is_symlink():
        raise ValueError("checkpoint path must not be a symlink")
    checkpoint = _path(spec["checkpoint"], "checkpoint")
    if not checkpoint.is_dir():
        raise ValueError("checkpoint is not a regular committed directory")
    identity_checkpoint = None
    if runtime is None:
        configured_root, _ = _verify_checkpoint_root(config, model, checkpoint)
        if configured_root != root:
            raise ValueError("checkpoint root differs from training config")
    else:
        _verify_runtime_checkpoint(runtime, checkpoint)
        identity_checkpoint = _original_rd_checkpoint(config, checkpoint)
    expected_source_hash = _verify_source_lineage(config, model, source_root)
    if str(source_root) != spec["source_root"] or pythonpath != spec["pythonpath"]:
        raise ValueError("native source root/PYTHONPATH differs from the pinned config")
    if runtime is None:
        if str(_executable(model_cfg.get("python"), "training python")) != spec["python"]:
            raise ValueError("native Python interpreter differs from the pinned config")
        expected_gpu_uuid = config.get("gpu_uuid")
    else:
        expected_gpu_uuid = runtime["gpu_uuid"]
    if expected_gpu_uuid != spec["gpu_uuid"] or expected_source_hash != spec["source_sha256"]:
        raise ValueError("GPU or native source lineage differs from the descriptor")
    progress, manifest_hash = _progress_and_manifest(
        config,
        config_path,
        model,
        model_cfg,
        checkpoint,
        source_root,
        identity_checkpoint=identity_checkpoint,
    )
    if progress != spec["progress"] or manifest_hash != spec["checkpoint_manifest_sha256"]:
        raise ValueError("checkpoint progress/manifest changed since descriptor preparation")
    _check_policy_contract(model, checkpoint / "policy")
    if _deployment_name(model, config, checkpoint) != spec["deployment"]:
        raise ValueError("deployment name does not identify the configured run/checkpoint")
    return config, model_cfg


def _worker_batch(payload: bytearray, task: str, torch, np) -> dict[str, Any]:
    if len(payload) != INPUT_BYTES:
        raise ValueError("SO101 observation payload must contain state and exactly two RGB frames")
    view = memoryview(payload)
    state = np.frombuffer(view[:STATE_BYTES], dtype=np.float32).reshape(1, 6)
    if not np.isfinite(state).all():
        raise ValueError("SO101 state contains non-finite values")
    batch_images: dict[str, Any] = {}
    offset = STATE_BYTES
    for camera in CAMERAS:
        image = np.frombuffer(view[offset : offset + IMAGE_BYTES], dtype=np.uint8).reshape(IMAGE_SHAPE)
        tensor = torch.from_numpy(image).permute(2, 0, 1).contiguous().to(torch.float32).div_(255).unsqueeze(0)
        batch_images[camera] = tensor
        offset += IMAGE_BYTES
    return {
        "state": torch.from_numpy(state),
        "images": batch_images,
        "task": [task],
    }


def _worker_main(reply_fd: int) -> int:
    os.dup2(2, 1)
    sys.stdout = sys.stderr
    try:
        first = _read_input(sys.stdin.buffer)
        if first is None:
            return 2
        init, payload = first
        if init.get("op") != "init" or payload or set(init) != {"op", "payload_bytes", "spec", "device", "execution_horizon", "cpu_threads", "memory_limit_gib"}:
            raise ValueError("invalid native adapter worker initialization frame")
        spec = _validate_descriptor(init["spec"], model=init["spec"].get("model"))
        if init["device"] not in {"cpu", "cuda:0"}:
            raise ValueError("native adapter worker only supports explicit cpu or cuda:0 devices")
        execution_horizon = init["execution_horizon"]
        if type(execution_horizon) is not int or not 1 <= execution_horizon <= PREDICTION_STEPS:
            raise ValueError("execution_horizon must be in [1, 50]")
        cpu_threads = init["cpu_threads"]
        if type(cpu_threads) is not int or cpu_threads <= 0:
            raise ValueError("cpu_threads must be a positive integer")
        memory_limit_gib = validate_memory_limit(init["memory_limit_gib"])
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
        sys.path[:0] = spec["pythonpath"]
        import numpy as np
        import torch

        torch.set_num_threads(min(4, cpu_threads))
        if init["device"] == "cpu":
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
        elif not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable; refusing a CPU or alternate-GPU fallback")
        _verify_full_descriptor(spec)
        if init["device"] == "cuda:0":
            apply_cuda_memory_limit(torch, memory_limit_gib)
        adapter_path = Path(spec["source_root"]) / f"{spec['model']}_adapter.py"
        module_name = "_so101_native_" + spec["model"]
        adapter_spec = importlib.util.spec_from_file_location(module_name, adapter_path)
        if adapter_spec is None or adapter_spec.loader is None:
            raise RuntimeError(f"cannot load native {spec['model']} adapter")
        adapter_module = importlib.util.module_from_spec(adapter_spec)
        sys.modules[module_name] = adapter_module
        adapter_spec.loader.exec_module(adapter_module)
        adapter = adapter_module.load(Path(spec["checkpoint"]) / "policy", torch.device(init["device"]))
        if getattr(adapter, "horizon", None) != PREDICTION_STEPS or getattr(adapter, "execution_horizon", None) != PREDICTION_STEPS:
            raise ValueError("native adapter does not retain its trained 50-step horizon")
        native_model = getattr(adapter, "model", None)
        if native_model is not None and callable(getattr(native_model, "eval", None)):
            native_model.eval()
        _write_reply(reply_fd, {"ok": True})
        while True:
            message = _read_input(sys.stdin.buffer)
            if message is None:
                raise EOFError("native adapter request pipe reached EOF")
            header, body = message
            if header.get("op") == "close" and not body:
                _write_reply(reply_fd, {"ok": True})
                return 0
            if header.get("op") != "predict" or set(header) != {
                "op", "payload_bytes", "task", "capture_time", "task_version", "observation_id"
            } or len(body) != INPUT_BYTES:
                raise ValueError("invalid native adapter prediction frame")
            task = header["task"]
            if not isinstance(task, str) or not task.strip() or len(task) > MAX_TASK_CHARS:
                raise ValueError("task must be nonempty and at most 4096 characters")
            if (
                not isinstance(header["capture_time"], (int, float))
                or not math.isfinite(header["capture_time"])
                or type(header["task_version"]) is not int
                or header["task_version"] < 0
                or not isinstance(header["observation_id"], str)
            ):
                raise ValueError("observation provenance metadata is invalid")
            batch = _worker_batch(body, task, torch, np)
            started = time.perf_counter()
            if spec["model"] == "anchormix" and init["device"] == "cuda:0":
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    prediction = adapter.predict(batch)
            else:
                prediction = adapter.predict(batch)
            if not isinstance(prediction, torch.Tensor) or tuple(prediction.shape) != (1, PREDICTION_STEPS, 6):
                raise RuntimeError("native adapter returned an action tensor outside [1,50,6]")
            if not bool(torch.isfinite(prediction).all().item()):
                raise FloatingPointError("native adapter returned non-finite actions")
            actions = prediction[0, :execution_horizon].to(device="cpu", dtype=torch.float32).contiguous()
            duration = time.perf_counter() - started
            _write_reply(
                reply_fd,
                {"ok": True, "steps": execution_horizon, "duration": duration},
                actions.numpy().tobytes(order="C"),
            )
    except BaseException as exc:
        try:
            _write_reply(reply_fd, {"ok": False, "error": f"{type(exc).__name__}: {exc}"[:2048]})
        except (BrokenPipeError, OSError, ValueError):
            pass
        print(f"SO101 native adapter worker failed: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        return 1
    finally:
        try:
            os.close(reply_fd)
        except OSError:
            pass


def _worker_cli() -> int:
    if len(sys.argv) != 3 or sys.argv[1] != "--adapter-worker":
        return 2
    try:
        return _worker_main(int(sys.argv[2]))
    except (ValueError, OSError) as exc:
        print(f"SO101 native adapter worker startup failed: {exc}", file=sys.stderr, flush=True)
        return 2


class AdapterRunner:
    """LeRobot PolicyRunner-compatible chunk runner backed by the training Python."""

    def __init__(
        self,
        spec: dict[str, Any],
        *,
        features: tuple,
        action_feature,
        fps: float,
        execution_horizon: int,
        device: str = "cuda:0",
        memory_limit_gib: float | None = None,
    ) -> None:
        _validate_descriptor(spec, model=spec.get("model") if isinstance(spec, dict) else None)
        if device not in {"cpu", "cuda:0"}:
            raise ValueError("device must be explicit 'cuda:0' or 'cpu'; no fallback is provided")
        if not isinstance(fps, (int, float)) or not math.isfinite(fps) or fps <= 0:
            raise ValueError("fps must be finite and positive")
        if type(execution_horizon) is not int or not 1 <= execution_horizon <= PREDICTION_STEPS:
            raise ValueError("execution_horizon must be an integer from 1 through 50")
        memory_limit_gib = validate_memory_limit(memory_limit_gib)
        features = tuple(features)
        expected_names = ("observation.state", "observation.images.top", "observation.images.wrist")
        if tuple(getattr(item, "name", None) for item in features) != expected_names:
            raise ValueError("AdapterRunner requires state, top RGB, wrist RGB in canonical feature order")
        expected_shapes = ((6,), IMAGE_SHAPE, IMAGE_SHAPE)
        expected_dtypes = ("float32", "uint8", "uint8")
        if any(tuple(getattr(item, "shape", ())) != shape or getattr(item, "dtype", None) != dtype for item, shape, dtype in zip(features, expected_shapes, expected_dtypes)):
            raise ValueError("AdapterRunner feature shapes/dtypes differ from the SO101 observation contract")
        if tuple(getattr(features[0], "names", ())) != JOINTS:
            raise ValueError("SO101 state channels must be in the recorded six-joint order")
        if any(getattr(item, "semantics", None) != SEMANTICS for item in (*features, action_feature)):
            raise ValueError("SO101 observation/action feature semantics differ from the deployment")
        if getattr(features[0], "kind", "tensor") != "tensor" or any(
            getattr(features[index], "kind", "tensor") != "rgb" for index in (1, 2)
        ):
            raise ValueError("SO101 state and camera feature kinds differ from the deployment contract")
        if (
            getattr(action_feature, "name", None) != "action"
            or tuple(getattr(action_feature, "shape", ())) != (6,)
            or getattr(action_feature, "dtype", None) != "float32"
            or getattr(action_feature, "kind", "tensor") != "tensor"
            or tuple(getattr(action_feature, "names", ())) != JOINTS
        ):
            raise ValueError("AdapterRunner requires the six recorded-order float32 SO101 actions")
        from lerobot.inference import ActionChunk, ActionProvenance, ExecutionMode, PolicyCapabilities

        self._ActionChunk, self._ActionProvenance = ActionChunk, ActionProvenance
        self._ExecutionMode = ExecutionMode
        self._spec = dict(spec)
        self._features = features
        self._execution_horizon = execution_horizon
        self.capabilities = PolicyCapabilities(
            modes=(ExecutionMode.CHUNK,),
            prediction_steps=PREDICTION_STEPS,
            execution_steps=execution_horizon,
            action_interval=1.0 / float(fps),
            features=features,
            action_feature=action_feature,
            language=False,
            current_observation_only=True,
            retains_session_state=False,
        )
        config_threads = 4
        try:
            # Only static JSON config data is read in this process; native model imports stay in the child.
            config = _json_object(Path(spec["training_config"]))
            configured_threads = config.get("cpu_threads", 4)
            if type(configured_threads) is int and configured_threads > 0:
                config_threads = min(4, configured_threads)
        except (OSError, ValueError):
            pass
        self._cpu_threads = config_threads
        self._lock = threading.Lock()
        self._termination_lock = threading.Lock()
        self._closed = False
        self._reply_fd, reply_write_fd = os.pipe()
        os.set_blocking(self._reply_fd, False)
        env = os.environ.copy()
        for key in (
            "PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP", "PYTHONINSPECT", "PYTHONUSERBASE",
            "CUDA_MPS_PIPE_DIRECTORY", "CUDA_MPS_LOG_DIRECTORY",
        ):
            env.pop(key, None)
        env["PYTHONNOUSERSITE"] = "1"
        env["HF_HUB_OFFLINE"] = "1"
        env["TRANSFORMERS_OFFLINE"] = "1"
        env["HF_HUB_DISABLE_TELEMETRY"] = "1"
        env["CUDA_VISIBLE_DEVICES"] = "" if device == "cpu" else spec["gpu_uuid"]
        env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        try:
            self._proc = subprocess.Popen(
                [spec["python"], "-I", "-B", "-u", str(Path(__file__).resolve()), "--adapter-worker", str(reply_write_fd)],
                stdin=subprocess.PIPE,
                stdout=None,
                stderr=None,
                cwd=spec["source_root"],
                env=env,
                close_fds=True,
                pass_fds=(reply_write_fd,),
            )
        except BaseException:
            os.close(self._reply_fd)
            os.close(reply_write_fd)
            raise
        finally:
            try:
                os.close(reply_write_fd)
            except OSError:
                pass
        if self._proc.stdin is None:
            self.close()
            raise RuntimeError("native adapter worker has no private request pipe")
        os.set_blocking(self._proc.stdin.fileno(), False)
        try:
            self._request(
                {
                    "op": "init",
                    "spec": self._spec,
                    "device": device,
                    "execution_horizon": execution_horizon,
                    "cpu_threads": self._cpu_threads,
                    "memory_limit_gib": memory_limit_gib,
                },
                (),
                START_TIMEOUT,
            )
        except BaseException:
            self._terminate_worker()
            raise


    def _request(self, header: dict[str, Any], payloads: tuple[bytes | memoryview, ...], timeout: float):
        if self._closed or self._proc.poll() is not None:
            raise RuntimeError("native adapter worker is closed or has exited")
        _send_request(self._proc, header, payloads, timeout)
        reply, payload = _receive_reply(self._reply_fd, timeout)
        if reply.get("ok") is not True:
            raise RuntimeError(f"native adapter worker rejected request: {reply.get('error', 'unknown failure')}")
        return reply, payload

    def predict(
        self,
        observation,
        *,
        mode=None,
        inference_delay: int = 0,
        model_continuation=None,
        canonical_continuation=None,
        provenance=None,
    ):
        mode = self._ExecutionMode.CHUNK if mode is None else self._ExecutionMode(mode)
        if mode != self._ExecutionMode.CHUNK:
            raise ValueError("native adapters support CHUNK execution only")
        expected = {item.name: item for item in self._features}
        if set(observation.features) != set(expected):
            raise ValueError("observation features differ from the SO101 contract")
        arrays = []
        for name in ("observation.state", "observation.images.top", "observation.images.wrist"):
            value = observation.features[name]
            array = value
            if tuple(array.shape) != tuple(expected[name].shape) or str(array.dtype) != expected[name].dtype:
                raise ValueError(f"observation feature has wrong shape or dtype: {name}")
            if name == "observation.state":
                import numpy as np

                if not np.isfinite(array).all():
                    raise ValueError("SO101 state contains non-finite values")
                arrays.append(memoryview(array).cast("B") if array.flags.c_contiguous else memoryview(np.ascontiguousarray(array)).cast("B"))
            else:
                arrays.append(memoryview(array).cast("B") if array.flags.c_contiguous else memoryview(array.copy(order="C")).cast("B"))
        task = observation.task
        if not isinstance(task, str) or not task.strip() or len(task) > MAX_TASK_CHARS:
            raise ValueError("task must be nonempty and at most 4096 characters")
        header = {
            "op": "predict",
            "task": task,
            "capture_time": float(observation.capture_time),
            "task_version": int(observation.task_version),
            "observation_id": str(observation.observation_id),
        }
        started = time.perf_counter()
        with self._lock:
            try:
                reply, payload = self._request(header, tuple(arrays), REQUEST_TIMEOUT)
            except BaseException:
                self._terminate_worker()
                raise
        if reply.get("steps") != self._execution_horizon or len(payload) != self._execution_horizon * 6 * 4:
            self._terminate_worker()
            raise RuntimeError("native adapter worker returned the wrong action chunk length")
        import numpy as np
        import torch

        actions = np.frombuffer(payload, dtype=np.float32).reshape(self._execution_horizon, 6)
        if not np.isfinite(actions).all():
            self._terminate_worker()
            raise FloatingPointError("native adapter worker returned non-finite actions")
        if provenance is None:
            provenance = self._ActionProvenance(
                capture_time=float(observation.capture_time),
                task=observation.task,
                task_version=int(observation.task_version),
                observation_id=str(observation.observation_id),
            )
        return self._ActionChunk(
            model_actions=None,
            canonical_actions=torch.from_numpy(actions),
            provenance=provenance,
            execution_steps=self._execution_horizon,
            server_durations={"adapter_worker_roundtrip": time.perf_counter() - started},
        )

    def reset(self, *, full: bool = True) -> None:
        if type(full) is not bool:
            raise TypeError("full must be a bool")
        if self._closed:
            raise RuntimeError("native adapter worker is closed")
        # These three native SO101 adapters are current-observation/stateless policies.

    def query(self, observation, *, kind: str, text: str) -> str:
        raise ValueError("native SO101 adapters do not support language queries")

    def _terminate_worker(self) -> None:
        with self._termination_lock:
            if self._closed:
                return
            self._closed = True
            proc = getattr(self, "_proc", None)
            try:
                if proc is not None and proc.poll() is None:
                    try:
                        proc.terminate()
                    except ProcessLookupError:
                        pass
                    try:
                        proc.wait(timeout=CLOSE_TIMEOUT)
                    except subprocess.TimeoutExpired:
                        try:
                            proc.kill()
                        except ProcessLookupError:
                            pass
                        try:
                            proc.wait(timeout=CLOSE_TIMEOUT)
                        except subprocess.TimeoutExpired:
                            pass
            finally:
                if proc is not None and proc.stdin is not None:
                    try:
                        proc.stdin.close()
                    except OSError:
                        pass
                fd = getattr(self, "_reply_fd", None)
                if fd is not None:
                    self._reply_fd = None
                    try:
                        os.close(fd)
                    except OSError:
                        pass

    def close(self) -> None:
        lock = getattr(self, "_lock", None)
        if lock is None or not lock.acquire(blocking=False):
            self._terminate_worker()
            return
        try:
            if self._closed:
                return
            try:
                if self._proc.poll() is None:
                    self._request({"op": "close"}, (), CLOSE_TIMEOUT)
            except (BrokenPipeError, EOFError, OSError, RuntimeError, TimeoutError, ValueError):
                pass
            finally:
                self._terminate_worker()
        finally:
            lock.release()


if __name__ == "__main__":
    raise SystemExit(_worker_cli())
