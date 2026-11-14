from __future__ import annotations

import contextlib
import enum
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import types
import unittest
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import so101_adapters as adapters


class AdapterDescriptorTests(unittest.TestCase):
    def test_module_import_is_stdlib_only(self):
        module_path = Path(adapters.__file__).resolve()
        code = (
            "import importlib.util,sys; "
            "s=importlib.util.spec_from_file_location('probe',sys.argv[1]); "
            "m=importlib.util.module_from_spec(s); s.loader.exec_module(m); "
            "assert not {'torch','numpy','lerobot'} & sys.modules.keys()"
        )
        subprocess.run([sys.executable, "-I", "-S", "-c", code, str(module_path)], check=True)

    def test_virtualenv_interpreter_symlink_is_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            interpreter = Path(directory) / "venv-python"
            interpreter.symlink_to(sys.executable)
            self.assertEqual(adapters._executable(str(interpreter), "python"), interpreter)

    def test_pi05_binds_external_native_source_without_inventing_a_copy(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "source"
            root.mkdir()
            native = Path(directory) / "native-LeRobot"
            (native / "src/lerobot").mkdir(parents=True)
            (native / "pyproject.toml").write_text("[project]\nname='fixture'\n", encoding="utf-8")
            (native / "src/lerobot/__init__.py").write_text("__version__ = 'fixture'\n", encoding="utf-8")
            native_policy = native / "src/lerobot/native.py"
            native_policy.write_text("NATIVE = True\n", encoding="utf-8")
            adapter_file = root / "pi05_adapter.py"
            adapter_file.write_text("# adapter fixture\n", encoding="utf-8")
            config = {
                "source_sha256": adapters._source_tree_identity(root),
                "lerobot_root": str(native),
            }
            original_binding = adapters._verify_source_lineage(config, "pi05", root)
            native_policy.write_text("NATIVE = False\n", encoding="utf-8")
            self.assertNotEqual(original_binding, adapters._verify_source_lineage(config, "pi05", root))
            adapter_file.write_text("# changed adapter\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "source differs"):
                adapters._verify_source_lineage(config, "pi05", root)

    def test_step_and_milestone_names_match_progress(self):
        adapters._validate_checkpoint_name(
            "rdvla",
            Path("epoch-01"),
            {"global_step": 10, "completed_epochs": 1, "epoch_offset": 0},
            {"epochs": 20},
        )
        adapters._validate_checkpoint_name(
            "anchormix",
            Path("epoch-10"),
            {"steps": 10, "milestone_epoch": 10, "completed_epochs": 10, "epoch_index": 9, "offset": 100},
            {"frames": 100},
        )
        with self.assertRaises(ValueError):
            adapters._validate_checkpoint_name(
                "anchormix",
                Path("epoch-10"),
                {"steps": 10, "milestone_epoch": None, "completed_epochs": 9, "epoch_index": 8, "offset": 100},
                {"frames": 100},
            )

    def test_copied_descriptor_checks_binding_without_server_files(self):
        progress = {
            "schema": 1,
            "status": "checkpoint_committed",
            "identity": {"run_id": "r", "model": "rdvla", "source_files": []},
            "wandb": {},
        }
        spec = {
            "schema": adapters.SCHEMA,
            "model": "rdvla",
            "deployment": "so101-rdvla-r-step-1",
            "artifact": "",
            "semantics": adapters.SEMANTICS,
            "checkpoint": "/server/not-mounted/checkpoint",
            "training_config": "/server/not-mounted/config.json",
            "python": "/server/not-mounted/python",
            "source_root": "/server/not-mounted/source",
            "pythonpath": ["/server/not-mounted/source"],
            "gpu_uuid": "GPU-00000000-0000-0000-0000-000000000000",
            "progress": progress,
            "checkpoint_manifest_sha256": "a" * 64,
            "training_config_sha256": "b" * 64,
            "source_sha256": "c" * 64,
        }
        spec["artifact"] = adapters._deployment_identity(spec)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "deployment.json"
            path.write_text(json.dumps(spec), encoding="utf-8")
            self.assertEqual(adapters.load_deployment(path, model="rdvla"), spec)
            with self.assertRaises(ValueError):
                adapters.load_deployment(path, model="pi05")
            spec["gpu_uuid"] = "GPU-11111111-1111-1111-1111-111111111111"
            path.write_text(json.dumps(spec), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "self-binding"):
                adapters.load_deployment(path, model="rdvla")

    def test_prepare_binds_committed_manifest_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            root = base / "run"
            source = root / "source"
            source.mkdir(parents=True)
            # A synthetic native source tree keeps this regression independent of private runs.
            (source / "training_state.py").write_text(
                "import hashlib, json\n"
                "def verify_checkpoint_manifest(checkpoint):\n"
                "    manifest = checkpoint / 'fixture-manifest.json'\n"
                "    for name, expected in json.loads(manifest.read_text()).items():\n"
                "        if hashlib.sha256((checkpoint / name).read_bytes()).hexdigest() != expected:\n"
                "            raise ValueError('checkpoint file differs from committed manifest')\n"
                "    return hashlib.sha256(manifest.read_bytes()).hexdigest()\n",
                encoding="utf-8",
            )
            (source / "anchormix_adapter.py").write_text("# source identity fixture\n", encoding="utf-8")
            joint_names = list(adapters.JOINTS)
            checkpoint_name = "step-00000001-" + "a" * 32
            policy = root / "runs/anchormix/checkpoints" / checkpoint_name / "policy"
            policy.mkdir(parents=True)
            (policy / "observation_schema.json").write_text(
                json.dumps(
                    {
                        "benchmark": "so101",
                        "camera_order": ["top", "wrist"],
                        "images": {name: {"color_space": "RGB"} for name in ("top", "wrist")},
                        "state": {"dimension": 6, "channel_order": joint_names},
                    }
                ),
                encoding="utf-8",
            )
            (policy / "action_schema.json").write_text(
                json.dumps(
                    {
                        "benchmark": "so101",
                        "action_dim": 6,
                        "action_horizon": 50,
                        "execution_horizon": 50,
                        "channel_order": joint_names,
                    }
                ),
                encoding="utf-8",
            )
            config = {
                "run_id": "fixture-run",
                "root": str(root),
                "gpu_uuid": "GPU-00000000-0000-0000-0000-000000000000",
                "action_horizon": 50,
                "execution_horizon": 50,
                "epochs": 20,
                "frames": 2,
                "batch_size": 1,
                "total_steps": 40,
                "data_sha256": "a" * 64,
                "normalization_sha256": "b" * 64,
                "wandb_entity": "fixture-entity",
                "wandb_project": "fixture-project",
                "source_sha256": adapters._source_tree_identity(source),
                "models": {
                    "anchormix": {"python": sys.executable, "wandb_run_id": "fixture-run-id"}
                },
            }
            config_path = base / "config.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")
            checkpoint = policy.parent
            progress = {
                "schema": "so101.training.v1",
                "model": "anchormix",
                "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
                "source_sha256": config["source_sha256"],
                "data_sha256": config["data_sha256"],
                "normalization_sha256": config["normalization_sha256"],
                "wandb_run_id": "fixture-run-id",
                "wandb_entity": config["wandb_entity"],
                "wandb_project": config["wandb_project"],
                "steps": 1,
                "training_samples": 1,
                "epoch_index": 0,
                "offset": 1,
                "completed_epochs": 0,
                "milestone_epoch": None,
            }
            (checkpoint / "progress.json").write_text(json.dumps(progress), encoding="utf-8")
            manifest = {
                str(path.relative_to(checkpoint)): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in checkpoint.rglob("*") if path.is_file()
            }
            (checkpoint / "fixture-manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            output = base / "deployment.json"
            with contextlib.redirect_stderr(io.StringIO()):
                spec = adapters.prepare_deployment("anchormix", config_path, checkpoint, output)
            self.assertEqual(adapters.load_deployment(output, model="anchormix"), spec)
            original = output.read_bytes()
            with self.assertRaises(FileExistsError):
                adapters.prepare_deployment("anchormix", config_path, checkpoint, output)
            self.assertEqual(output.read_bytes(), original)
            progress["steps"] = 2
            (checkpoint / "progress.json").write_text(json.dumps(progress), encoding="utf-8")
            with self.assertRaises(ValueError):
                adapters._verify_full_descriptor(spec)

    def _rd_runtime_fixture(self, base: Path, *, adapter_source: str | None = None):
        base = base.resolve()
        original_root = base / "training"
        original_source = original_root / "source"
        (original_source / "RD-VLA").mkdir(parents=True)
        (original_source / "training_state.py").write_text(
            "import hashlib, json\n"
            "def verify_checkpoint_manifest(checkpoint):\n"
            "    manifest = checkpoint / 'fixture-manifest.json'\n"
            "    for name, expected in json.loads(manifest.read_text()).items():\n"
            "        if hashlib.sha256((checkpoint / name).read_bytes()).hexdigest() != expected:\n"
            "            raise ValueError('checkpoint file differs from committed manifest')\n"
            "    return hashlib.sha256(manifest.read_bytes()).hexdigest()\n",
            encoding="utf-8",
        )
        if adapter_source is None:
            adapter_source = (
                "class FixtureAdapter:\n"
                "    horizon = 50\n"
                "    execution_horizon = 50\n"
                "    def predict(self, batch):\n"
                "        import torch\n"
                "        offset = batch['images']['top'].mean() + 2 * batch['images']['wrist'].mean()\n"
                "        actions = batch['state'] + offset\n"
                "        return actions.unsqueeze(1).expand(-1, 50, -1).contiguous()\n"
                "def load(policy_dir, device):\n"
                "    return FixtureAdapter()\n"
            )
        (original_source / "rdvla_adapter.py").write_text(adapter_source, encoding="utf-8")
        (original_source / "so101_data.py").write_text("# fixture data code\n", encoding="utf-8")
        (original_source / "train.py").write_text("# fixture launcher\n", encoding="utf-8")
        (original_source / "RD-VLA/model.py").write_text("# fixture model source\n", encoding="utf-8")
        config = {
            "run_id": "fixture-rdvla",
            "root": str(original_root),
            "gpu_uuid": "GPU-00000000-0000-0000-0000-000000000000",
            "manifest_sha256": "a" * 64,
            "action_horizon": 50,
            "execution_horizon": 50,
            "epochs": 2,
            "wandb_entity": "fixture-entity",
            "wandb_project": "fixture-project",
            "source_sha256": adapters._source_tree_identity(original_source),
            "source_files": adapters._source_tree_records(original_source),
            "models": {
                "rdvla": {"python": str(base / "training-host-python"), "wandb_run_id": "fixture-wandb"}
            },
        }
        config_path = base / "training_config.json"
        config_path.write_text(json.dumps(config, sort_keys=True), encoding="utf-8")
        original_checkpoint = original_root / "runs/rdvla/checkpoints/step-1"
        policy = original_checkpoint / "policy"
        policy.mkdir(parents=True)
        (policy / "model_config.json").write_text(
            json.dumps(
                {
                    "robot_platform": "SO101",
                    "camera_order": list(adapters.CAMERAS),
                    "image_sequence_len": 2,
                    "action_dim": 6,
                    "action_chunk_len": 50,
                    "execution_horizon": 50,
                }
            ),
            encoding="utf-8",
        )
        (policy / "weights.bin").write_bytes(b"synthetic-checkpoint")
        progress = {
            "schema": 1,
            "status": "checkpoint_committed",
            "checkpoint": str(original_checkpoint),
            "identity": {
                "run_id": config["run_id"],
                "model": "rdvla",
                "manifest_sha256": config["manifest_sha256"],
                "config_sha256": adapters._rd_config_identity(config, config["models"]["rdvla"], "rdvla"),
                "source_files": config["source_files"],
                "total_steps": 5,
            },
            "wandb": {
                "id": "fixture-wandb",
                "entity": config["wandb_entity"],
                "project": config["wandb_project"],
            },
            "global_step": 1,
            "total_steps": 5,
            "completed_epochs": 0,
            "training_samples": 1,
            "epoch_offset": 1,
            "frames_per_epoch": 2,
        }
        (original_checkpoint / "progress.json").write_text(json.dumps(progress), encoding="utf-8")
        manifest = {
            path.relative_to(original_checkpoint).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in original_checkpoint.rglob("*")
            if path.is_file()
        }
        (original_checkpoint / "fixture-manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        runtime_root = base / "ubuntu-runtime"
        runtime_source = runtime_root / "source"
        shutil.copytree(original_source, runtime_source)
        checkpoint = runtime_root / "runs/rdvla/checkpoints/step-1"
        checkpoint.parent.mkdir(parents=True)
        shutil.copytree(original_checkpoint, checkpoint)
        return {
            "config": config,
            "config_bytes": config_path.read_bytes(),
            "config_path": config_path,
            "original_checkpoint": original_checkpoint,
            "runtime_root": runtime_root,
            "runtime_source": runtime_source,
            "checkpoint": checkpoint,
            "runtime": {
                "root": str(runtime_root),
                "python": sys.executable,
                "gpu_uuid": "GPU-11111111-1111-1111-1111-111111111111",
            },
            "output": base / "deployment.json",
            "adapter_source": adapter_source.encode("utf-8"),
        }

    def test_rd_runtime_relocation_preserves_identity_and_rejects_tampering(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = self._rd_runtime_fixture(Path(directory))
            with contextlib.redirect_stderr(io.StringIO()):
                spec = adapters.prepare_deployment(
                    "rdvla",
                    fixture["config_path"],
                    fixture["checkpoint"],
                    fixture["output"],
                    runtime=fixture["runtime"],
                )
            self.assertEqual(spec["runtime"], fixture["runtime"])
            self.assertEqual(spec["training_config"], str(fixture["config_path"].resolve()))
            self.assertEqual(spec["checkpoint"], str(fixture["checkpoint"]))
            self.assertEqual(spec["progress"]["checkpoint"], str(fixture["original_checkpoint"]))
            self.assertEqual(adapters.load_deployment(fixture["output"], model="rdvla"), spec)
            self.assertEqual(adapters._verify_full_descriptor(spec)[0], fixture["config"])
            self.assertEqual(fixture["config_path"].read_bytes(), fixture["config_bytes"])

            adapter_file = fixture["runtime_source"] / "rdvla_adapter.py"
            adapter_file.write_text("# tampered source\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                adapters._verify_full_descriptor(spec)
            adapter_file.write_bytes(fixture["adapter_source"])

            weights = fixture["checkpoint"] / "policy/weights.bin"
            weights.write_bytes(b"tampered checkpoint")
            with self.assertRaisesRegex(ValueError, "checkpoint file differs"):
                adapters._verify_full_descriptor(spec)
            weights.write_bytes(b"synthetic-checkpoint")

            tampered = dict(spec)
            tampered["runtime"] = dict(spec["runtime"], root=str(Path(directory) / "other-runtime"))
            tampered["artifact"] = adapters._deployment_identity(tampered)
            with self.assertRaisesRegex(ValueError, "runtime paths or hardware"):
                adapters._verify_full_descriptor(tampered)
            self.assertEqual(fixture["config_path"].read_bytes(), fixture["config_bytes"])

    def test_rd_runtime_cpu_native_worker_round_trip(self):
        try:
            import numpy as np
            import torch
        except ImportError as exc:
            self.skipTest(f"CPU worker dependencies are unavailable: {exc}")
        with tempfile.TemporaryDirectory() as directory:
            fixture = self._rd_runtime_fixture(Path(directory))
            with contextlib.redirect_stderr(io.StringIO()):
                spec = adapters.prepare_deployment(
                    "rdvla",
                    fixture["config_path"],
                    fixture["checkpoint"],
                    fixture["output"],
                    runtime=fixture["runtime"],
                )

            class ExecutionMode(enum.Enum):
                CHUNK = "chunk"
                RTC = "rtc"

            class Record:
                def __init__(self, **values):
                    self.__dict__.update(values)

            inference = types.ModuleType("lerobot.inference")
            inference.ActionChunk = Record
            inference.ActionProvenance = Record
            inference.ExecutionMode = ExecutionMode
            inference.PolicyCapabilities = Record
            lerobot = types.ModuleType("lerobot")
            lerobot.__path__ = []
            lerobot.inference = inference
            features = (
                SimpleNamespace(
                    name="observation.state", shape=(6,), dtype="float32",
                    names=adapters.JOINTS, semantics=adapters.SEMANTICS, kind="tensor",
                ),
                SimpleNamespace(
                    name="observation.images.top", shape=adapters.IMAGE_SHAPE, dtype="uint8",
                    semantics=adapters.SEMANTICS, kind="rgb",
                ),
                SimpleNamespace(
                    name="observation.images.wrist", shape=adapters.IMAGE_SHAPE, dtype="uint8",
                    semantics=adapters.SEMANTICS, kind="rgb",
                ),
            )
            action_feature = SimpleNamespace(
                name="action", shape=(6,), dtype="float32", names=adapters.JOINTS,
                semantics=adapters.SEMANTICS, kind="tensor",
            )
            runner = None
            try:
                with patch.dict(sys.modules, {"lerobot": lerobot, "lerobot.inference": inference}):
                    runner = adapters.AdapterRunner(
                        spec,
                        features=features,
                        action_feature=action_feature,
                        fps=30,
                        execution_horizon=7,
                        device="cpu",
                    )
                state = np.arange(1, 7, dtype=np.float32)
                observation = SimpleNamespace(
                    features={
                        "observation.state": state,
                        "observation.images.top": np.full(adapters.IMAGE_SHAPE, 64, dtype=np.uint8),
                        "observation.images.wrist": np.full(adapters.IMAGE_SHAPE, 128, dtype=np.uint8),
                    },
                    task="fixture task",
                    capture_time=12.5,
                    task_version=3,
                    observation_id="fixture-observation",
                )
                chunk = runner.predict(
                    observation,
                    mode=ExecutionMode.CHUNK,
                    inference_delay=2,
                    model_continuation={"queue": "metadata"},
                    canonical_continuation={"queue": "metadata"},
                )
                actions = chunk.canonical_actions.numpy()
                expected = state + (64 / 255) + 2 * (128 / 255)
                self.assertEqual(actions.shape, (7, 6))
                self.assertEqual(actions.dtype, np.float32)
                self.assertTrue(np.isfinite(actions).all())
                np.testing.assert_allclose(actions, np.broadcast_to(expected, (7, 6)), rtol=0, atol=1e-6)
                self.assertEqual(chunk.execution_steps, 7)
                self.assertEqual(chunk.provenance.capture_time, observation.capture_time)
                self.assertEqual(chunk.provenance.task, observation.task)
                self.assertEqual(chunk.provenance.task_version, observation.task_version)
                self.assertEqual(chunk.provenance.observation_id, observation.observation_id)
                self.assertEqual(chunk.canonical_actions.dtype, torch.float32)
                with self.assertRaisesRegex(ValueError, "CHUNK execution only"):
                    runner.predict(observation, mode=ExecutionMode.RTC)
            finally:
                if runner is not None:
                    runner.close()


    def test_close_interrupts_inflight_native_prediction(self):
        try:
            import numpy as np
            import torch
        except ImportError as exc:
            self.skipTest(f"CPU worker dependencies are unavailable: {exc}")
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory).resolve()
            marker = base / "prediction-started"
            adapter_source = (
                "import time\n"
                "class FixtureAdapter:\n"
                "    horizon = 50\n"
                "    execution_horizon = 50\n"
                "    def predict(self, batch):\n"
                f"        open({str(marker)!r}, 'w', encoding='utf-8').write('started')\n"
                "        while True:\n"
                "            time.sleep(1)\n"
                "def load(policy_dir, device):\n"
                "    return FixtureAdapter()\n"
            )
            fixture = self._rd_runtime_fixture(base, adapter_source=adapter_source)
            with contextlib.redirect_stderr(io.StringIO()):
                spec = adapters.prepare_deployment(
                    "rdvla",
                    fixture["config_path"],
                    fixture["checkpoint"],
                    fixture["output"],
                    runtime=fixture["runtime"],
                )

            class ExecutionMode(enum.Enum):
                CHUNK = "chunk"
                RTC = "rtc"

            class Record:
                def __init__(self, **values):
                    self.__dict__.update(values)

            inference = types.ModuleType("lerobot.inference")
            inference.ActionChunk = Record
            inference.ActionProvenance = Record
            inference.ExecutionMode = ExecutionMode
            inference.PolicyCapabilities = Record
            lerobot = types.ModuleType("lerobot")
            lerobot.__path__ = []
            lerobot.inference = inference
            features = (
                SimpleNamespace(
                    name="observation.state", shape=(6,), dtype="float32",
                    names=adapters.JOINTS, semantics=adapters.SEMANTICS, kind="tensor",
                ),
                SimpleNamespace(
                    name="observation.images.top", shape=adapters.IMAGE_SHAPE, dtype="uint8",
                    semantics=adapters.SEMANTICS, kind="rgb",
                ),
                SimpleNamespace(
                    name="observation.images.wrist", shape=adapters.IMAGE_SHAPE, dtype="uint8",
                    semantics=adapters.SEMANTICS, kind="rgb",
                ),
            )
            action_feature = SimpleNamespace(
                name="action", shape=(6,), dtype="float32", names=adapters.JOINTS,
                semantics=adapters.SEMANTICS, kind="tensor",
            )
            runner = None
            prediction_thread = None
            close_thread = None
            results = []
            prediction_errors = []
            close_errors = []
            try:
                with patch.dict(sys.modules, {"lerobot": lerobot, "lerobot.inference": inference}):
                    runner = adapters.AdapterRunner(
                        spec,
                        features=features,
                        action_feature=action_feature,
                        fps=30,
                        execution_horizon=7,
                        device="cpu",
                    )
                worker = runner._proc
                observation = SimpleNamespace(
                    features={
                        "observation.state": np.arange(1, 7, dtype=np.float32),
                        "observation.images.top": np.full(adapters.IMAGE_SHAPE, 64, dtype=np.uint8),
                        "observation.images.wrist": np.full(adapters.IMAGE_SHAPE, 128, dtype=np.uint8),
                    },
                    task="fixture task",
                    capture_time=12.5,
                    task_version=3,
                    observation_id="blocking-observation",
                )

                def predict():
                    try:
                        results.append(runner.predict(observation, mode=ExecutionMode.CHUNK))
                    except BaseException as exc:
                        prediction_errors.append(exc)

                def close():
                    try:
                        runner.close()
                    except BaseException as exc:
                        close_errors.append(exc)

                prediction_thread = threading.Thread(target=predict, name="adapter-predict-regression")
                prediction_thread.start()
                deadline = time.monotonic() + 10
                while not marker.exists() and time.monotonic() < deadline and worker.poll() is None:
                    time.sleep(0.01)
                self.assertTrue(marker.exists(), "synthetic adapter prediction did not reach its blocking point")

                close_thread = threading.Thread(target=close, name="adapter-close-regression", daemon=True)
                started = time.monotonic()
                close_thread.start()
                close_thread.join(timeout=2)
                close_duration = time.monotonic() - started
                if close_thread.is_alive():
                    runner._terminate_worker()
                close_thread.join(timeout=10)
                prediction_thread.join(timeout=10)

                self.assertLess(close_duration, 2, "close waited for the in-flight prediction lock")
                self.assertFalse(close_thread.is_alive(), "close thread did not exit")
                self.assertFalse(prediction_thread.is_alive(), "prediction thread did not exit after worker termination")
                self.assertEqual(close_errors, [])
                self.assertIsNotNone(worker.poll(), "native worker survived close")
                self.assertEqual(results, [], "close returned a stale prediction result")
                self.assertEqual(len(prediction_errors), 1)
                with self.assertRaisesRegex(RuntimeError, "closed or has exited"):
                    runner.predict(observation, mode=ExecutionMode.CHUNK)
                self.assertIs(runner._proc, worker, "closed runner unexpectedly accepted/restarted work")
                runner.close()
            finally:
                if runner is not None:
                    runner._terminate_worker()
                if prediction_thread is not None:
                    prediction_thread.join(timeout=10)
                if close_thread is not None:
                    close_thread.join(timeout=10)
                if runner is not None:
                    runner.close()


class AdapterPipeTests(unittest.TestCase):
    def test_reply_eof_and_timeout_fail_closed(self):
        read_fd, write_fd = os.pipe()
        try:
            os.close(write_fd)
            with self.assertRaises(EOFError):
                adapters._receive_reply(read_fd, 0.1)
        finally:
            os.close(read_fd)
        read_fd, write_fd = os.pipe()
        try:
            with self.assertRaises(TimeoutError):
                adapters._receive_reply(read_fd, 0.01)
        finally:
            os.close(read_fd)
            os.close(write_fd)


if __name__ == "__main__":
    unittest.main()
