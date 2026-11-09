from __future__ import annotations

import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

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
