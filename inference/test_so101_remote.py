"""Hardware-free input-boundary regressions; never instantiate a robot or camera."""
import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import tempfile
import subprocess
import unittest
from unittest.mock import patch

from so101_remote import (
    GPU0, duration_seconds, execution_horizon_steps, positive_fps, positive_seconds,
    read_cameras, require_gpu0, verify_policy,
)


class ObservationContractTest(unittest.TestCase):
    def test_incompatible_camera_streams_are_rejected(self):
        valid = {
            "top": {"type": "opencv", "index_or_path": 3, "width": 640, "height": 480, "fps": 30},
            "wrist": {"type": "opencv", "index_or_path": 4, "width": 640, "height": 480, "fps": 30},
        }
        invalid = [None, {"top": valid["top"]}]
        for name, key, value in (
            ("top", "type", "intelrealsense"),
            ("top", "type", "zmq"),
            ("top", "index_or_path", ""),
            ("top", "color_mode", "bgr"),
            ("top", "width", 320),
            ("wrist", "index_or_path", ""),
            ("wrist", "index_or_path", 3),
        ):
            cameras = deepcopy(valid)
            cameras[name][key] = value
            invalid.append(cameras)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cameras.json"
            path.write_text(json.dumps(valid))
            read_cameras(path)
            for cameras in invalid:
                with self.subTest(cameras=cameras):
                    path.write_text(json.dumps(cameras))
                    with self.assertRaises(ValueError):
                        read_cameras(path)

    def test_explicit_unlimited_duration_and_invalid_limits(self):
        self.assertEqual(duration_seconds("0"), 0.0)
        self.assertEqual(duration_seconds("2.5"), 2.5)
        for value in ("-1", "nan", "inf"):
            with self.subTest(value=value), self.assertRaises(argparse.ArgumentTypeError):
                duration_seconds(value)

    def test_refill_requires_positive_finite_seconds(self):
        for value in ("0", "-1", "nan", "inf"):
            with self.subTest(value=value), self.assertRaises(argparse.ArgumentTypeError):
                positive_seconds(value)

    def test_action_rate_requires_positive_finite_hz(self):
        self.assertEqual(positive_fps("15"), 15.0)
        for value in ("0", "-1", "nan", "inf"):
            with self.subTest(value=value), self.assertRaises(argparse.ArgumentTypeError):
                positive_fps(value)

    def test_execution_slice_stays_within_trained_prediction(self):
        self.assertEqual(execution_horizon_steps("1"), 1)
        self.assertEqual(execution_horizon_steps("50"), 50)
        for value in ("0", "-1", "51", "2.5"):
            with self.subTest(value=value), self.assertRaises(argparse.ArgumentTypeError):
                execution_horizon_steps(value)


class CheckpointEpochSelectionTest(unittest.TestCase):
    def test_other_epoch_and_modified_weights_are_rejected(self):
        # Tiny integrity fixtures only; no model, GPU, robot, or camera is opened.
        with tempfile.TemporaryDirectory() as directory:
            roots, hashes = {}, {}
            for epoch in (10, 20):
                roots[epoch] = Path(directory) / f"epoch-{epoch}"
                policy = roots[epoch] / "policy"
                policy.mkdir(parents=True)
                (policy / "adapter.json").write_text('{"non_training_fixture": true}')
                (policy / "model.safetensors").write_bytes(f"integrity-fixture-{epoch}".encode())
                hashes[epoch] = {
                    name: hashlib.sha256((policy / name).read_bytes()).hexdigest()
                    for name in ("adapter.json", "model.safetensors")
                }
            with patch("so101_remote.POLICY_HASHES", hashes):
                verify_policy(roots[20], epoch=20)
                for requested, supplied in ((10, 20), (20, 10)):
                    with self.subTest(requested=requested, supplied=supplied), self.assertRaises(ValueError):
                        verify_policy(roots[supplied], epoch=requested)
                (roots[20] / "policy/model.safetensors").write_bytes(b"modified")
                with self.assertRaises(ValueError):
                    verify_policy(roots[20], epoch=20)


class GPUAdmissionTest(unittest.TestCase):
    def test_busy_or_unhealthy_gpu_requires_explicit_admission(self):
        healthy = subprocess.CompletedProcess([], 0, f"0, {GPU0}, None\n", "")
        busy = subprocess.CompletedProcess([], 0, f"{GPU0}, {os.getpid() + 1}, python\n", "")
        with patch.dict(os.environ, {}, clear=True):
            with patch("so101_remote.subprocess.run", side_effect=[healthy, busy]):
                with self.assertRaisesRegex(RuntimeError, "allow-shared-gpu"):
                    require_gpu0()
                self.assertNotIn("CUDA_VISIBLE_DEVICES", os.environ)
            available = subprocess.CompletedProcess([], 0, "81920, 65536\n", "")
            with patch("so101_remote.subprocess.run", side_effect=[healthy, busy, available]):
                require_gpu0(allow_shared=True, memory_limit_gib=1)
                self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], GPU0)
            os.environ["CUDA_VISIBLE_DEVICES"] = "GPU-another-allocation"
            with self.assertRaisesRegex(RuntimeError, "not overridden"):
                require_gpu0(allow_shared=True, memory_limit_gib=1)
            self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], "GPU-another-allocation")
        unhealthy = subprocess.CompletedProcess([], 0, f"0, {GPU0}, Reset\n", "")
        with patch.dict(os.environ, {}, clear=True):
            with patch("so101_remote.subprocess.run", return_value=unhealthy):
                with self.assertRaisesRegex(RuntimeError, "not ready"):
                    require_gpu0(allow_shared=True, memory_limit_gib=1)
                self.assertNotIn("CUDA_VISIBLE_DEVICES", os.environ)

    def test_only_registered_named_mps_servers_are_ignored(self):
        healthy = subprocess.CompletedProcess([], 0, f"0, {GPU0}, None\n", "")
        mps1, mps2, peer = range(os.getpid() + 1, os.getpid() + 4)
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / "config.json"
            config.write_text(json.dumps({"gpu_uuid": GPU0, "idle_mps_pids": [mps1, mps2]}))
            deployment = {
                "training_config": str(config),
                "training_config_sha256": hashlib.sha256(config.read_bytes()).hexdigest(),
            }
            rows = f"{GPU0}, {mps1}, /usr/bin/nvidia-cuda-mps-server\n{GPU0}, {mps2}, nvidia-cuda-mps-server\n"
            inherited = {"CUDA_MPS_PIPE_DIRECTORY": "/old/mps", "CUDA_MPS_LOG_DIRECTORY": "/old/log", "CUDA_DEVICE_ORDER": "FASTEST_FIRST"}
            with patch.dict(os.environ, inherited, clear=True):
                with patch("so101_remote.subprocess.run", side_effect=[healthy, subprocess.CompletedProcess([], 0, rows, "")]):
                    require_gpu0(deployment=deployment)
                self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], GPU0)
                self.assertEqual(os.environ["CUDA_DEVICE_ORDER"], "PCI_BUS_ID")
                self.assertNotIn("CUDA_MPS_PIPE_DIRECTORY", os.environ)
                self.assertNotIn("CUDA_MPS_LOG_DIRECTORY", os.environ)
            for process_rows in (
                rows + f"{GPU0}, {peer}, python\n",
                rows + f"{GPU0}, {peer}, nvidia-cuda-mps-server\n",
                rows.replace("/usr/bin/nvidia-cuda-mps-server", "python", 1),
            ):
                with patch.dict(os.environ, {}, clear=True):
                    processes = subprocess.CompletedProcess([], 0, process_rows, "")
                    with patch("so101_remote.subprocess.run", side_effect=[healthy, processes]):
                        with self.assertRaisesRegex(RuntimeError, "allow-shared-gpu"):
                            require_gpu0(deployment=deployment)
                        self.assertNotIn("CUDA_VISIBLE_DEVICES", os.environ)
            config.write_text(json.dumps({"gpu_uuid": GPU0, "idle_mps_pids": [mps1, mps2, peer]}))
            with self.assertRaisesRegex(ValueError, "config changed"):
                require_gpu0(deployment=deployment)

    def test_gpu_sharing_requires_budget_and_free_reserve(self):
        with self.assertRaisesRegex(ValueError, "explicit --memory-limit-gib"):
            require_gpu0(allow_shared=True)
        for value in (0, -1, float("nan"), float("inf")):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "finite and positive"):
                require_gpu0(memory_limit_gib=value)
        healthy = subprocess.CompletedProcess([], 0, f"0, {GPU0}, None\n", "")
        empty = subprocess.CompletedProcess([], 0, "", "")
        for free_mib, passes in ((16383, False), (16384, True)):
            memory = subprocess.CompletedProcess([], 0, f"81920, {free_mib}\n", "")
            with patch.dict(os.environ, {}, clear=True):
                with patch("so101_remote.subprocess.run", side_effect=[healthy, empty, memory]):
                    if passes:
                        require_gpu0(allow_shared=True, memory_limit_gib=12)
                        self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], GPU0)
                    else:
                        with self.assertRaisesRegex(RuntimeError, "4 GiB free"):
                            require_gpu0(allow_shared=True, memory_limit_gib=12)
                        self.assertNotIn("CUDA_VISIBLE_DEVICES", os.environ)


if __name__ == "__main__":
    unittest.main()
