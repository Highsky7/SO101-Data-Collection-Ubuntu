"""Hardware-free local playback regressions; no real weights, GPU or devices."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

import so101_local as local
import so101_remote as shared


class Clock:
    def __init__(self):
        self.now = 0.0

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


class LocalPlaybackTest(unittest.TestCase):
    def setUp(self):
        self.clock = Clock()
        self.args = SimpleNamespace(task="pick up the object", fps=10, duration=0.39,
                                    execution_horizon=2, max_observation_age=5)
        self.observations = []
        self.sent = []
        self.image = np.zeros((480, 640, 3), dtype=np.uint8)
        self.robot = SimpleNamespace(get_observation=self.observe, send_action=self.sent.append)
        self.runner = SimpleNamespace(predict=self.predict)

    def observe(self):
        # Deliberately reverse dictionary order; the trained joint order must prevail.
        return {**{name: index + 0.5 for index, name in reversed(list(enumerate(shared.JOINTS)))},
                "top": self.image, "wrist": self.image}

    def predict(self, observation):
        self.observations.append(observation)
        values = torch.arange(12, dtype=torch.float32).reshape(2, 6) + 10 * len(self.observations)
        return SimpleNamespace(canonical_actions=values)

    def test_local_chunks_use_fresh_observations_and_recorded_joint_order(self):
        with patch.object(local, "time", self.clock):
            local.control_loop(self.robot, self.runner, self.args)
        self.assertEqual(len(self.observations), 2)
        np.testing.assert_array_equal(self.observations[0].features["observation.state"], np.arange(6) + 0.5)
        self.assertEqual(self.observations[1].task, self.args.task)
        self.assertEqual([row[shared.JOINTS[0]] for row in self.sent], [10, 16, 20, 26])
        self.assertEqual(self.sent[0][shared.JOINTS[-1]], 15)
        self.assertEqual(tuple(self.sent[0]), shared.JOINTS)

    def test_slow_inference_cannot_send_actions_after_duration(self):
        def slow(observation):
            self.clock.now += 1
            return self.predict(observation)

        with patch.object(local, "time", self.clock):
            local.control_loop(self.robot, SimpleNamespace(predict=slow), self.args)
        self.assertEqual(self.sent, [])

    def test_stale_or_invalid_predictions_never_reach_motors(self):
        self.args.duration = 10
        self.args.max_observation_age = 0.5

        def stale(observation):
            self.clock.now += 0.6
            return self.predict(observation)

        for predict, error in (
            (stale, TimeoutError),
            (lambda obs: SimpleNamespace(canonical_actions=torch.full((2, 6), float("nan"))), ValueError),
            (lambda obs: SimpleNamespace(canonical_actions=torch.zeros((1, 6))), ValueError),
            (lambda obs: SimpleNamespace(canonical_actions=torch.zeros((2, 6), dtype=torch.float64)), ValueError),
        ):
            with self.subTest(error=error), patch.object(local, "time", self.clock):
                self.clock.now = 0
                with self.assertRaises(error):
                    local.control_loop(self.robot, SimpleNamespace(predict=predict), self.args)
                self.assertEqual(self.sent, [])

    def test_partial_connect_teardown_still_disables_torque_when_camera_close_fails(self):
        closed = []

        def bad_camera():
            closed.append("camera")
            raise OSError("camera close failed")

        robot = SimpleNamespace(
            cameras={"top": SimpleNamespace(is_connected=True, disconnect=bad_camera),
                     "wrist": SimpleNamespace(is_connected=False)},
            bus=SimpleNamespace(is_connected=True, disconnect=lambda disable: closed.append(("bus", disable))),
        )
        with self.assertRaisesRegex(OSError, "camera close failed"):
            local.close_robot(robot)
        self.assertEqual(closed, [("bus", True), "camera"])


class LocalConfigurationTest(unittest.TestCase):
    def test_native_preflight_uses_ubuntu_calibration_rgb_and_degree_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "follower.json").write_text("{}")
            cameras = root / "cameras.json"
            cameras.write_text(json.dumps({name: {
                "type": "opencv", "index_or_path": index, "width": 640, "height": 480,
                "fps": 30, "color_mode": "rgb",
            } for index, name in enumerate(("top", "wrist"))}))
            args = SimpleNamespace(robot_port="/dev/not-opened", robot_id="follower", task="pick",
                                   calibration_dir=root, cameras=cameras, max_relative_target=5.0)
            config = local.robot_config(args)
            self.assertEqual(config.port, args.robot_port)
            self.assertTrue((config.calibration_dir / "follower.json").samefile(root / "follower.json"))
            self.assertTrue(config.use_degrees)
            self.assertTrue(config.disable_torque_on_disconnect)
            self.assertEqual(config.max_relative_target, 5.0)
            self.assertEqual(tuple(config.cameras), ("top", "wrist"))
            self.assertEqual(config.cameras["wrist"].index_or_path, 1)

    def test_run_without_explicit_motion_confirmation_stops_before_loading(self):
        with self.assertRaises(SystemExit) as result:
            local.main(["run", "--deployment-file", "/not-read", "--robot-port", "/not-opened",
                        "--robot-id", "follower", "--cameras", "/not-read", "--task", "pick", "--duration", "1"])
        self.assertEqual(result.exception.code, 2)

    def test_relocated_gpu_uses_uuid_and_never_training_host_mps_pid_approvals(self):
        gpu = "GPU-local-ubuntu"
        pid = os.getpid() + 1
        healthy = subprocess.CompletedProcess([], 0, f"2, {gpu}, None\n", "")
        busy = subprocess.CompletedProcess([], 0, f"{gpu}, {pid}, nvidia-cuda-mps-server\n", "")
        empty = subprocess.CompletedProcess([], 0, "", "")
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / "config.json"
            config.write_text(json.dumps({"gpu_uuid": shared.GPU0, "idle_mps_pids": [pid]}))
            spec = {"training_config": str(config), "training_config_sha256": shared.hashlib.sha256(config.read_bytes()).hexdigest(),
                    "runtime": {"gpu_uuid": gpu}}
            with patch.dict(os.environ, {}, clear=True):
                with patch.object(shared.subprocess, "run", side_effect=[healthy, busy]):
                    with self.assertRaisesRegex(RuntimeError, "allow-shared-gpu"):
                        shared.require_gpu0(gpu, deployment=spec)
                self.assertNotIn("CUDA_VISIBLE_DEVICES", os.environ)
                with patch.object(shared.subprocess, "run", side_effect=[healthy, empty]):
                    shared.require_gpu0(gpu, deployment=spec)
                self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], gpu)


if __name__ == "__main__":
    unittest.main()
