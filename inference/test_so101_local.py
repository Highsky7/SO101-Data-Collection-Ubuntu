"""Hardware-free local playback regressions; no real weights, GPU or devices."""
import argparse
import json
import os
from pathlib import Path
from queue import Queue
import subprocess
import tempfile
from types import SimpleNamespace
from threading import Event, Lock
import unittest
from unittest.mock import patch

import numpy as np
import torch

import so101_local as local
import so101_remote as shared


class Clock:
    def __init__(self):
        self.now = 0.0
        self.on_sleep = None

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        if self.on_sleep is not None:
            self.on_sleep()
        self.now += seconds


class ControlledRunner:
    """Synthetic backend gated by events; production always uses AdapterRunner."""

    def __init__(self, steps):
        self.steps = steps
        self.calls = Queue()
        self.lock = Lock()
        self.active = None
        self.closed = False
        self.count = 0

    def predict(self, observation, **kwargs):
        from lerobot.inference import ActionChunk, ActionProvenance

        with self.lock:
            if self.closed:
                raise EOFError("synthetic worker closed")
            self.count += 1
            call = SimpleNamespace(
                observation=observation, release=Event(), index=self.count,
                actions=None, error=None,
            )
            self.active = call
        self.calls.put(call)
        if not call.release.wait(5):
            raise TimeoutError("test did not release the synthetic prediction")
        if self.closed:
            raise EOFError("synthetic worker closed")
        if call.error is not None:
            raise call.error
        actions = call.actions
        if actions is None:
            state = torch.tensor(observation.features["observation.state"].copy())
            actions = torch.stack([state + 10 * call.index + step for step in range(self.steps)])
        return ActionChunk(
            model_actions=None, canonical_actions=actions, execution_steps=self.steps,
            provenance=ActionProvenance(
                capture_time=observation.capture_time, task=observation.task,
                task_version=observation.task_version, observation_id=observation.observation_id,
            ),
        )

    def close(self):
        with self.lock:
            self.closed = True
            if self.active is not None:
                self.active.release.set()


class LocalPlaybackTest(unittest.TestCase):
    def setUp(self):
        self.clock = Clock()
        self.args = SimpleNamespace(
            task="pick up the object", fps=10, duration=0.59, execution_horizon=4,
            max_observation_age=5, refill_seconds=0.2, startup_timeout=1, action_timeout=1,
        )
        self.sent = []
        self.captures = 0
        self.image = np.zeros((480, 640, 3), dtype=np.uint8)
        self.robot = SimpleNamespace(get_observation=self.observe, send_action=self.sent.append)

    def observe(self):
        self.captures += 1
        base = 10 * (self.captures - 1)
        return {**{name: base + index for index, name in reversed(list(enumerate(shared.JOINTS)))},
                "top": self.image, "wrist": self.image}

    def make_engine(self):
        runner = ControlledRunner(self.args.execution_horizon)
        engine = local.LocalChunkEngine(runner, self.args, clock=self.clock.monotonic)
        record = SimpleNamespace(engine=engine, runner=runner, accepted=Queue(), faulted=Event())
        accept, fault = engine.runtime.accept, engine.runtime.fault

        def accepted(*args, **kwargs):
            result = accept(*args, **kwargs)
            record.accepted.put(result)
            return result

        def faulted(reason):
            fault(reason)
            record.faulted.set()

        engine.runtime.accept = accepted
        engine.runtime.fault = faulted
        self.addCleanup(engine.close)
        return record

    def submit(self, record):
        observation = local.snapshot(self.observe(), self.args.task, self.clock.now)
        record.engine.submit(observation)
        return record.runner.calls.get(timeout=5)

    def test_control_keeps_dispatching_while_refill_prediction_is_blocked(self):
        record = self.make_engine()
        calls = []

        def advance():
            if not calls:
                call = record.runner.calls.get(timeout=5)
                calls.append(call)
                call.release.set()
                self.assertTrue(record.accepted.get(timeout=5))
            elif len(self.sent) == 3:
                call = record.runner.calls.get(timeout=5)
                calls.append(call)  # Hold refill while the next motor tick runs.
            elif len(self.sent) == 4:
                self.assertFalse(calls[1].release.is_set())
                calls[1].release.set()
                self.assertTrue(record.accepted.get(timeout=5))

        self.clock.on_sleep = advance
        with patch.object(local, "time", self.clock):
            local.control_loop(self.robot, record.engine, self.args)
        self.assertEqual([row[shared.JOINTS[0]] for row in self.sent], [10, 11, 12, 13, 30])
        self.assertEqual(self.sent[0][shared.JOINTS[-1]], 15)
        self.assertEqual(tuple(self.sent[0]), shared.JOINTS)
        np.testing.assert_array_equal(calls[0].observation.features["observation.state"], np.arange(6))
        np.testing.assert_array_equal(calls[1].observation.features["observation.state"], np.arange(6) + 10)

    def test_duration_does_not_wait_for_prediction_or_accept_late_results(self):
        record = self.make_engine()
        calls = []

        def expire():
            if not calls:
                calls.append(record.runner.calls.get(timeout=5))
                self.clock.now = 2

        self.clock.on_sleep = expire
        with patch.object(local, "time", self.clock):
            local.control_loop(self.robot, record.engine, self.args)
        self.assertEqual(self.sent, [])
        self.assertFalse(calls[0].release.is_set())
        calls[0].release.set()
        record.engine.close()
        self.assertTrue(record.engine.runtime.queue.empty())
        self.assertTrue(record.accepted.empty())

    def test_exhausted_queue_faults_instead_of_replaying_old_actions(self):
        record = self.make_engine()
        record.engine.start()
        first = self.submit(record)
        first.release.set()
        self.assertTrue(record.accepted.get(timeout=5))
        old_source = record.engine.runtime.pop()[1].request_id
        record.engine.runtime.pop()
        self.assertTrue(record.engine.needs_observation())
        second = self.submit(record)
        self.assertFalse(record.engine.needs_observation())
        self.assertEqual(record.engine.runtime.pop()[1].request_id, old_source)
        record.engine.runtime.pop()
        self.assertIsNone(record.engine.runtime.pop())
        with self.assertRaisesRegex(RuntimeError, "buffer exhausted"):
            record.engine.raise_if_failed()
        record.engine.stop()
        second.release.set()
        record.engine.close()
        self.assertIsNone(record.engine.runtime.pop())

    def test_stale_queued_source_is_rejected_before_dispatch(self):
        self.args.startup_timeout = 10
        record = self.make_engine()
        record.engine.start()
        call = self.submit(record)
        call.release.set()
        self.assertTrue(record.accepted.get(timeout=5))
        self.clock.now = 6
        self.assertIsNone(record.engine.runtime.pop())
        with self.assertRaisesRegex(RuntimeError, "observation is too old"):
            record.engine.raise_if_failed()

    def test_invalid_or_failed_predictions_revoke_motion(self):
        for actions, error in (
            (torch.full((4, 6), float("nan")), None),
            (torch.zeros((1, 6)), None),
            (torch.zeros((4, 6), dtype=torch.float64), None),
            (None, OSError("native worker failed")),
        ):
            with self.subTest(error=error, shape=None if actions is None else tuple(actions.shape)):
                record = self.make_engine()
                record.engine.start()
                call = self.submit(record)
                call.actions, call.error = actions, error
                call.release.set()
                self.assertTrue(record.faulted.wait(5))
                with self.assertRaises(RuntimeError):
                    record.engine.raise_if_failed()
                self.assertIsNone(record.engine.runtime.pop())
                record.engine.close()

    def test_startup_and_request_deadlines_apply_while_predictor_is_blocked(self):
        for name in ("startup_timeout", "action_timeout"):
            with self.subTest(deadline=name):
                self.clock.now = 0
                self.args.startup_timeout = self.args.action_timeout = 1
                setattr(self.args, name, 0.2)
                record = self.make_engine()
                record.engine.start()
                call = self.submit(record)
                self.clock.now = 0.3
                with self.assertRaisesRegex(RuntimeError, "deadline exceeded"):
                    record.engine.raise_if_failed()
                self.assertIsNone(record.engine.runtime.pop())
                self.assertFalse(call.release.is_set())
                record.engine.close()

    def test_interrupt_disconnects_hardware_before_closing_pending_model(self):
        order = []
        runner = ControlledRunner(self.args.execution_horizon)
        close = runner.close

        def close_model():
            order.append("model")
            close()

        runner.close = close_model
        robot = SimpleNamespace(
            connect=lambda **kwargs: None, is_calibrated=True,
            cameras={"top": SimpleNamespace(is_connected=True, disconnect=lambda: order.append("camera"))},
            bus=SimpleNamespace(is_connected=True, disconnect=lambda disable: order.append("bus")),
        )

        def interrupt(robot, engine, args):
            engine.start()
            engine.submit(local.snapshot(self.observe(), args.task, engine.runtime.clock()))
            runner.calls.get(timeout=5)
            raise KeyboardInterrupt

        with tempfile.TemporaryDirectory() as directory:
            port = Path(directory) / "serial"
            port.touch()
            with patch("lerobot.robots.so_follower.SO101Follower", return_value=robot), \
                    patch.object(local, "control_loop", side_effect=interrupt):
                with self.assertRaises(KeyboardInterrupt):
                    local.run_robot(SimpleNamespace(port=str(port)), runner, self.args)
        self.assertEqual(order, ["bus", "camera", "model"])

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
