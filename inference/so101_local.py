"""Asynchronous RD-VLA CHUNK inference on this Ubuntu host; no server or SSH.

prepare/preflight never open hardware or initialize CUDA. smoke loads real local
weights with synthetic observations. Only run connects cameras and moves motors.
"""
from __future__ import annotations

import argparse
from contextlib import closing, ExitStack
import json
import logging
import os
from pathlib import Path
import signal
import sys
import time
from threading import Condition, Event, Thread

import so101_adapters as adapters
import so101_remote as shared


def robot_config(args):
    """Parse native camera/robot dataclasses without constructing hardware."""
    if not args.robot_port.strip() or not args.task.strip() or len(args.task) > adapters.MAX_TASK_CHARS:
        raise ValueError("robot-port and task must be nonempty; task is limited to 4096 characters")
    if not args.robot_id or Path(args.robot_id).name != args.robot_id or args.robot_id in {".", ".."}:
        raise ValueError("robot-id must be a calibration filename stem")
    calibration = args.calibration_dir.expanduser().resolve()
    if not (calibration / f"{args.robot_id}.json").is_file():
        raise FileNotFoundError(f"Existing follower calibration required: {calibration / (args.robot_id + '.json')}")
    cameras = shared.read_cameras(args.cameras)
    shared.require_native_version()
    import draccus
    from lerobot.cameras.opencv import OpenCVCameraConfig  # noqa: F401 — register camera type
    from lerobot.robots.so_follower import SO101FollowerConfig

    return draccus.parse(SO101FollowerConfig, args=[
        f"--port={args.robot_port}", f"--id={args.robot_id}",
        f"--calibration_dir={calibration}", f"--cameras={json.dumps(cameras)}",
        "--use_degrees=true", "--disable_torque_on_disconnect=true",
        f"--max_relative_target={args.max_relative_target}",
    ])


def load_runner(spec, args):
    """Reuse the reviewed native worker and its original preprocessing/statistics."""
    shared.require_native_version()
    adapters._verify_full_descriptor(spec)
    if args.device == "cuda:0":
        shared.require_gpu0(
            spec["gpu_uuid"], deployment=spec, allow_shared=args.allow_shared_gpu,
            memory_limit_gib=args.memory_limit_gib,
        )
    elif args.allow_shared_gpu or args.memory_limit_gib is not None:
        raise ValueError("GPU sharing/memory options cannot be used with --device cpu")
    features, action = shared.feature_contract()
    return adapters.AdapterRunner(
        spec, features=features, action_feature=action, fps=args.fps,
        execution_horizon=args.execution_horizon, device=args.device,
        memory_limit_gib=args.memory_limit_gib,
    )


def snapshot(observation, task, capture_time):
    import numpy as np
    from lerobot.inference import ObservationSnapshot

    return ObservationSnapshot(
        features={
            "observation.state": np.asarray([observation[name] for name in shared.JOINTS], dtype=np.float32),
            **{f"observation.images.{name}": observation[name] for name in adapters.CAMERAS},
        },
        capture_time=capture_time, task=task,
    )


class LocalChunkEngine:
    """One local predictor, bounded handoff, and LeRobot's shared CHUNK scheduler."""

    def __init__(self, runner, args, *, clock=None):
        from lerobot.inference import ChunkRuntime, ExecutionMode

        self.runner = runner
        self.execution_horizon = args.execution_horizon
        self.runtime = ChunkRuntime(
            mode=ExecutionMode.CHUNK, action_interval=1.0 / args.fps,
            refill_seconds=args.refill_seconds, max_observation_age_s=args.max_observation_age,
            startup_timeout_s=args.startup_timeout, action_timeout_s=args.action_timeout,
            clock=clock or time.monotonic,
        )
        self._condition = Condition()
        self._stop = Event()
        self._observation = None
        self._thread = None
        self._error = None

    def start(self):
        if self._thread is not None or self._stop.is_set():
            raise RuntimeError("Local inference engine cannot be restarted")
        self.runtime.activate()
        self._thread = Thread(target=self._loop, name="LocalRDVLAInference", daemon=True)
        self._thread.start()

    def needs_observation(self):
        with self._condition:
            if self._stop.is_set() or self._observation is not None:
                return False
        return self.runtime.should_request()

    def submit(self, observation):
        with self._condition:
            if not self._stop.is_set():
                self._observation = observation
                self._condition.notify()

    def _loop(self):
        import torch

        try:
            while not self._stop.is_set():
                with self._condition:
                    self._condition.wait_for(lambda: self._stop.is_set() or self._observation is not None)
                    if self._stop.is_set():
                        return
                    observation, self._observation = self._observation, None
                request = self.runtime.begin(observation)
                if request is None:
                    continue
                chunk = self.runner.predict(
                    observation, mode=request.mode, inference_delay=request.delay,
                    model_continuation=request.continuation.model_actions,
                    canonical_continuation=request.continuation.canonical_actions,
                )
                if self._stop.is_set():
                    return
                actions = chunk.canonical_actions
                if (
                    actions.shape != (self.execution_horizon, len(shared.JOINTS))
                    or chunk.execution_steps != self.execution_horizon
                    or actions.dtype != torch.float32 or actions.device.type != "cpu"
                    or not bool(torch.isfinite(actions).all())
                ):
                    raise ValueError("RD-VLA must return finite CPU float32 [execution_horizon,6] actions")
                accepted = self.runtime.accept(request, chunk, task_version=observation.task_version)
                logging.info(
                    "Local RD-VLA prediction: %.3fs accepted=%s queued=%.3fs refill=%.3fs",
                    self.runtime.clock() - request.submitted_at, accepted,
                    self.runtime.queue.qsize() * self.runtime.interval, self.runtime.effective_refill,
                )
        except Exception as exc:
            if not self._stop.is_set():
                self._error = exc
                self.runtime.fault(f"{type(exc).__name__}: {exc}")

    def raise_if_failed(self):
        self.runtime.check_deadlines()
        if self.runtime.failure is not None:
            raise RuntimeError(f"Local RD-VLA inference stopped: {self.runtime.failure}") from self._error

    def stop(self):
        with self._condition:
            if self._stop.is_set():
                return
            self._stop.set()
            self._observation = None
            self._condition.notify_all()
        self.runtime.deactivate()

    def close(self):
        self.stop()
        try:
            # AdapterRunner.close interrupts a pending native pipe call.
            self.runner.close()
        finally:
            if self._thread is not None:
                self._thread.join(timeout=adapters.CLOSE_TIMEOUT)
                if self._thread.is_alive():
                    raise RuntimeError("Local inference thread did not stop after native worker shutdown")


def control_loop(robot, engine, args):
    """Capture on refill demand; dispatch queued actions independently of prediction."""
    deadline = time.monotonic() + args.duration
    interval = 1.0 / args.fps
    engine.start()
    try:
        while time.monotonic() < deadline:
            tick = time.monotonic()
            engine.raise_if_failed()
            if engine.needs_observation():
                captured = time.monotonic()
                observation = snapshot(robot.get_observation(), args.task, captured)
                if time.monotonic() >= deadline:
                    break
                engine.submit(observation)
            item = engine.runtime.pop()
            engine.raise_if_failed()
            if item is not None:
                action, _ = item
                with engine.runtime.lock:
                    engine.raise_if_failed()
                    if time.monotonic() < deadline and engine.runtime.dispatch_allowed():
                        robot.send_action(dict(zip(shared.JOINTS, map(float, action.numpy()))))
            # No catch-up bursts after slow camera/serial I/O.
            time.sleep(min(max(0.0, interval - (time.monotonic() - tick)), max(0.0, deadline - time.monotonic())))
    finally:
        engine.stop()


def close_robot(robot):
    # Native robot.disconnect() requires ALL devices connected; also clean partial connects.
    with ExitStack() as stack:
        for camera in robot.cameras.values():
            if camera.is_connected:
                stack.callback(camera.disconnect)
        if robot.bus.is_connected:
            stack.callback(robot.bus.disconnect, True)


def run_robot(config, runner, args):
    from lerobot.robots.so_follower import SO101Follower

    if not Path(config.port).exists():
        raise FileNotFoundError(f"Follower serial device does not exist: {config.port}")
    robot = SO101Follower(config)
    engine = LocalChunkEngine(runner, args)
    try:
        robot.connect(calibrate=False)
        if not robot.is_calibrated:
            raise RuntimeError("Motor calibration differs from the existing file; calibrate separately before inference")
        control_loop(robot, engine, args)
    finally:
        engine.stop()
        try:
            close_robot(robot)
        finally:
            engine.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare", help="verify one committed checkpoint and pin its local runtime; no GPU/hardware")
    prepare.add_argument("--training-config", type=Path, required=True)
    prepare.add_argument("--checkpoint", type=Path, required=True)
    prepare.add_argument("--runtime-config", type=Path, help="local execution JSON: exactly root, python, gpu_uuid")
    prepare.add_argument("--output", type=Path, required=True)
    for command in ("preflight", "smoke", "run"):
        sub = commands.add_parser(command, help="MOVES THE ROBOT" if command == "run" else "no hardware access")
        sub.add_argument("--deployment-file", type=Path, required=True)
        sub.add_argument("--fps", type=shared.positive_fps, default=30.0)
        sub.add_argument("--execution-horizon", type=shared.execution_horizon_steps, default=50)
        if command != "preflight":
            sub.add_argument("--device", choices=("cpu", "cuda:0"), default="cuda:0", help="cpu is smoke-only; no fallback")
            sub.add_argument("--allow-shared-gpu", action="store_true")
            sub.add_argument("--memory-limit-gib", type=float)
        if command != "smoke":
            sub.add_argument("--robot-port", required=True)
            sub.add_argument("--robot-id", required=True)
            sub.add_argument("--cameras", type=Path, required=True)
            calibration_root = Path(os.environ.get("HF_LEROBOT_CALIBRATION", shared.PROJECT / "ubuntu" / "calibration"))
            sub.add_argument("--calibration-dir", type=Path, default=calibration_root / "robots" / "so_follower")
            sub.add_argument("--task", required=True)
            sub.add_argument("--duration", type=shared.positive_seconds, required=True, help="finite positive seconds including inference time")
            sub.add_argument("--max-relative-target", type=shared.positive_seconds, default=5.0, help="native per-command limit: body degrees, gripper 0–100 units; not collision avoidance")
            sub.add_argument("--max-observation-age", type=shared.positive_seconds, default=5.0)
            sub.add_argument("--refill-seconds", type=shared.positive_seconds, default=0.5, help="queued playback threshold; native scheduler also reserves measured inference headroom")
            sub.add_argument("--startup-timeout", type=shared.positive_seconds, default=120.0, help="seconds allowed for the first usable action")
            sub.add_argument("--action-timeout", type=shared.positive_seconds, default=10.0, help="seconds allowed for each background prediction; expiry stops motion")
        if command == "run":
            sub.add_argument("--confirm-robot-motion", action="store_true", help="confirm workspace, power cutoff and actual motor motion")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    if args.command == "run":
        if not args.confirm_robot_motion:
            parser.error("run requires --confirm-robot-motion after checking the workspace and emergency power cutoff")
        if sys.platform != "linux" or args.device != "cuda:0":
            parser.error("robot run requires Ubuntu/Linux and explicit CUDA; CPU is hardware-free smoke-only")
    try:
        if args.command == "prepare":
            runtime = None if args.runtime_config is None else adapters._json_object(args.runtime_config.expanduser())
            spec = adapters.prepare_deployment("rdvla", args.training_config, args.checkpoint, args.output, runtime=runtime)
            print(json.dumps(spec, indent=2))
            return
        spec = adapters.load_deployment(args.deployment_file, model="rdvla")
        config = None if args.command == "smoke" else robot_config(args)
        if args.command == "preflight":
            adapters._verify_full_descriptor(spec)
            import draccus

            print(json.dumps({
                "model": "rdvla", "transport": "local process pipes; no network",
                "artifact": spec["artifact"], "robot": draccus.encode(config),
                "execution": {
                    "mode": "async_chunk", "fps": args.fps, "execution_horizon": args.execution_horizon,
                    "refill_seconds": args.refill_seconds, "startup_timeout": args.startup_timeout,
                    "action_timeout": args.action_timeout, "max_observation_age": args.max_observation_age,
                },
            }, indent=2))
            print("PASS: local configuration and artifact verified; no GPU, cameras, serial ports or motors opened")
            return
        with closing(load_runner(spec, args)) as runner:
            started = time.monotonic()
            chunk = runner.predict(shared.synthetic_observation())
            runner.reset(full=True)
            print(json.dumps({
                "status": "passed", "kind": "synthetic local checkpoint inference; no hardware",
                "model": "rdvla", "device": args.device, "artifact": spec["artifact"],
                "shape": list(chunk.canonical_actions.shape), "seconds": time.monotonic() - started,
            }, indent=2), flush=True)
            if args.command == "run":
                def interrupt(signum, frame):
                    raise KeyboardInterrupt

                previous = signal.signal(signal.SIGTERM, interrupt)
                try:
                    run_robot(config, runner, args)
                finally:
                    signal.signal(signal.SIGTERM, previous)
    except (OSError, ValueError, RuntimeError) as exc:
        parser.exit(1, f"{type(exc).__name__}: {exc}\n")


if __name__ == "__main__":
    main()
