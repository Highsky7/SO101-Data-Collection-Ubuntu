"""SO101 multi-model deployment using LeRobot's native remote server and rollout.

preflight and smoke never open cameras or serial ports. Only run connects hardware.
Both cameras reuse the original collection guide's OpenCV capture configuration.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from uuid import uuid4

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
# These overlays never modify the collection venv or the server's training env.
for directory in (HERE / "runtime" / "deps", HERE / "runtime" / "vendor"):
    if directory.is_dir():
        sys.path.insert(0, str(directory))

GPU0 = "GPU-bdb795ca-21c6-9a7c-7691-2511db60090f"
SEMANTICS = "so101.absolute-position.degrees.gripper-0-100.rgb.v1"
PREDICTION_STEPS = 50
JOINTS = (
    "shoulder_pan.pos", "shoulder_lift.pos", "elbow_flex.pos",
    "wrist_flex.pos", "wrist_roll.pos", "gripper.pos",
)
CHECKPOINT_ROOT = Path(
    "/data/seungminlee/so101-rdvla-smolvla-h50-e20-20261108/runs/smolvla/checkpoints"
)
# Epoch 10 matches the previously requested HF artifact; epoch 20 is the completed
# local training export. Their seven metadata files match; only model weights differ.
POLICY_HASHES = {
    10: {
        "adapter.json": "f44460cae41303c63bfd14dbc802364b238c0b51d12b381609854026733eb6eb",
        "config.json": "7458823afc49a33eb934d04b43e74d5fa59970b6731ba55dd2d1285b22dbdf62",
        "model.safetensors": "63ed473e1671bd50e030ebd469872ffeef1976bfa6c5fb9db5f56c057c35c55c",
        "vlm/chat_template.jinja": "830005085ddd94cec605b6e21b885b28ffad6e3f82ee6a212c523533e8f04f01",
        "vlm/config.json": "874d9a23e507677891a46ee781f6c791ca62bbf1edad495cf471d4d97f5f36ac",
        "vlm/processor_config.json": "ea77bb48f233947b5d1ee3aa6547dd27663ba7b91c059446edd7fd75c6d1d7aa",
        "vlm/tokenizer.json": "b90474ce689a6f06dc920d3ebdb1f9cb8d3608da9e93b8b4eae63e9101bbc5d5",
        "vlm/tokenizer_config.json": "6d2b6de492a5e53f6a740c37d37ef8c507a4e1a96f7e70f688dbf872b00be38e",
    }
}
POLICY_HASHES[20] = {
    **POLICY_HASHES[10],
    "model.safetensors": "937852471906eb250565e271fd673a4ac04c58728272a3304fbb4740be8703da",
}
DEPLOYMENTS = {epoch: f"so101-smolvla-{epoch}epochs" for epoch in POLICY_HASHES}
ARTIFACTS = {
    epoch: "sha256:" + hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    for epoch, hashes in POLICY_HASHES.items()
}
MODELS = ("smolvla", "anchormix", "rdvla", "pi05")


def deployment_selection(model: str, epoch: int, deployment_file: Path | None):
    if model == "smolvla":
        if deployment_file is not None:
            raise ValueError("SmolVLA uses --epoch and its pinned artifact, not --deployment-file")
        return DEPLOYMENTS[epoch], ARTIFACTS[epoch], None
    from so101_adapters import load_deployment

    if deployment_file is None:
        raise ValueError(f"{model} requires --deployment-file from prepare; no latest/fallback checkpoint")
    spec = load_deployment(deployment_file, model=model)
    return spec["deployment"], spec["artifact"], spec


def require_native_version() -> None:
    from lerobot import __version__

    if __version__ != "0.7.0":
        raise RuntimeError(f"Expected the collection environment's LeRobot 0.7.0, got {__version__}")


def feature_contract():
    from lerobot.inference import FeatureSpec

    state = FeatureSpec("observation.state", (6,), "float32", names=JOINTS, semantics=SEMANTICS)
    images = tuple(
        FeatureSpec(f"observation.images.{name}", (480, 640, 3), "uint8", "rgb", semantics=SEMANTICS)
        for name in ("top", "wrist")
    )
    action = FeatureSpec("action", (6,), "float32", names=JOINTS, semantics=SEMANTICS)
    return (state, *images), action


def remote_config(
    endpoint: str, *, epoch: int = 10, model: str = "smolvla",
    deployment_file: Path | None = None, log_level: str = "INFO", refill_seconds: float = 0.5,
):
    from lerobot.inference import RemoteInferenceConfig

    deployment, artifact, _ = deployment_selection(model, epoch, deployment_file)
    return RemoteInferenceConfig(
        endpoint=endpoint, deployment=deployment, semantics=SEMANTICS,
        expected_artifact=artifact, log_level=log_level, refill_seconds=refill_seconds,
    )


def verify_policy(checkpoint: Path, *, epoch: int = 10) -> tuple[Path, dict]:
    policy = checkpoint.expanduser().resolve() / "policy"
    for name, expected in POLICY_HASHES[epoch].items():
        with (policy / name).open("rb") as stream:
            actual = hashlib.file_digest(stream, "sha256").hexdigest()
        if actual != expected:
            raise ValueError(f"Checkpoint differs from the selected epoch-{epoch} artifact: policy/{name}")
    with (policy / "adapter.json").open() as stream:
        metadata = json.load(stream)
    return policy, metadata


def require_gpu0(
    expected_uuid: str = GPU0, *, allow_shared: bool = False, deployment: dict | None = None,
    memory_limit_gib: float | None = None,
) -> None:
    if allow_shared and memory_limit_gib is None:
        raise ValueError("--allow-shared-gpu requires an explicit --memory-limit-gib budget")
    if memory_limit_gib is not None:
        from so101_adapters import GPU_RESERVE_BYTES, validate_memory_limit

        memory_limit_gib = validate_memory_limit(memory_limit_gib)
    idle_mps_pids = ()
    if deployment is not None:
        raw = Path(deployment["training_config"]).read_bytes()
        if hashlib.sha256(raw).hexdigest() != deployment["training_config_sha256"]:
            raise ValueError("Training config changed since deployment preparation; refusing its MPS allowlist")
        config = json.loads(raw)
        runtime = deployment.get("runtime")
        admission_config = config if runtime is None else runtime
        if not isinstance(admission_config, dict) or admission_config.get("gpu_uuid") != expected_uuid:
            raise ValueError("Admission config differs from the selected physical GPU")
        # Training-host PID approvals must never apply to a relocated Ubuntu host.
        idle_mps_pids = config.get("idle_mps_pids", ()) if runtime is None else ()
        if not isinstance(idle_mps_pids, (list, tuple)) or any(type(pid) is not int or pid <= 0 for pid in idle_mps_pids):
            raise ValueError("idle_mps_pids must contain positive integer process IDs")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    relocated = deployment is not None and deployment.get("runtime") is not None
    allowed_visible = {expected_uuid} if relocated else {"0", expected_uuid}
    if visible is not None and visible not in allowed_visible:
        raise RuntimeError("CUDA_VISIBLE_DEVICES differs from the selected physical GPU; not overridden")
    result = subprocess.run(
        ["nvidia-smi", f"--id={expected_uuid}", "--query-gpu=index,uuid,gpu_recovery_action",
         "--format=csv,noheader"],
        check=True, capture_output=True, text=True, timeout=10,
    )
    fields = [field.strip() for field in result.stdout.strip().split(",")]
    valid_index = len(fields) == 3 and fields[0].isdigit() and (relocated or fields[0] == "0")
    if not valid_index or fields[1:] != [expected_uuid, "None"]:
        raise RuntimeError(f"Selected GPU is not ready; no reset or fallback GPU will be attempted: {fields}")
    processes = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name", "--format=csv,noheader,nounits"],
        check=True, capture_output=True, text=True, timeout=10,
    )
    peers = []
    for line in processes.stdout.splitlines():
        row = [field.strip() for field in line.split(",", 2)]
        if len(row) != 3:
            raise RuntimeError(f"Cannot identify GPU process ownership: {line}")
        if row[0] != expected_uuid or int(row[1]) == os.getpid():
            continue
        pid = int(row[1])
        if pid in idle_mps_pids and Path(row[2]).name == "nvidia-cuda-mps-server":
            logging.info("Excluding approved idle MPS server pid=%d from compute peers", pid)
            continue
        peers.append(pid)
    if peers and not allow_shared:
        raise RuntimeError(f"GPU0 has compute processes {peers}; --allow-shared-gpu is explicit opt-in; nothing stopped")
    if memory_limit_gib is not None:
        memory = subprocess.run(
            ["nvidia-smi", f"--id={expected_uuid}", "--query-gpu=memory.total,memory.free",
             "--format=csv,noheader,nounits"],
            check=True, capture_output=True, text=True, timeout=10,
        )
        total_mib, free_mib = (int(field.strip()) for field in memory.stdout.strip().split(","))
        if not 0 <= free_mib <= total_mib or memory_limit_gib > total_mib / 1024:
            raise RuntimeError("Requested inference memory budget exceeds physical GPU capacity")
        budget = int(memory_limit_gib * 1024**3)
        if budget < 1 or free_mib * 1024**2 < budget + GPU_RESERVE_BYTES:
            raise RuntimeError("Inference needs its memory limit plus 4 GiB free GPU reserve; nothing was stopped")
    # An explicit physical UUID becomes the worker's process-local cuda:0.
    os.environ["CUDA_VISIBLE_DEVICES"] = expected_uuid
    for key in ("CUDA_MPS_PIPE_DIRECTORY", "CUDA_MPS_LOG_DIRECTORY"):
        os.environ.pop(key, None)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"


def make_runner(
    checkpoint: Path, *, device: str = "cuda:0", epoch: int = 10,
    fps: float = 30.0, execution_horizon: int = PREDICTION_STEPS,
):
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError("fps must be finite and positive")
    if type(execution_horizon) is not int or not 1 <= execution_horizon <= PREDICTION_STEPS:
        raise ValueError(f"execution_horizon must be an integer from 1 to {PREDICTION_STEPS}")
    import torch
    from lerobot.configs import PreTrainedConfig
    from lerobot.inference import PolicyRunner
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from lerobot.policies.smolvla.processor_smolvla import make_smolvla_pre_post_processors
    from lerobot.processor import NormalizerProcessorStep, UnnormalizerProcessorStep

    policy_path, metadata = verify_policy(checkpoint, epoch=epoch)
    config = PreTrainedConfig.from_pretrained(policy_path, local_files_only=True)
    if config.chunk_size != PREDICTION_STEPS:
        raise ValueError("This checkpoint must retain its trained 50-step prediction")
    # Runtime playback slice only; never rewrite the pinned checkpoint config.
    config.n_action_steps = execution_horizon
    config.device = device
    config.load_vlm_weights = False
    config.vlm_model_name = str(policy_path / "vlm")
    # The export has adapter statistics, not saved native processor pipelines.
    # Use native processors once, with the same mean/std and epsilon as training.
    stats = {
        "observation.state": metadata["statistics"]["state"],
        "action": metadata["statistics"]["action"],
    }
    policy = SmolVLAPolicy.from_pretrained(
        policy_path, config=config, local_files_only=True, strict=True,
    )
    preprocessor, postprocessor = make_smolvla_pre_post_processors(config, stats)
    for pipeline, kind, required in (
        (preprocessor, NormalizerProcessorStep, {"observation.state", "action"}),
        (postprocessor, UnnormalizerProcessorStep, {"action"}),
    ):
        steps = [step for step in pipeline.steps if isinstance(step, kind)]
        if len(steps) != 1 or not required <= steps[0]._tensor_stats.keys():
            raise RuntimeError("Required training normalization statistics are missing; refusing to serve")
    features, action = feature_contract()
    runner = PolicyRunner(
        policy, preprocessor, postprocessor, action_interval=1 / fps,
        features=features, action_feature=action, robot_type="so_follower",
        language_enabled=False,
    )
    if (runner.capabilities.prediction_steps, runner.capabilities.execution_steps) != (
        PREDICTION_STEPS, execution_horizon,
    ):
        raise ValueError("Policy prediction/execution lengths differ from the deployment settings")
    if device == "cuda:0" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; CPU or another GPU is not a deployment fallback")
    return runner


def synthetic_observation():
    import numpy as np
    from lerobot.inference import ObservationSnapshot

    features = {"observation.state": np.zeros(6, dtype=np.float32)}
    for name, color in (("top", [231, 17, 63]), ("wrist", [9, 151, 247])):
        image = np.empty((480, 640, 3), dtype=np.uint8)
        image[:] = color
        features[f"observation.images.{name}"] = image
    return ObservationSnapshot(
        features, time.perf_counter(), "Synthetic software check; not a robot demonstration",
        observation_id=uuid4().hex, action_cursor=0, execution_generation=0,
    )


def serve(args) -> None:
    deployment, artifact, spec = deployment_selection(args.model, args.epoch, args.deployment_file)
    gpu_uuid = GPU0 if spec is None else spec["gpu_uuid"]
    require_gpu0(
        gpu_uuid, allow_shared=args.allow_shared_gpu, deployment=spec,
        memory_limit_gib=args.memory_limit_gib,
    )
    logging.info(
        "SO101 inference server pid=%d physical_gpu=%s model=%s fps=%.6g execution_horizon=%d deployment=%s artifact=%s",
        os.getpid(), gpu_uuid, args.model, args.fps, args.execution_horizon, deployment, artifact,
    )
    # All assets below are local. Never download a substitute VLM or execute remote code.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    require_native_version()
    from lerobot.remote_inference import PolicyServer, SessionWorker
    from lerobot.transport import ZenohConfig, ZenohTransport

    if spec is None:
        if args.memory_limit_gib is not None:
            import torch
            from so101_adapters import apply_cuda_memory_limit

            apply_cuda_memory_limit(torch, args.memory_limit_gib)
        runner = make_runner(
            args.checkpoint, epoch=args.epoch, fps=args.fps, execution_horizon=args.execution_horizon,
        )
    else:
        from so101_adapters import AdapterRunner

        features, action = feature_contract()
        runner = AdapterRunner(
            spec, features=features, action_feature=action,
            fps=args.fps, execution_horizon=args.execution_horizon,
            memory_limit_gib=args.memory_limit_gib,
        )
    previous = {}
    try:
        for _ in range(2):
            runner.predict(synthetic_observation())
        runner.reset(full=True)
        worker = SessionWorker(
            runner, deployment=deployment, artifact_identity=artifact, semantics=SEMANTICS,
        )
        server = PolicyServer(
            worker, ZenohTransport(ZenohConfig(listen_endpoints=[f"tcp/127.0.0.1:{args.port}"])),
        )
        previous = {
            sig: signal.signal(sig, lambda signum, _: server.stop(reason=signal.Signals(signum).name))
            for sig in (signal.SIGTERM, signal.SIGINT)
        }
        server.serve()
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)
        if spec is not None:
            runner.close()


def read_cameras(path: Path) -> dict:
    with path.expanduser().open() as stream:
        cameras = json.load(stream)
    if not isinstance(cameras, dict) or set(cameras) != {"top", "wrist"}:
        raise ValueError("The trained checkpoint requires exactly top and wrist cameras")
    for name, camera in cameras.items():
        if not isinstance(camera, dict):
            raise ValueError(f"Camera {name} must be a configuration object")
        if (camera.get("width"), camera.get("height"), camera.get("fps")) != (640, 480, 30):
            raise ValueError(f"Camera {name} must retain the collected 640x480 RGB at 30 FPS")
        if camera.get("color_mode", "rgb") != "rgb":
            raise ValueError(f"Camera {name} must provide RGB, not BGR/depth")
        index = camera.get("index_or_path")
        valid_index = type(index) is int and index >= 0
        valid_path = isinstance(index, str) and bool(index.strip())
        if camera.get("type") != "opencv" or not (valid_index or valid_path):
            raise ValueError(f"Camera {name} must reuse its original OpenCV index/path; no SDK camera path")
    if str(cameras["top"]["index_or_path"]) == str(cameras["wrist"]["index_or_path"]):
        raise ValueError("top and wrist must use different camera indices/paths")
    return {name: cameras[name] for name in ("top", "wrist")}


def rollout_config(args):
    if not args.robot_port.strip():
        raise ValueError("Follower port cannot be empty; identify its current USB serial path first")
    if not args.task.strip():
        raise ValueError("Task cannot be empty")
    if not args.robot_id or Path(args.robot_id).name != args.robot_id or args.robot_id in {".", ".."}:
        raise ValueError("robot-id must be a calibration filename stem")
    cameras = read_cameras(args.cameras)
    calibration_dir = PROJECT / "calibration" / "robots" / "so_follower"
    if not (calibration_dir / f"{args.robot_id}.json").is_file():
        raise FileNotFoundError(f"Preserve/use the existing follower calibration: {calibration_dir / (args.robot_id + '.json')}")
    require_native_version()
    import draccus
    from lerobot.scripts.lerobot_rollout import RolloutConfig

    deployment, artifact, _ = deployment_selection(args.model, args.epoch, args.deployment_file)
    # Parse official dataclasses only; no robot/camera instance or connection here.
    return draccus.parse(RolloutConfig, args=[
        "--strategy.type=base", "--robot.type=so101_follower",
        f"--robot.port={args.robot_port}", f"--robot.id={args.robot_id}",
        f"--robot.calibration_dir={calibration_dir}",
        f"--robot.cameras={json.dumps(cameras)}", "--inference.type=remote",
        f"--inference.endpoint={args.endpoint}", f"--inference.deployment={deployment}",
        f"--inference.semantics={SEMANTICS}", f"--inference.expected_artifact={artifact}",
        f"--inference.log_level={args.inference_log_level}",
        f"--inference.refill_seconds={args.refill_seconds}",
        f"--fps={args.fps}", f"--duration={args.duration}", f"--task={args.task}",
    ])


def client(args) -> None:
    cfg = rollout_config(args)
    if args.command == "preflight":
        import draccus

        print(json.dumps(draccus.encode(cfg), indent=2))
        print("PASS: native rollout config only; no server, camera, serial port or motor opened")
        return
    if not Path(cfg.robot.port).exists():
        raise FileNotFoundError(f"Follower serial device does not exist: {cfg.robot.port}")
    from lerobot.scripts.lerobot_rollout import rollout

    # Official LeRobot owns capture, admission, action scheduling, dispatch and teardown.
    rollout(cfg)


def smoke(args) -> None:
    import torch
    from lerobot.inference import ChunkRequest, ExecutionMode
    from lerobot.policies import QueueSnapshot
    from lerobot.remote_inference import RemoteClient

    require_native_version()
    config = remote_config(
        args.endpoint, epoch=args.epoch, model=args.model, deployment_file=args.deployment_file,
        log_level=args.inference_log_level, refill_seconds=args.refill_seconds,
    )
    client = RemoteClient.connect(config)
    try:
        capabilities = client.capabilities
        if (
            capabilities.prediction_steps != PREDICTION_STEPS
            or not 1 <= capabilities.execution_steps <= PREDICTION_STEPS
        ):
            raise ValueError("Remote deployment differs from this checkpoint's 50-step prediction contract")
        features, action = feature_contract()
        client.admit(
            features=features, action_feature=action, semantics=SEMANTICS,
            action_interval=1 / args.fps, mode="chunk",
        )
        started = time.perf_counter()
        chunk = client.infer(ChunkRequest(
            observation=synthetic_observation(),
            continuation=QueueSnapshot(0, 0, 0, None, None, ()),
            request_id=uuid4().hex, generation=0, mode=ExecutionMode.CHUNK, delay=0,
            submitted_at=started,
        ))
        actions = chunk.canonical_actions
        if (
            tuple(actions.shape) != (capabilities.execution_steps, len(JOINTS))
            or actions.dtype != torch.float32 or not torch.isfinite(actions).all()
        ):
            raise AssertionError("Remote checkpoint must return finite float32 [execution_horizon,6] actions")
        print(json.dumps({
            "status": "passed", "kind": "synthetic remote checkpoint inference; no hardware",
            "model": args.model, "deployment": config.deployment,
            **({"epoch": args.epoch} if args.model == "smolvla" else {}),
            "artifact": client.artifact_identity, "shape": list(actions.shape),
            "fps": 1 / capabilities.action_interval,
            "prediction_steps": capabilities.prediction_steps,
            "execution_horizon": capabilities.execution_steps,
            "execution_horizon_seconds": capabilities.execution_steps * capabilities.action_interval,
            "roundtrip_seconds": time.perf_counter() - started,
            "server_durations": chunk.server_durations,
        }, indent=2))
    finally:
        client.close()


def positive_seconds(value: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("seconds must be finite and positive")
    return number


def duration_seconds(value: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise argparse.ArgumentTypeError("duration must be finite and nonnegative; 0 means no time limit")
    return number


def positive_fps(value: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("fps must be finite and positive")
    return number


def execution_horizon_steps(value: str) -> int:
    try:
        number = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("execution horizon must be an integer") from exc
    if not 1 <= number <= PREDICTION_STEPS:
        raise argparse.ArgumentTypeError(f"execution horizon must be between 1 and {PREDICTION_STEPS} actions")
    return number


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare", help="verify a committed training checkpoint and pin its deployment; no GPU or hardware")
    prepare.add_argument("--model", choices=MODELS[1:], required=True)
    prepare.add_argument("--training-config", type=Path, required=True)
    prepare.add_argument("--checkpoint", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    server = commands.add_parser("server", help="serve one pinned checkpoint on its configured physical GPU0; no robot")
    server.add_argument("--epoch", type=int, choices=tuple(POLICY_HASHES), help="explicit trained SmolVLA epoch; defaults to 10")
    server.add_argument("--checkpoint", type=Path, help="override checkpoint directory; must match the selected epoch's pinned hashes")
    server.add_argument("--model", choices=MODELS, default="smolvla")
    server.add_argument("--deployment-file", type=Path, help="prepared exact artifact descriptor for non-SmolVLA models")
    server.add_argument("--allow-shared-gpu", action="store_true", help="explicitly allow concurrent GPU0 compute; never stops existing jobs")
    server.add_argument("--memory-limit-gib", type=float, help="PyTorch GPU allocator limit in GiB; required with --allow-shared-gpu; needs 4 GiB free reserve")
    server.add_argument("--port", type=int, default=7447)
    server.add_argument("--fps", type=positive_fps, default=30.0, help="action playback Hz; clients must use the same --fps")
    server.add_argument(
        "--execution-horizon", type=execution_horizon_steps, default=PREDICTION_STEPS,
        help="maximum playback slice from each 50-step prediction; clients read it from the server",
    )
    for command in ("preflight", "run", "smoke"):
        sub = commands.add_parser(command, help="connects and MOVES THE ROBOT" if command == "run" else "no hardware access")
        sub.add_argument("--endpoint", default="tcp/127.0.0.1:7447")
        sub.add_argument("--epoch", type=int, choices=tuple(POLICY_HASHES), help="SmolVLA epoch; defaults to 10 and must match server")
        sub.add_argument("--model", choices=MODELS, default="smolvla")
        sub.add_argument("--deployment-file", type=Path, help="copy of the server's prepared exact artifact descriptor")
        sub.add_argument("--fps", type=positive_fps, default=30.0, help="policy/control Hz; must match the server (camera capture stays at 30 FPS)")
        sub.add_argument("--inference.log_level", dest="inference_log_level", choices=("INFO", "DEBUG"), default="INFO")
        sub.add_argument(
            "--inference.refill_seconds", dest="refill_seconds", type=positive_seconds, default=0.5,
            help="remaining playback that triggers refill; earlier requests cannot fix sustained excessive latency",
        )
        if command != "smoke":
            sub.add_argument("--robot-port", required=True)
            sub.add_argument("--robot-id", default="so101_follower_01")
            sub.add_argument("--cameras", type=Path, required=True, help="JSON file containing the original top/wrist OpenCV CAMERAS configuration")
            sub.add_argument("--task", required=True)
            sub.add_argument(
                "--duration", type=duration_seconds, required=True,
                help="seconds; 0 disables the duration limit (Ctrl+C to stop)",
            )
    args = parser.parse_args()
    if args.command == "prepare":
        from so101_adapters import prepare_deployment

        spec = prepare_deployment(args.model, args.training_config, args.checkpoint, args.output)
        print(json.dumps(spec, indent=2))
        return
    if args.model != "smolvla" and args.epoch is not None:
        parser.error("--epoch selects only SmolVLA; other models use --deployment-file")
    if args.epoch is None:
        args.epoch = 10
    try:
        if args.command == "server":
            if args.allow_shared_gpu and args.memory_limit_gib is None:
                raise ValueError("--allow-shared-gpu requires an explicit --memory-limit-gib budget")
            if args.memory_limit_gib is not None:
                from so101_adapters import validate_memory_limit

                args.memory_limit_gib = validate_memory_limit(args.memory_limit_gib)
        deployment_selection(args.model, args.epoch, args.deployment_file)
        if args.model != "smolvla" and args.command == "server" and args.checkpoint is not None:
            parser.error("non-SmolVLA --checkpoint is pinned by prepare, not a server override")
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    if args.command == "server" and args.model == "smolvla" and args.checkpoint is None:
        args.checkpoint = CHECKPOINT_ROOT / f"epoch-{args.epoch}"
    if args.command == "server" and not 1024 <= args.port <= 65535:
        parser.error("--port must be between 1024 and 65535")
    logging.basicConfig(
        level=getattr(logging, getattr(args, "inference_log_level", "INFO")),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    if args.command == "server":
        serve(args)
    elif args.command == "smoke":
        smoke(args)
    else:
        client(args)


if __name__ == "__main__":
    main()
