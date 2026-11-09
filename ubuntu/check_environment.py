"""Exercise local LeRobot video recording, fresh-process resume, and decoding.

Uses synthetic images and joint values only. Never opens cameras or serial ports,
never moves a robot, and never uploads data. Temporary data is removed on exit.
"""

import argparse
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

FPS = 30
FRAMES = 12
REPO_ID = "local/so101_environment_check"
JOINTS = [
    "shoulder_pan.pos", "shoulder_lift.pos", "elbow_flex.pos",
    "wrist_flex.pos", "wrist_roll.pos", "gripper.pos",
]
CAMERAS = ("top", "wrist")


def expected_state(episode, frame):
    import numpy as np
    return np.array([episode * 20 + frame, 2, 3, 4, 5, 6], dtype=np.float32)


def expected_color(episode, frame, camera):
    return (40 + episode * 40 + frame * 5, 50, 100) if camera == "top" else (20, 120, 50 + episode * 40 + frame * 5)


def record_episode(root, episode, codec):
    import numpy as np
    from lerobot.configs import RGBEncoderConfig
    from lerobot.datasets import LeRobotDataset

    features = {
        "observation.state": {"dtype": "float32", "shape": (6,), "names": JOINTS},
        "action": {"dtype": "float32", "shape": (6,), "names": JOINTS},
        **{
            f"observation.images.{camera}": {
                "dtype": "video", "shape": (480, 640, 3),
                "names": ["height", "width", "channels"],
            }
            for camera in CAMERAS
        },
    }
    options = {
        "root": root,
        "rgb_encoder": RGBEncoderConfig(vcodec=codec, crf=23),
        "streaming_encoding": True,
        "encoder_threads": 2,
    }
    if episode == 0:
        dataset = LeRobotDataset.create(
            REPO_ID, fps=FPS, features=features,
            robot_type="so101_follower", **options,
        )
    else:
        dataset = LeRobotDataset.resume(REPO_ID, **options)
    try:
        if dataset.num_episodes != episode:
            raise AssertionError(f"Unexpected resume position: {dataset.num_episodes}")
        for frame in range(FRAMES):
            state = expected_state(episode, frame)
            images = {}
            for camera in CAMERAS:
                image = np.empty((480, 640, 3), dtype=np.uint8)
                image[:] = expected_color(episode, frame, camera)
                images[f"observation.images.{camera}"] = image
            dataset.add_frame({
                "observation.state": state,
                "action": state + np.float32(0.5),
                **images,
                "task": "Synthetic software check; not a robot demonstration",
            })
        dataset.save_episode()
    finally:
        dataset.finalize()
    print(f"PASS: saved synthetic episode {episode} with {codec}", flush=True)


def verify_dataset(root):
    import numpy as np
    import torch
    from lerobot.datasets import LeRobotDataset

    dataset = LeRobotDataset(REPO_ID, root=root, return_uint8=True)
    if dataset.num_episodes != 2 or dataset.num_frames != 2 * FRAMES:
        raise AssertionError("Episode/frame count changed across fresh-process resume")
    for index in (0, FRAMES - 1, FRAMES, 2 * FRAMES - 1):
        episode, frame = divmod(index, FRAMES)
        sample = dataset[index]
        if sample["episode_index"].item() != episode or sample["frame_index"].item() != frame:
            raise AssertionError(f"Wrong episode boundary at index {index}")
        np.testing.assert_allclose(sample["observation.state"].numpy(), expected_state(episode, frame))
        np.testing.assert_allclose(sample["action"].numpy(), expected_state(episode, frame) + 0.5)
        np.testing.assert_allclose(sample["timestamp"].item(), frame / FPS, atol=1e-6)
        for camera in CAMERAS:
            image = sample[f"observation.images.{camera}"]
            if image.shape != (3, 480, 640) or image.dtype != torch.uint8:
                raise AssertionError(f"Unexpected image representation: {camera}")
            np.testing.assert_allclose(
                image[:, 240, 320].numpy().astype(float),
                expected_color(episode, frame, camera), atol=8,
            )
    print("PASS: default video decoder, both camera identities, state/action values, timestamps, and episode boundaries", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--codec",
        default="h264_videotoolbox" if sys.platform == "darwin" else "h264",
        help="Video encoder (default: h264_videotoolbox on macOS, h264 elsewhere)",
    )
    parser.add_argument("--phase", choices=("create", "resume", "verify"))
    parser.add_argument("--root", type=Path)
    args = parser.parse_args()
    if args.phase:
        if args.root is None:
            parser.error("--root is required with --phase")
        if args.phase == "verify":
            verify_dataset(args.root)
        else:
            record_episode(args.root, int(args.phase == "resume"), args.codec)
        return

    for module in ("cv2", "av", "torch", "torchcodec", "rustypot", "serial", "lerobot.robots.so_follower", "lerobot.teleoperators.so_leader"):
        importlib.import_module(module)
    packages = ("lerobot", "torch", "torchvision", "torchcodec", "av", "opencv-python-headless", "rustypot", "rerun-sdk")
    print(json.dumps({name: importlib.metadata.version(name) for name in packages}, indent=2), flush=True)
    with tempfile.TemporaryDirectory(prefix="so101-software-check-") as directory:
        root = Path(directory) / "dataset"
        environment = {
            **os.environ, "HF_HUB_OFFLINE": "1", "HF_HUB_DISABLE_TELEMETRY": "1",
            "HF_HOME": str(Path(directory) / "hf-cache"), "PYTHONUNBUFFERED": "1",
        }
        for phase in ("create", "resume", "verify"):
            subprocess.run(
                [sys.executable, str(Path(__file__).resolve()), "--phase", phase,
                 "--root", str(root), "--codec", args.codec],
                env=environment, check=True,
            )
    print("PASS: hardware-free software check complete; temporary synthetic data removed.")
    print("NOT TESTED: physical cameras, robot motion/calibration, real-time cadence, or power-loss recovery.")


if __name__ == "__main__":
    main()
