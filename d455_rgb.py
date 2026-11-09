"""Validate D455 dedicated RGB and optionally serve it to local, unprivileged LeRobot."""

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
import math
from pathlib import Path
import sys
import time

WIDTH = 640
HEIGHT = 480
FPS = 30
FRAME_COUNT = 60
READ_TIMEOUT_MS = 5000


def _nonnegative_index(value: str) -> int:
    try:
        index = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer camera index") from exc
    if index < 0:
        raise argparse.ArgumentTypeError("must be a non-negative camera index")
    return index


def _prepare_output_dir(path: Path) -> Path:
    path = path.expanduser().resolve()
    if path.exists():
        if not path.is_dir():
            raise ValueError(f"output path is not a directory: {path}")
        if next(path.iterdir(), None) is not None:
            raise FileExistsError(f"output directory is not empty; preserving existing files: {path}")
    else:
        path.mkdir(parents=True)
    return path


def _select_serial(camera_class, requested_serial: str | None) -> str:
    if requested_serial is not None:
        return requested_serial

    d455s = [
        camera for camera in camera_class.find_cameras()
        if "D455" in str(camera.get("name", "")).upper()
    ]
    if len(d455s) != 1:
        serials = [str(camera.get("id", "?")) for camera in d455s]
        if not d455s:
            raise RuntimeError("no D455 found; connect one or supply --serial")
        raise RuntimeError(f"found {len(d455s)} D455 cameras; supply --serial to select one: {serials}")

    serial = str(d455s[0].get("id", ""))
    if not serial:
        raise RuntimeError("discovered D455 has no serial number")
    return serial


def _stream_details(profile) -> dict:
    details = {
        "stream_name": str(profile.stream_name()),
        "stream_type": str(profile.stream_type()),
        "stream_index": int(profile.stream_index()),
        "profile_id": int(profile.unique_id()),
        "format": str(profile.format().name),
        "fps": int(profile.fps()),
    }
    if profile.is_video_stream_profile():
        video = profile.as_video_stream_profile()
        details.update(width=int(video.width()), height=int(video.height()))
    return details


def _inspect_active_rgb(camera, rs) -> tuple[dict, dict, list[dict]]:
    profile = camera.rs_profile
    if profile is None:
        raise RuntimeError("camera connected without an active RealSense profile")

    device = profile.get_device()
    device_fields = {
        "name": rs.camera_info.name,
        "serial_number": rs.camera_info.serial_number,
        "firmware_version": rs.camera_info.firmware_version,
        "usb_type_descriptor": rs.camera_info.usb_type_descriptor,
        "physical_port": rs.camera_info.physical_port,
        "product_id": rs.camera_info.product_id,
        "product_line": rs.camera_info.product_line,
    }
    device_info = {key: str(device.get_info(value)) for key, value in device_fields.items()}
    if "D455" not in device_info["name"].upper():
        raise RuntimeError(f"selected device is not a D455: {device_info['name']!r}")
    if device_info["serial_number"] != camera.serial_number:
        raise RuntimeError(
            f"active device serial {device_info['serial_number']!r} does not match selected serial "
            f"{camera.serial_number!r}"
        )

    color_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()
    if color_profile.stream_type() != rs.stream.color:
        raise RuntimeError(f"active stream is not color: {_stream_details(color_profile)}")
    stream_info = _stream_details(color_profile)
    if stream_info["format"].lower() != "rgb8":
        raise RuntimeError(f"active color stream is not RGB8: {stream_info}")
    if (stream_info["width"], stream_info["height"], stream_info["fps"]) != (WIDTH, HEIGHT, FPS):
        raise RuntimeError(
            f"active color profile is {stream_info['width']}x{stream_info['height']}@{stream_info['fps']}; "
            f"requested {WIDTH}x{HEIGHT}@{FPS}"
        )

    active_profile_id = int(color_profile.unique_id())
    sensor_info = []
    profile_owners = []
    rgb_sensor_count = 0
    for sensor in device.query_sensors():
        name = str(sensor.get_info(rs.camera_info.name))
        active_profiles = sensor.get_active_streams()
        sensor_info.append(
            {"name": name, "active_streams": [_stream_details(item) for item in active_profiles]}
        )
        if name == "RGB Camera":
            rgb_sensor_count += 1
        for item in active_profiles:
            if int(item.unique_id()) == active_profile_id:
                profile_owners.append(name)

    if rgb_sensor_count != 1:
        names = [sensor["name"] for sensor in sensor_info]
        raise RuntimeError(f"expected one dedicated 'RGB Camera' sensor; found sensors: {names}")
    if profile_owners != ["RGB Camera"]:
        raise RuntimeError(
            f"active color profile ID {active_profile_id} is not owned only by the dedicated "
            f"'RGB Camera' sensor (owners: {profile_owners})"
        )

    stream_info["sensor"] = "RGB Camera"
    return device_info, stream_info, sensor_info


def _write_json_exclusive(path: Path, value: dict) -> None:
    with path.open("x", encoding="utf-8") as output:
        json.dump(value, output, indent=2, sort_keys=True)
        output.write("\n")


def _encode_rgb_png(image, *, transport: bool = False) -> bytes:
    import cv2

    # LeRobot 0.7.0 ImageServer packs RGB directly; ZMQCamera returns imdecode's BGR
    # bytes unchanged. Preserve that wire order, but save standard-color PNG files.
    pixels = image if transport else cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    encoded_ok, encoded = cv2.imencode(
        ".png", pixels, [cv2.IMWRITE_PNG_COMPRESSION, 1]
    )
    if not encoded_ok:
        raise RuntimeError("OpenCV could not encode the captured RGB frame as PNG")
    return encoded.tobytes()


def _capture(camera, np) -> tuple[object, list[dict], dict]:
    start_utc = datetime.now(timezone.utc).isoformat()
    capture_start = time.perf_counter()
    frames = []
    last_image = None

    for sequence in range(1, FRAME_COUNT + 1):
        read_start = time.perf_counter()
        image = camera.async_read(timeout_ms=READ_TIMEOUT_MS)
        read_end = time.perf_counter()
        if image.shape != (HEIGHT, WIDTH, 3) or image.dtype != np.uint8:
            raise RuntimeError(
                f"frame {sequence} has shape/dtype {image.shape}/{image.dtype}; "
                f"expected {(HEIGHT, WIDTH, 3)}/uint8"
            )
        if sequence == FRAME_COUNT:
            last_image = image.copy()
        frames.append({
            "sequence": sequence,
            "read_started_monotonic_seconds": round(read_start, 6),
            "read_completed_monotonic_seconds": round(read_end, 6),
            "read_duration_ms": round((read_end - read_start) * 1000, 3),
        })

    return last_image, frames, {
        "started_at_utc": start_utc,
        "duration_seconds": round(time.perf_counter() - capture_start, 3),
        "observed_frame_count": len(frames),
        "read_timeout_ms": READ_TIMEOUT_MS,
        "timing_semantics": "host API read times, not sensor timestamps or a sustained-FPS guarantee",
        "frames": frames,
    }


def _serve_rgb(camera, port: int, seconds: float) -> None:
    import base64
    import zmq

    context = zmq.Context()
    socket = context.socket(zmq.PUB)
    socket.setsockopt(zmq.SNDHWM, 1)
    socket.setsockopt(zmq.LINGER, 0)
    try:
        address = f"tcp://127.0.0.1:{port}"
        socket.bind(address)
        print(f"Serving dedicated RGB as lossless PNG at {address}; Ctrl+C stops the camera.", flush=True)
        started = time.perf_counter()
        while seconds == 0 or time.perf_counter() - started < seconds:
            image = camera.async_read(timeout_ms=READ_TIMEOUT_MS)
            socket.send_json({
                "images": {"top": base64.b64encode(_encode_rgb_png(image, transport=True)).decode("ascii")},
                "timestamps": {"top": time.time()},
            })
    finally:
        socket.close()
        context.term()


def run(args: argparse.Namespace, output_dir: Path) -> None:
    import numpy as np
    import pyrealsense2 as rs
    from lerobot.cameras.configs import ColorMode
    from lerobot.cameras.realsense import RealSenseCamera, RealSenseCameraConfig
    from lerobot.cameras.utils import make_cameras_from_configs

    serial = _select_serial(RealSenseCamera, args.serial)
    config = RealSenseCameraConfig(
        serial_number_or_name=serial, fps=FPS, width=WIDTH, height=HEIGHT,
        color_mode=ColorMode.RGB, use_rgb=True, use_depth=False,
    )
    camera = make_cameras_from_configs({"top": config})["top"]
    try:
        camera.connect(warmup=False)
        device_info, stream_info, sensor_info = _inspect_active_rgb(camera, rs)
        last_image, frames, capture_info = _capture(camera, np)
        if last_image is None:
            raise RuntimeError("capture completed without a final RGB image")

        png_path = output_dir / "d455_rgb.png"
        metadata_path = output_dir / "metadata.json"
        cameras_path = output_dir / "cameras.json"
        with png_path.open("xb") as output:
            output.write(_encode_rgb_png(last_image))
        top = {
            "type": "intelrealsense", "serial_number_or_name": device_info["serial_number"],
            "fps": FPS, "width": WIDTH, "height": HEIGHT, "color_mode": "rgb",
            "use_rgb": True, "use_depth": False,
        }
        if args.serve:
            top = {
                "type": "zmq", "server_address": "127.0.0.1", "port": args.port,
                "camera_name": "top", "fps": FPS, "width": WIDTH, "height": HEIGHT,
                "color_mode": "rgb",
            }
        cameras = {"top": top}
        if args.wrist_index is not None:
            cameras["wrist"] = {
                "type": "opencv", "index_or_path": args.wrist_index,
                "width": WIDTH, "height": HEIGHT, "fps": FPS,
            }
        _write_json_exclusive(cameras_path, cameras)

        sdk_package = "pyrealsense2-macosx" if sys.platform == "darwin" else "pyrealsense2"
        _write_json_exclusive(metadata_path, {
            "versions": {
                "lerobot": importlib.metadata.version("lerobot"),
                "pyrealsense2_package": sdk_package,
                "pyrealsense2_package_version": importlib.metadata.version(sdk_package),
                "pyrealsense2_sdk_version": getattr(rs, "__version__", None),
            },
            "device": device_info, "sensors": sensor_info, "active_stream": stream_info,
            "requested_stream": {
                "stream": "color", "format": "rgb8", "width": WIDTH, "height": HEIGHT, "fps": FPS,
            },
            "capture": capture_info,
            "artifacts": {"rgb_png": png_path.name, "cameras_json": cameras_path.name},
        })
        print(
            f"PASS: {device_info['name']} serial={device_info['serial_number']}; "
            f"{stream_info['sensor']} {stream_info['format']} "
            f"{stream_info['width']}x{stream_info['height']}@{stream_info['fps']}; {len(frames)} frames",
            flush=True,
        )
        print(f"Artifacts: {png_path}, {metadata_path}, {cameras_path}", flush=True)
        if args.serve:
            _serve_rgb(camera, args.port, args.serve_seconds)
    finally:
        if camera.is_connected or camera.thread is not None:
            camera.disconnect()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--serial", help="D455 serial number; if omitted, require exactly one connected D455")
    parser.add_argument("--wrist-index", type=_nonnegative_index, help="write this OpenCV index into cameras.json without opening it")
    parser.add_argument("--serve", action="store_true", help="publish lossless PNG on loopback for LeRobot's ZMQ backend")
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument("--serve-seconds", type=float, default=0, help="stop serving after this many seconds; 0 runs until Ctrl+C")
    args = parser.parse_args()
    if not 1024 <= args.port <= 65535:
        parser.error("--port must be between 1024 and 65535")
    if not math.isfinite(args.serve_seconds) or args.serve_seconds < 0:
        parser.error("--serve-seconds must be finite and non-negative")
    if args.serial is not None:
        args.serial = args.serial.strip()
        if not args.serial:
            parser.error("--serial cannot be empty")

    output_dir = _prepare_output_dir(args.output_dir)
    try:
        run(args, output_dir)
    except KeyboardInterrupt:
        print("Stopped D455 camera.", file=sys.stderr)


if __name__ == "__main__":
    main()
