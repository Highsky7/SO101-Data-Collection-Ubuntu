"""Build the pinned macOS D455 SDK repair without changing global packages.

Prerequisites and source checkout commands are in the repository README.md.
Only the native SDK library changes; the original Python bindings and libusb stay intact.
"""

import base64
import csv
import hashlib
import io
import json
import platform
from pathlib import Path
import subprocess
import sys
import urllib.request
import zipfile

SDK_REVISION = "38a41441971387197193ad3aeae3cefe6a11f2cb"
USB_REVISION = "d52e355daa09f17ce64819122cb067b8a2ee0d4b"
BASE_VERSION = "2.56.5"
VERSION = "2.56.5+macos.rgb1"
TAG = "cp312-cp312-macosx_15_0_arm64"
WHEEL_URL = (
    "https://files.pythonhosted.org/packages/45/42/"
    "df369b9c8943395e8166dfc2006163279a850d7990e9c7fb7a0f0bdc4bbb/"
    f"pyrealsense2_macosx-{BASE_VERSION}-{TAG}.whl"
)
WHEEL_SHA256 = "7a0c2e124d74da97c3a928f0dc084c51f1473745a863db08cc2191bd2bdf8623"


def run(*args: str) -> None:
    subprocess.run(args, check=True)


def main() -> None:
    if sys.platform != "darwin" or platform.machine() != "arm64" or sys.version_info[:2] != (3, 12):
        raise RuntimeError("This wheel build requires macOS arm64 and Python 3.12.")
    project = Path(__file__).resolve().parents[1]
    cache = Path.home() / ".local/share/so101-data-collection"
    source = cache / "librealsense-v2.56.5"
    usb_source = cache / "libusb-v1.0.27"
    tools = cache / "sdk-build-tools/bin"
    build = cache / "librealsense-build-noimu"
    patch = Path(__file__).with_name("macos-d455.patch")
    output = project / "wheels" / f"pyrealsense2_macosx-{VERSION}-{TAG}.whl"
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    for checkout, revision in ((source, SDK_REVISION), (usb_source, USB_REVISION)):
        actual = subprocess.check_output(["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True).strip()
        if actual != revision:
            raise RuntimeError(f"Unexpected source revision at {checkout}: {actual}")
    applied = subprocess.run(
        ["git", "-C", str(source), "apply", "--reverse", "--check", str(patch)],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    ).returncode == 0
    if not applied:
        run("git", "-C", str(source), "apply", "--check", str(patch))
        run("git", "-C", str(source), "apply", str(patch))

    base_wheel = cache / f"pyrealsense2_macosx-{BASE_VERSION}-{TAG}.whl"
    if not base_wheel.exists():
        with urllib.request.urlopen(WHEEL_URL, timeout=60) as response:
            payload = response.read()
        if hashlib.sha256(payload).hexdigest() != WHEEL_SHA256:
            raise RuntimeError("Original wheel SHA256 mismatch.")
        with base_wheel.open("xb") as stream:
            stream.write(payload)
    payload = base_wheel.read_bytes()
    if hashlib.sha256(payload).hexdigest() != WHEEL_SHA256:
        raise RuntimeError("Cached original wheel SHA256 mismatch.")
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        files = {name: archive.read(name) for name in archive.namelist() if not name.endswith("/")}
    usb_library = build / "usb/libusb-1.0.0.dylib"
    usb_library.parent.mkdir(parents=True, exist_ok=True)
    usb_library.write_bytes(files["pyrealsense2/.dylibs/libusb-1.0.0.dylib"])
    flags = [
        "-DCMAKE_BUILD_TYPE=Release", "-DCMAKE_OSX_ARCHITECTURES=arm64",
        "-DCMAKE_OSX_DEPLOYMENT_TARGET=15.0", "-DFORCE_RSUSB_BACKEND=ON",
        "-DHWM_OVER_XU=OFF", "-DBUILD_WITH_OPENMP=OFF",
        "-DBUILD_SHARED_LIBS=ON", "-DBUILD_EXAMPLES=OFF", "-DBUILD_TOOLS=OFF",
        "-DBUILD_GRAPHICAL_EXAMPLES=OFF", "-DBUILD_GLSL_EXTENSIONS=OFF",
        "-DBUILD_UNIT_TESTS=OFF", "-DBUILD_PYTHON_BINDINGS=OFF", "-DBUILD_RS2_ALL=OFF",
        "-DBUILD_WITH_DDS=OFF", "-DIMPORT_DEPTH_CAM_FW=OFF", "-DCHECK_FOR_UPDATES=OFF",
        "-DENABLE_CCACHE=OFF", "-DCMAKE_SKIP_RPATH=ON",
        "-DCMAKE_INSTALL_NAME_DIR=/DLC/pyrealsense2/.dylibs",
        f"-DLIBUSB_LIB={usb_library}", f"-DLIBUSB_INC={usb_source / 'libusb'}",
        f"-DCMAKE_MAKE_PROGRAM={tools / 'ninja'}",
    ]
    run(str(tools / "cmake"), "-S", str(source), "-B", str(build), "-G", "Ninja", *flags)
    run(str(tools / "cmake"), "--build", str(build), "--target", "realsense2", "--parallel", "4")
    library = build / "Release/librealsense2.2.56.5.dylib"
    run("xcrun", "install_name_tool", "-change", "/DLC/pyrealsense2/.dylibs/libusb-1.0.0.dylib",
        "@loader_path/libusb-1.0.0.dylib", str(library))
    run("codesign", "--force", "--sign", "-", str(library))
    files["pyrealsense2/.dylibs/librealsense2.2.56.5.dylib"] = library.read_bytes()
    old_info = f"pyrealsense2_macosx-{BASE_VERSION}.dist-info"
    info = f"pyrealsense2_macosx-{VERSION}.dist-info"
    files = {name.replace(old_info + "/", info + "/", 1): data for name, data in files.items()}
    metadata = files[f"{info}/METADATA"].decode()
    old_version = f"Version: {BASE_VERSION}\n"
    if metadata.count(old_version) != 1:
        raise RuntimeError("Unexpected original wheel metadata.")
    files[f"{info}/METADATA"] = metadata.replace(old_version, f"Version: {VERSION}\n", 1).encode()
    files[f"{info}/D455_MACOS_BUILD.json"] = (json.dumps({
        "sdk_revision": SDK_REVISION, "usb_header_revision": USB_REVISION,
        "base_wheel_sha256": WHEEL_SHA256,
        "patch_sha256": hashlib.sha256(patch.read_bytes()).hexdigest(),
        "cmake_flags": flags, "imu_supported": False,
        "python_bindings_and_libusb": "unchanged from original wheel",
    }, indent=2) + "\n").encode()
    record = f"{info}/RECORD"
    files.pop(record)
    rows = io.StringIO(newline="")
    writer = csv.writer(rows, lineterminator="\n")
    for name, data in sorted(files.items()):
        digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()
        writer.writerow((name, f"sha256={digest}", len(data)))
    writer.writerow((record, "", ""))
    files[record] = rows.getvalue().encode()
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, data in sorted(files.items()):
            archive.writestr(name, data)
    print(f"Built: {output}")
    print(f"SHA256: {hashlib.sha256(output.read_bytes()).hexdigest()}")


if __name__ == "__main__":
    main()
