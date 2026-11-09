#!/usr/bin/env bash
# Install software only: never open cameras/serial ports or operate a robot.
set -euo pipefail

if [[ ${1:-} == --help && $# == 1 ]]; then
  printf '%s\n' \
    'Usage: bash setup_ubuntu.sh' \
    'Requires Ubuntu x86_64, uv, FFmpeg, and a C compiler (build-essential).' \
    'Creates a new project-local Python 3.12 .venv; never replaces an existing one.' \
    'Does not install apt packages, change permissions, or run hardware commands.'
  exit 0
fi
if [[ $# != 0 ]]; then
  printf 'Unexpected argument. Run: bash setup_ubuntu.sh --help\n' >&2
  exit 2
fi

PROJECT="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
if [[ $(uname -s) != Linux || ! -r /etc/os-release ]]; then
  printf 'This installer is for Ubuntu Linux. The original macOS environment is unchanged.\n' >&2
  exit 1
fi
. /etc/os-release
if [[ ${ID:-} != ubuntu || $(uname -m) != x86_64 ]]; then
  printf 'requirements.lock targets Ubuntu x86_64 with Python 3.12.\n' >&2
  exit 1
fi
for program in uv ffmpeg cc; do
  if ! command -v "$program" >/dev/null 2>&1; then
    printf 'Missing prerequisite: %s. See README.md before installing.\n' "$program" >&2
    exit 1
  fi
done
if [[ -e "$PROJECT/.venv" || -L "$PROJECT/.venv" ]]; then
  printf 'Refusing to replace existing %s/.venv. Activate it and check it, or use a fresh checkout.\n' "$PROJECT" >&2
  exit 1
fi

uv venv --python 3.12 "$PROJECT/.venv"
uv pip sync "$PROJECT/requirements.lock" \
  --python "$PROJECT/.venv/bin/python" \
  --require-hashes \
  --torch-backend cpu \
  --only-binary :all: \
  --no-binary evdev
uv pip check --python "$PROJECT/.venv/bin/python"
printf '\nInstalled. Next: cd "%s" && source .venv/bin/activate\n' "$PROJECT"
printf 'Then run: python check_environment.py\n'
