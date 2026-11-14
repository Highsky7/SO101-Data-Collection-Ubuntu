# SO-ARM101 데이터 수집 및 로컬·원격 추론

LeRobot 0.7.0 기반 SO101 leader–follower 데이터 수집, Ubuntu RD-VLA 로컬 추론과 Mac–GPU 서버 원격 추론 코드를 관리합니다. 기존 저장소 URL과 `main` 이력은 유지합니다.

## 구성

- 루트: 기존 Mac 작업 폴더의 `check_environment.py`, `d455_rgb.py`, 의존성 입력·lock, 합성 검사.
- [`sdk/`](sdk/): macOS D455 RGB용 SDK 빌드 도구와 패치. 바이너리 wheel은 공개하지 않습니다.
- [`inference/`](inference/README.md): **Ubuntu RD-VLA 로컬 추론**(`so101_local.py`, 외부 서버 없음), SmolVLA·RD-VLA·AnchorMix-VLA·pi-0.5 원격 추론, 배포 descriptor·회귀 검사.
- [`ubuntu/`](ubuntu/README.md): 기존 Ubuntu 22.04/24.04 x86_64 수집 구성. **Ubuntu 설치 명령은 이제 `ubuntu/`에서 실행합니다.**

## Ubuntu에서 새로 설치

```bash
git clone https://github.com/Highsky7/SO101-Data-Collection-Ubuntu.git
cd SO101-Data-Collection-Ubuntu/ubuntu
bash setup_ubuntu.sh
```

시스템 사전 요구사항과 전체 수집 절차는 [Ubuntu README](ubuntu/README.md) 및 [한국어 수집 가이드](ubuntu/SO101_데이터수집_가이드.md)를 확인하십시오. 기존 환경·데이터를 덮어쓰지 않습니다.

RD-VLA를 로봇과 같은 Ubuntu PC에서 실행하려면 [로컬 추론 절차](inference/README.md#ubuntu-rd-vla-로컬-추론--외부-서버-없음)를 따르십시오. `prepare → smoke → preflight → run`을 제공하며, 실제 가중치·고정 native 소스/Python 환경·보정 파일·승인된 NVIDIA GPU는 별도 필요합니다. 자동 학습·다운로드·로봇 실행은 하지 않습니다.

## Mac의 기존 환경에서 사용

기존 `SO101-Data-Collection` 작업 폴더의 소스 위치, `.venv` 링크와 로컬 데이터 경로는 변경하지 않습니다. 해당 폴더에서:

```bash
source .venv/bin/activate
export HF_LEROBOT_HOME="$PWD/data"
export HF_LEROBOT_CALIBRATION="$PWD/calibration"
python check_environment.py
python -m unittest test_d455_rgb -v
python -m unittest discover -s inference -p 'test_so101_*.py' -v
python inference/so101_remote.py --help
```

위 검사는 합성 데이터·임시 파일만 사용하며 실제 카메라·시리얼 포트를 열거나 로봇을 구동하지 않습니다. 카메라 확인·수집·`run` 명령은 이 목록에 포함하지 않습니다.

Mac의 `requirements.in`과 `requirements.lock`은 현재 환경의 기록입니다. `wheels/`에 있는 로컬 빌드 SDK wheel을 참조하므로 **공개 저장소 clone만으로 Mac 환경을 그대로 설치할 수는 없습니다.** [`sdk/build_macos_d455.py`](sdk/build_macos_d455.py)의 고정 revision·상위 wheel 해시와 [`sdk/macos-d455.patch`](sdk/macos-d455.patch)를 함께 보존합니다. 재빌드 wheel의 바이트·해시가 원래 lock과 같다고 보장하지 않으며, 의존성 재생성은 별도 유지보수 작업입니다. 이번 Git 연결은 기존 환경이나 lock을 갱신하지 않습니다.

### SDK 빌드 소스 준비 — 필요할 때만

macOS arm64·Python 3.12·Git·Apple command-line tools(`xcrun`, `codesign`)가 필요합니다. wheel의 배포 대상은 macOS 15.0 이상이며, 이 패치는 motion sensor를 지원하지 않습니다. Python binding과 libusb 바이너리는 고정된 상위 wheel을 유지합니다.

아래는 **기존 SDK 환경이 없는 경우**의 별도 준비 예시입니다. 다운로드·도구 설치·컴파일은 이번 Git 동기화에서 실행하지 않습니다.

```bash
SDK_HOME="$HOME/.local/share/so101-data-collection"
mkdir -p "$SDK_HOME"
git clone https://github.com/IntelRealSense/librealsense.git "$SDK_HOME/librealsense-v2.56.5"
git -C "$SDK_HOME/librealsense-v2.56.5" checkout 38a41441971387197193ad3aeae3cefe6a11f2cb
git clone https://github.com/libusb/libusb.git "$SDK_HOME/libusb-v1.0.27"
git -C "$SDK_HOME/libusb-v1.0.27" checkout d52e355daa09f17ce64819122cb067b8a2ee0d4b
uv venv --python 3.12 "$SDK_HOME/sdk-build-tools"
uv pip install --python "$SDK_HOME/sdk-build-tools/bin/python" cmake ninja
# 프로젝트 루트, Python 3.12에서 실행
python sdk/build_macos_d455.py
```

빌더는 상위 wheel을 다운로드해 고정 해시를 확인하고 패치를 적용한 뒤 `wheels/`에 결과를 생성합니다. 기존 결과는 덮어쓰지 않습니다. 빌드 도구 버전·로컬 경로·ZIP timestamp까지 재현하는 빌드는 아니므로, 이 절차도 원래 lock의 wheel 해시 일치를 보장하지 않습니다.

## 원격 추론

[공개 추론 안내](inference/README.md)에 CLI, 배포·연결 순서와 하드웨어 안전 경계를 정리했습니다. 모델 가중치, 원래 학습 source/config, 학습 Python 환경, GPU와 보정 파일은 별도로 필요합니다. 이 저장소는 학습 실행이나 모델 배포 완료를 의미하지 않습니다.

## 공개하지 않는 항목

`.gitignore`로 실제 `data/`, `calibration/`, `logs/`, `backups/`, `training/`, 가상환경, SDK wheel, `inference/runtime/`, 검증 산출물, 인증정보와 모델 가중치를 제외합니다. 서버·실험별 운영 내역이 있는 루트의 `SO101_데이터수집_가이드.md`와 `SO101_추론_가이드.md`도 로컬에 그대로 보존하며 공개하지 않습니다.

로봇 구동은 작업영역·전원·배선·즉시 전원 차단 수단을 확인한 사용자가 직접 실행해야 합니다. 합성 검사 통과는 실물 안전성·실시간 성능·모델 성능 검증이 아닙니다.

## 이후 코드 변경 push

현재 작업 폴더의 `main`은 기존 GitHub `origin/main`을 추적합니다. 자동 게시나 예약 push는 설정하지 않습니다.

```bash
git status --short
# 실제 변경한 코드 경로만 선택하는 예시
git add -- inference/so101_remote.py
git diff --cached
git commit -m "Update SO101 remote inference"
git push
```

게시 전 선택한 diff에 비밀값·개인 데이터가 없는지 확인하십시오. 기존 Ubuntu GitHub Actions는 `ubuntu/`에서 실행되며, 같은 환경에서 기존 추론 경계 회귀 검사도 수행합니다. 실행 상태는 [Actions](https://github.com/Highsky7/SO101-Data-Collection-Ubuntu/actions)에서 확인합니다.
