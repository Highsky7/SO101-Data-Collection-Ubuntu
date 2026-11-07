# SO101 Ubuntu 데이터 수집

공식 LeRobot 0.7.0의 SO-ARM101 수집 흐름을 Ubuntu 22.04/24.04 x86_64에서 사용하도록 구성했습니다. 다른 Ubuntu 버전과 ARM64는 미검증이며, 설치 스크립트와 Linux lock의 대상은 x86_64입니다.

장치 연결·보정·카메라 설정·텔레오퍼레이션·수집·검수·이어찍기는 [전체 한국어 가이드](SO101_데이터수집_가이드.md)를 따르십시오.

## 새 Ubuntu 환경에서 시작

Bash 터미널에서 시스템 도구와 `uv`를 먼저 설치합니다. 프로젝트 설치 스크립트는 시스템 패키지나 장치 권한을 변경하지 않습니다.

```bash
sudo apt update
sudo apt install -y git curl ffmpeg build-essential python3-dev linux-libc-dev
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
```

저장소를 복제하고 프로젝트 전용 환경을 만듭니다.

```bash
git clone https://github.com/Highsky7/SO101-Data-Collection-Ubuntu.git
cd SO101-Data-Collection-Ubuntu
bash setup_ubuntu.sh
```

새 checkout에서만 실행하십시오. Python 3.12 `.venv`를 만들고 CPU 전용 Linux `requirements.lock`을 해시 검증해 설치합니다. 기존 `.venv`는 교체하지 않으며, 설치 실패 시 부분 생성된 환경도 자동 삭제하지 않습니다. `requirements.macos.lock`은 원본 Mac 환경의 참고용이며 Ubuntu 설치에는 사용하지 않습니다.

환경 활성화 후 로봇·카메라 없이 소프트웨어를 검사합니다.

```bash
export PROJECT="$(pwd -P)"
source .venv/bin/activate
export HF_LEROBOT_HOME="$PROJECT/data"
export HF_LEROBOT_CALIBRATION="$PROJECT/calibration"
mkdir -p data calibration logs backups

uv pip check --python "$PROJECT/.venv/bin/python"
python check_environment.py
```

합성 RGB 2개 스트림 × 2 episodes × 12 frames의 저장·새 프로세스 이어쓰기·디코딩·관절/행동 값·timestamp를 확인합니다. 실제 카메라나 시리얼 포트는 열지 않습니다.

## 안전과 공개 범위

- 보정·`lerobot-teleoperate`·`lerobot-record`는 실물 장치에 연결할 수 있습니다. 작업영역·전원·배선·비상 전원 차단 수단을 먼저 확인하십시오.
- 시리얼과 카메라 권한은 가이드의 `dialout`/조건부 `video` 그룹 절차를 따릅니다. `sudo python`이나 장치의 `chmod 666`으로 우회하지 마십시오.
- 코드만 공개합니다. 실제 `data/`, `calibration/`, `logs/`, `backups/`와 인증정보는 Git에서 제외합니다.

## 자동 검증 범위

[GitHub Actions](https://github.com/Highsky7/SO101-Data-Collection-Ubuntu/actions)는 Ubuntu 22.04/24.04에서 CPU 의존성 설치, CLI 도움말 6개, 합성 저장·이어쓰기·읽기, 기존 `.venv` 보호를 검사합니다. 실행별 결과는 해당 로그를 확인하십시오. 대화형 `lerobot-find-port`는 `--help`를 지원하지 않아 자동 실행하지 않습니다.

실제 로봇 구동·보정·촬영, 카메라 동기화와 실측 수집 속도는 자동 검증하지 않습니다.
