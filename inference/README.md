# SO101 원격 추론

`so101_remote.py`는 LeRobot 0.7.0의 native remote server와 rollout을 사용합니다. `so101_adapters.py`는 RD-VLA·AnchorMix-VLA·pi-0.5를 원래 학습 Python/source 환경의 별도 worker에서 실행합니다. 학습 환경이나 체크포인트를 추론 환경으로 덮어쓰지 않습니다.

## 준비물과 공개 범위

- Mac: 기존 수집 가상환경, LeRobot 0.7.0, 기존 follower 보정 파일, 실제 top/wrist RGB 카메라 설정.
- GPU 서버: 승인된 GPU 자원, 로컬 체크포인트와 원래 학습 source/config/Python 환경. 이 저장소에는 모델 가중치·학습 환경·비공개 배포 descriptor가 없습니다.
- 클라이언트 추론 의존성은 [`requirements-remote.txt`](requirements-remote.txt), 서버 overlay 의존성은 [`requirements-server.txt`](requirements-server.txt)로 분리합니다. 기존 학습 환경에 직접 설치하지 않습니다.
- Mac 클라이언트의 분리 설치 예시: `uv pip install --python .venv/bin/python --target inference/runtime/deps -r inference/requirements-remote.txt`. `runtime/deps`와 `runtime/vendor`는 존재할 때 import 경로에 추가됩니다. 서버의 LeRobot 0.7.0 및 모델별 native 환경은 별도로 준비해야 합니다.
- SmolVLA의 기본 GPU UUID·체크포인트 경로·epoch별 해시는 현재 배치에 고정되어 있습니다. 임의 가중치로 대체하거나 해시 검사를 생략하는 범용 모델 서버가 아닙니다. 다른 배치로 옮길 때 이 설정과 GPU 사용 권한을 먼저 검토하십시오.

## 명령 확인 — 하드웨어 접근 없음

프로젝트 루트에서 실행합니다.

```bash
python inference/so101_remote.py --help
python inference/so101_remote.py prepare --help
python inference/so101_remote.py server --help
python inference/so101_remote.py preflight --help
python inference/so101_remote.py smoke --help
python inference/so101_remote.py run --help
```

| 명령 | 동작과 접근 범위 |
| --- | --- |
| `prepare` | 완성된 checkpoint·원래 학습 source/config를 검증하고 덮어쓰기 없이 descriptor 생성. GPU·로봇 접근 없음. |
| `server` | 지정 GPU 사용 상태 확인, 고정 모델 로딩·합성 warm-up, loopback TCP 서버 시작. GPU 사용. 로봇 접근 없음. |
| `smoke` | 실행 중인 서버에 합성 관측 전송. 로봇·카메라 접근 없음. 서버 GPU 추론 발생. |
| `preflight` | 보정 파일·카메라 JSON·native rollout 설정 파싱. 서버·카메라·시리얼 연결 없음. |
| `run` | 실제 카메라·시리얼 연결 및 follower 구동. 사용자 안전 확인 필요. |

## 연결 순서

아래 값은 사용자가 이미 보유한 실제 절대 경로와 SSH 별칭으로 지정합니다. 문서의 예시를 자동 실행하는 설정은 아닙니다.

1. RD-VLA·AnchorMix-VLA·pi-0.5는 서버에서 `prepare`로 **완성된 committed checkpoint 한 개**를 고정합니다. `--model`, `--training-config`, `--checkpoint`, `--output`을 명시합니다. 생성한 descriptor를 Mac에 별도로 복사하고 server/client 양쪽에 같은 파일 내용을 사용합니다. `latest`를 실행 중 동적으로 따라가지 않습니다.
2. 서버에서 해당 모델의 `server` 명령을 실행합니다. 예: `python inference/so101_remote.py server --model rdvla --deployment-file "$DEPLOYMENT_FILE" --port 7447`. SmolVLA는 descriptor 대신 `--epoch 10` 또는 `--epoch 20`과 검증된 `--checkpoint`를 선택합니다.
3. Mac에서 SSH 터널을 연결합니다: `ssh -N -o ExitOnForwardFailure=yes -L 127.0.0.1:7447:127.0.0.1:7447 "$REMOTE_HOST"`. 서버는 `127.0.0.1`에만 bind하며 포트를 공개하지 않습니다.
4. Mac에서 서버와 같은 모델·descriptor/epoch를 사용해 `smoke`를 실행합니다. `--endpoint tcp/127.0.0.1:7447`을 사용합니다.
5. 실제 포트·보정 ID·task·카메라 JSON을 지정해 `preflight`합니다. 같은 설정을 유지하고, 안전 확인 후에만 짧은 `run`으로 진행합니다.

`server`는 기본적으로 사용 중인 GPU를 거부하고 다른 프로세스를 중단하지 않습니다. 공유 사용은 별도 승인과 `--allow-shared-gpu --memory-limit-gib`의 명시적 메모리 예산이 필요합니다. 다른 GPU로 자동 fallback하지 않습니다.

## 클라이언트 입력 계약

카메라 JSON에는 서로 다른 실제 OpenCV 장치의 `top`과 `wrist`가 필요합니다. 각 스트림은 `type="opencv"`, RGB, 640×480, 30 FPS여야 합니다. 예시 인덱스를 실제 장치 확인 없이 사용하지 마십시오. 기존 수집용 D455 ZMQ 입력이 이 원격 추론의 OpenCV 카메라 계약을 자동 충족하는 것은 아닙니다.

`preflight`와 `run`은 `--robot-port`, `--robot-id`, `--cameras`, `--task`, `--duration`을 사용합니다. 보정 파일은 프로젝트의 `calibration/robots/so_follower/<robot-id>.json`에 있어야 합니다. `--duration 0`은 무제한이므로 최초 확인에는 사용하지 마십시오. 서버와 클라이언트의 `--fps`를 일치시키고, 학습 지시문·관절 단위·카메라 순서를 유지합니다.

`run`은 실제 로봇을 움직입니다. Follower 고정, 관절 작업영역, 배선·전원, 즉시 전원 차단 수단을 확인하십시오. 소프트웨어 종료는 비상정지나 충돌 방지 장치가 아닙니다.

## 기존 하드웨어 없는 회귀 검사

```bash
python -m unittest discover -s inference -p 'test_so101_*.py' -v
```

입력 경계, descriptor/checkpoint binding, GPU admission의 mock 기반 분기와 worker protocol 등을 검사합니다. 실제 GPU·체크포인트 모델·로봇·카메라는 실행하지 않습니다.

`verify_training_equivalence.py`는 기존 배치 경로·학습 환경·실제 학습 frame을 요구하는 별도 CPU 비교 도구입니다. 일반 clone에서 실행되는 단위 검사가 아니며, 공개 저장소에는 그 결과·데이터를 포함하지 않습니다. 새 추론 성능이나 네 모델의 실제 동등성을 이 문서만으로 주장하지 않습니다.
