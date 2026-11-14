# SO101 로컬·원격 추론

`so101_remote.py`는 LeRobot 0.7.0의 native remote server와 rollout을 사용합니다. `so101_adapters.py`는 RD-VLA·AnchorMix-VLA·pi-0.5를 원래 학습 Python/source 환경의 별도 worker에서 실행합니다. 학습 환경이나 체크포인트를 추론 환경으로 덮어쓰지 않습니다.

## Ubuntu RD-VLA 로컬 추론 — 외부 서버 없음

`so101_local.py`는 **로봇·카메라가 연결된 Ubuntu PC에서** RD-VLA 체크포인트를 로드합니다. 수집 환경의 Python과 기존 RD-VLA native Python을 같은 PC의 pipe로 연결하며 TCP 서버·SSH·클라우드 API를 사용하지 않습니다. 두 환경의 Torch/Transformers를 섞거나 학습 환경을 수정하지 않습니다.

필수 준비:

- [Ubuntu 설치](../ubuntu/README.md)로 만든 `ubuntu/.venv`의 LeRobot 0.7.0. 로컬 경로에는 `requirements-server.txt`의 SmolVLA/원격 서버용 Transformers overlay가 필요 없습니다.
- Ubuntu에서 실행 가능한 **기존 학습과 동일한 RD-VLA native Python 환경**. 이 Python에는 원본 RD-VLA의 Torch·torchvision·Transformers·PEFT·timm 의존성이 있어야 합니다. Mac/다른 서버의 `.venv`를 경로만 바꿔 복사하거나 최신 버전으로 일괄 교체하지 마십시오.
- 완성된 checkpoint 전체(`policy/`, `training_state/`, `progress.json`, manifest), 원본 `config.json`과 선언된 compatibility config, 고정된 `source/` 전체. 가중치만 복사하면 무결성/학습 계보 검사를 통과하지 못합니다.
- 승인된 로컬 NVIDIA GPU, 기존 follower 보정 파일, 수집 때와 같은 두 RGB 카메라. 가중치·native 소스·환경은 이 공개 저장소에 포함하지 않습니다. 자동 다운로드·학습은 없습니다.

배치 폴더의 구조를 유지합니다. 원본 config·source·checkpoint 내용은 편집하지 않습니다.

```text
<LOCAL_ROOT>/
  config.json
  source/
    rdvla_adapter.py
    training_state.py
    RD-VLA/
    ... 원본 config에 고정된 나머지 소스
  runs/rdvla/checkpoints/epoch-10/
    ... 원본 committed checkpoint 전체
```

프로젝트 **루트**에서 아래 변수에 실제 로컬 절대 경로를 지정합니다. `LOCAL_ROOT`는 위 배치 폴더이며 native Python 경로는 아래 runtime JSON에 지정합니다.

```bash
PY="$PWD/ubuntu/.venv/bin/python"
LOCAL_ROOT=/absolute/local/rdvla-run
mkdir -p inference/runtime
DEPLOYMENT="$PWD/inference/runtime/rdvla-local.deployment.json"
nvidia-smi --query-gpu=index,uuid,name,memory.free --format=csv
```

`inference/runtime/rdvla-runtime.json`을 직접 만듭니다. 키는 정확히 `root`, `python`, `gpu_uuid` 세 개이며, GPU UUID는 위 출력에서 **사용 승인된 GPU**를 선택합니다. 다음 값은 형식 설명일 뿐 실제 설정이 아닙니다.

```json
{
  "root": "/absolute/local/rdvla-run",
  "python": "/absolute/local/rdvla-native/bin/python",
  "gpu_uuid": "GPU-<approved-local-device-UUID>"
}
```

```bash
# 1. CPU 파일/계보 검증과 descriptor 생성. 기존 output은 덮어쓰지 않음.
"$PY" inference/so101_local.py prepare \
  --training-config "$LOCAL_ROOT/config.json" \
  --checkpoint "$LOCAL_ROOT/runs/rdvla/checkpoints/epoch-10" \
  --runtime-config inference/runtime/rdvla-runtime.json --output "$DEPLOYMENT"

# 2. 실제 로컬 가중치 + 합성 관측. GPU 사용, 로봇/카메라 연결 없음.
"$PY" inference/so101_local.py smoke --deployment-file "$DEPLOYMENT"
```

기존 학습 경로·Python·GPU를 그대로 사용하는 동일 호스트라면 `prepare`의 `--runtime-config`를 생략할 수 있습니다. relocation 설정도 descriptor의 artifact 해시에 포함되며 원본 학습 identity·source hash·checkpoint manifest 검사는 유지됩니다. GPU는 UUID로 고정하고 다른 GPU/CPU로 자동 대체하지 않습니다. 기본적으로 점유 GPU를 거부하며, 공유가 승인된 경우에만 `--allow-shared-gpu --memory-limit-gib <승인된_GiB>`를 `smoke`/`run`에 함께 지정합니다(추가 여유 4 GiB 필요). `smoke --device cpu`는 GPU 없는 실제 가중치 검사에만 사용하며 매우 느릴 수 있습니다.

카메라 JSON 계약은 아래 [클라이언트 입력 계약](#클라이언트-입력-계약)과 같습니다. `CAMERAS_JSON`, `ROBOT_PORT`, `ROBOT_ID`, `TASK`에 실제 수집 설정을 지정한 뒤 다음 순서로 진행합니다.

```bash
# 3. 설정·artifact 검사만 실행. CUDA/카메라/시리얼 접근 없음.
"$PY" inference/so101_local.py preflight --deployment-file "$DEPLOYMENT" \
  --robot-port "$ROBOT_PORT" --robot-id "$ROBOT_ID" --cameras "$CAMERAS_JSON" \
  --task "$TASK" --duration 10

# 4. 사용자가 작업영역·고정·배선·즉시 전원 차단 수단 확인 후 직접 실행.
# 실제 로봇이 움직입니다.
"$PY" inference/so101_local.py run --deployment-file "$DEPLOYMENT" \
  --robot-port "$ROBOT_PORT" --robot-id "$ROBOT_ID" --cameras "$CAMERAS_JSON" \
  --task "$TASK" --duration 10 --confirm-robot-motion
```

보정 기본 경로는 `ubuntu/calibration/robots/so_follower/<ROBOT_ID>.json`입니다. `HF_LEROBOT_CALIBRATION`이 있으면 해당 root를 따르며 `--calibration-dir`로 기존 보정 디렉터리를 명시할 수도 있습니다. 자동 재보정하지 않고 모터 보정 불일치는 중단합니다.

로컬 제어는 동기식 CHUNK입니다. native 전처리·학습 통계·50×6 prediction은 그대로 유지하고 `--execution-horizon 1..50`으로 실행할 앞부분만 고릅니다(기본 50). chunk 사이 추론 중에는 새 action을 보내지 않으므로 30 Hz 연속 제어 성능을 보장하지 않습니다. `--fps`는 재생 속도이며 기본 30입니다. `--duration`은 추론 시간을 포함한 양의 유한 초이고, 만료 후 action을 보내지 않습니다. `--max-observation-age`(기본 5초)를 넘긴 관측·비유한 action은 거부합니다.

`--max-relative-target` 기본 5는 native follower의 **한 명령당** 상대 목표 제한입니다(몸체 degree, gripper 0–100 단위). 출력 목표가 제한될 수 있으며 충돌 방지·비상정지가 아닙니다. 정상 종료·예외·Ctrl+C·제어 중 SIGTERM에서 연결된 카메라/시리얼을 정리하고 torque 해제를 시도합니다. 전원 차단 시 팔이 낙하하지 않게 지지하십시오. 실제 GPU/실물 성능은 별도 검증이 필요합니다.

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
