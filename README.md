# Areumii LeRobot plugins

현재 노트북의 녹화·실로봇 구동 파일을 그대로 관리하는 저장소다. 데탑에서는
Hugging Face 데이터셋으로 SmolVLA / π₀를 학습하고 모델 추론을 개발한다.
CAN 저장소나 IsaacTeleop 전체를 이 저장소에 포함하지 않는다.

## 파일과 실행 환경

| 파일 | 역할 |
| --- | --- |
| `lerobot_robot_areumii/` | UDP follower, 고정 카메라, 16D state/action, `areumii-record` 진입점 |
| `lerobot_teleoperator_areumii_xr/` | IsaacTeleop/OpenXR 양손 컨트롤러 → IK → 관절 목표값 |
| `train_night.sh` | v2 green → blue → both SmolVLA 학습, 각 30,000 steps |
| `train_v3_compare.sh` | v3 → v3-single10 SmolVLA 학습, 각 30,000 steps |
| `eval_areumii_smolvla.py` | 공식 `lerobot-rollout` 실행 및 수동 성공률 CSV 기록 |
| `areumii_rollout_prompt.py` | episodic rollout에서 매 에피소드 직전 task 입력 |
| `preflight_check.py` | 노트북 플러그인/XR import 확인; 로봇 connect는 호출하지 않음 |
| `requirements-desktop.txt` | 데탑 학습용 LeRobot extras; IsaacTeleop 제외 |

2026-09-28에 기존 `~/projects/env_lerobot`에서 확인한 환경:

| 항목 | 버전 |
| --- | --- |
| Python | 3.12.13 |
| LeRobot | 0.6.1 (`site-packages` 설치; editable 아님) |
| PyTorch / torchvision | 2.11.0 / 0.26.0 |
| transformers | 5.5.4 |
| NumPy / SciPy | 2.2.6 / 1.18.0 |
| placo | 0.9.15 |
| isaacteleop | 1.5.123rc1 |
| 두 Areumii 플러그인 | 0.1.0, 이 폴더를 editable 설치로 참조 |

설치된 LeRobot의 Python 요구사항은 `>=3.12`다. 두 플러그인의 메타데이터도
이에 맞췄다. 플러그인은 `lerobot>=0.6.1,<0.7`을 요구하지만 데탑 시작 버전은
검증 기준과 같은 `0.6.1`로 고정한다. 전체 전이 의존성 lock은 아니므로 데탑
설치 후 실제 버전을 다시 기록한다. 인접 `~/projects/lerobot` 소스 체크아웃은
설치본과 별개이며, 그 Git commit을 설치본의 commit이라고 간주하지 않는다.

재점검에서 기존 노트북 venv의 `python -m pip check`는 다음 누락을 보고했다:
`generate-parameter-library-py 0.7.3 requires typeguard, which is not installed.`
플러그인 import 성공과 전체 환경의 의존성 일관성은 별개다. 기존 녹화 환경을
자동 변경하지 않았으며, 이 누락 때문에 전체 venv 검증이 통과했다고 볼 수 없다.
데탑은 새 venv에서 설치하고 `pip check`를 별도로 통과시킨다.

## 노트북: 기존 환경 유지 및 녹화

이미 editable 설치되어 있으므로 소스 수정은 바로 반영된다. 노트북 환경을
데탑 requirements로 재설치할 필요는 없다. 새로 플러그인을 등록해야 할 때만:

```bash
cd ~/projects/areumii_lerobot_plugins
source ~/projects/env_lerobot/bin/activate
python -m pip install --no-deps -e ./lerobot_robot_areumii
python -m pip install --no-deps -e ./lerobot_teleoperator_areumii_xr
```

녹화에는 LeRobot의 `core_scripts`, IK용 `kinematics` 의존성(placo), NumPy,
SciPy, 기존 IsaacTeleop/OpenXR/CloudXR 환경이 필요하다. `isaac_teleop`은
인접 `IsaacLab_teleop/real_teleop`의 모듈이고 `isaacteleop`은 별도 설치 패키지다.
XR 플러그인의 pyproject만 설치한다고 VR 런타임까지 준비되는 것은 아니다.

기본 경로는 기존 노트북과 같다. 다른 경로에서는 선택적으로 설정한다:

```bash
cp .env.example .env
# .env의 경로를 편집한 뒤 명시적으로 로드한다. 자동 로드하지 않는다.
set -a
source .env
set +a
export PYTHONPATH="${AREUMII_TELEOP_PATH:-$HOME/projects/IsaacLab_teleop/real_teleop}${PYTHONPATH:+:$PYTHONPATH}"
python preflight_check.py
```

`AREUMII_TELEOP_PATH` 기본값은 `$HOME/projects/IsaacLab_teleop/real_teleop`,
`AREUMII_URDF` 기본값은 그 폴더의 `areumii_c1_kinematics.urdf`다.
`areumii-record`는 자체적으로 teleop 경로를 추가한다. preflight와 일반
rollout은 플러그인 자동 탐색 시 XR도 import할 수 있어 노트북에서는 위
`PYTHONPATH` 설정을 유지한다.

실행 전 CAN 담당 저장소에서 기존 CAN 제어기와 UDP/SHM 브리지를 준비한다.
다음 명령은 운영자가 직접 실행하는 예시이며, 저장소 정리 과정에서는 실행하지 않았다.

명령 전송 없는 녹화 확인(카메라·VR·브리지 피드백은 여전히 필요):

```bash
areumii-record \
  --repo-id YOUR_HF_NAME/areumii-dryrun \
  --num-episodes 1 \
  --push-to-hub false \
  --display-data true \
  --enable-command false
```

실제 녹화(로봇 명령 전송 및 종료 후 데이터셋 Hub 업로드):

```bash
areumii-record \
  --repo-id YOUR_HF_NAME/areumii-pickplace \
  --num-episodes 50 \
  --push-to-hub true \
  --display-data true \
  --enable-command true
```

고정값은 `record_areumii.py`의 robot/teleop ID, UDP 5005/5006,
**episode_time_s=3000**, reset_time_s=5, 30 FPS, video=true다.
3000초는 의도적인 긴 상한이며 **오른쪽 화살표로 에피소드를 수동 종료**한다.
왼쪽 화살표는 재녹화, Esc는 녹화 종료다. reset 단계에서도 오른쪽 화살표로
일찍 끝낼 수 있다. reset hook은 기존 control target에서 준비 자세로 양팔을
이동하며, `enable-command=false`이면 reset 명령도 보내지 않는다.
현재 task는 `Place the green can on the upper shelf, then place the blue can on the lower shelf.`다.

### 고정 카메라 설정 — 변경하지 않음

| 키 | 장치 | 캡처 해상도 | FPS / FourCC | 데이터셋에 전달되는 이미지 |
| --- | --- | --- | --- | --- |
| head | index 8 (`/dev/video8`) | 640×480 | 30 / UYVY | 640×480 |
| left_wrist | index 10 (`/dev/video10`) | 1280×720 | 30 / MJPG | 320×180, 180도 회전 |
| right_wrist | index 12 (`/dev/video12`) | 1280×720 | 30 / MJPG | 320×180, 180도 회전 |

모두 OpenCV V4L2다. 실제 코드는 정수 device index를 사용하며 `/dev/v4l/by-id`
경로가 아니다. 재부팅 후 번호가 바뀌면 장치 연결 상태부터 확인한다.

## SmolVLA 학습

```bash
source ~/projects/env_lerobot/bin/activate
cd ~/projects/areumii_lerobot_plugins
hf auth login
wandb login
bash train_night.sh
# 또는 별도의 비교 실행:
bash train_v3_compare.sh
```

**두 스크립트 모두 `--policy.push_to_hub=true`, `--wandb.enable=true`다.**
실행하면 설정된 `1ys1/...` 모델 저장소로 업로드하고 W&B로 기록한다.
`wandb.disable_artifact=true`는 모델 Hub 업로드를 끄는 옵션이 아니다.
다른 계정에서 실행하기 전 스크립트의 dataset/model repo ID를 검토한다.
로컬 전용 학습은 `policy.push_to_hub=false`, `wandb.enable=false`로 바꾼다.
기존 자동 업로드 동작은 변경하지 않았으며 실행 시작 시 안내를 출력한다.

두 스크립트는 `lerobot/smolvla_base`, batch 8, CUDA를 사용한다.
`train_night.sh`는 10,000 step마다, v3 스크립트는 5,000 step마다 저장한다.
출력은 이 저장소의 `outputs/train/`이다. 기존 출력 폴더와 충돌하면 새
`output_dir`/`job_name`을 사용한다. venv 선택 우선순위는 `LEROBOT_VENV` →
활성 `VIRTUAL_ENV` → `$HOME/projects/env_lerobot`이며, 작업 경로는 스크립트 위치다.

학습 카메라 이름은 head → camera1, left_wrist → camera2, right_wrist → camera3로
매핑한다. rollout에도 같은 매핑을 전달해야 한다.

## 노트북: 실로봇 rollout

`eval_areumii_smolvla.py`의 기본 train-dir/task는 과거 v1용이므로 지금 학습한
모델에는 `--model`과 `--task`를 명시한다. 또는 `--train-dir`과 `--step`을 쓴다.

```bash
export AREUMII_MODEL=outputs/train/areumii-smolvla-real-pickplace-v3/checkpoints/030000/pretrained_model
export AREUMII_RENAME_MAP='{"observation.images.head":"observation.images.camera1","observation.images.left_wrist":"observation.images.camera2","observation.images.right_wrist":"observation.images.camera3"}'
python eval_areumii_smolvla.py \
  --model "$AREUMII_MODEL" \
  --rename-map "$AREUMII_RENAME_MAP" \
  --task "Place the green can on the upper shelf, then place the blue can on the lower shelf." \
  --mode smoke --enable-command false
```

실제 episodic 평가 명령은 다음과 같다. enable-command=true는 로봇을 움직인다.

```bash
python eval_areumii_smolvla.py \
  --model "$AREUMII_MODEL" --rename-map "$AREUMII_RENAME_MAP" \
  --task "Place the green can on the upper shelf, then place the blue can on the lower shelf." \
  --mode eval --enable-command true \
  --repo-id YOUR_HF_NAME/areumii-eval --num-episodes 5 --push-to-hub false
```

평가 기본값은 episode 60초, reset 12초이며 녹화 런처의 3000초와 별개다.
평가 후 성공/실패 라벨은 `results/`에 CSV로 저장한다.
새 `--rename-map`은 지정한 경우에만 전달하므로 기존 호출의 기본 동작은 유지한다.

매 에피소드 task를 직접 입력하려면:

```bash
python areumii_rollout_prompt.py \
  --policy.path="$AREUMII_MODEL" --rename_map="$AREUMII_RENAME_MAP" \
  --robot.type=areumii --robot.id=areumii_c1 --robot.enable_command=true \
  --strategy.type=episodic --inference.type=sync --device=cuda \
  --dataset.repo_id=YOUR_HF_NAME/areumii-prompt-eval \
  --dataset.num_episodes=5 --dataset.episode_time_s=60 \
  --dataset.reset_time_s=12 --dataset.push_to_hub=false
```

이 스크립트는 LeRobot 내부 `_policy_loop`와 `_engine._task`를 사용하므로
LeRobot 업그레이드 시 호환성 확인이 필요하다.

**현재 두 rollout 경로는 데탑 원격 추론을 지원하지 않는다.** `sync`는 로컬
동기 추론, `rtc`는 같은 프로세스의 백그라운드 추론이다. 모델을 Hub에서
다운로드하는 것은 원격 추론이 아니다. 노트북 카메라/state를 데탑으로 보내고
action을 돌려받는 클라이언트·서버 연결은 구현되어 있지 않다. UDP 5005/5006도
정책 서버용이 아니며 C++ 브리지의 주소는 127.0.0.1로 고정되어 있다.
데탑에서 학습한 모델을 노트북에 전달해 로컬 rollout하거나, 별도로 원격 추론
전송 계층을 개발해야 한다. 데탑 학습/모델 추론에는 IsaacTeleop이 필요하지 않다.

## 데탑: clone, venv, Hugging Face 데이터셋 학습

Linux/NVIDIA 드라이버와 Python 3.12가 준비된 24GB GPU PC 기준이다.
GitHub 주소는 저장소를 만든 뒤 아래 placeholder를 바꾼다.

```bash
git clone https://github.com/YOUR_GITHUB_NAME/areumii_lerobot_plugins.git
cd areumii_lerobot_plugins
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-desktop.txt
python -m pip check
python -c 'import sys, torch, importlib.metadata as m; print(sys.version); print("LeRobot", m.version("lerobot")); print("CUDA", torch.version.cuda, torch.cuda.is_available())'
hf auth login
```

CUDA가 false라면 학습 전 드라이버와 PyTorch CUDA 설치를 맞춘다.
순수 데이터셋 학습에는 두 플러그인 모두 필요 없다. 로봇 API를 개발할 때만
`python -m pip install --no-deps -e ./lerobot_robot_areumii`로 robot 플러그인을
설치한다. **데탑에는 XR 플러그인과 IsaacTeleop을 설치하지 않는다.**
`preflight_check.py`는 XR까지 검사하므로 데탑 설치 검증 명령으로 쓰지 않는다.
토큰은 `hf auth login`으로 관리한다. 비공개 데이터셋 접근 권한 및 모델/tokenizer의
Hub 사용 조건이 있으면 먼저 승인한다. 데이터는 `dataset.repo_id`로 다운로드한다.

24GB에서는 π₀ 전체 파라미터 학습이 가능하다고 가정하지 않는다. 아래는 batch 1,
bfloat16, gradient checkpointing, action expert만 학습하는 **초기 메모리 확인용**
설정이다. 3개 카메라를 사용하는 이 데이터로 실제 VRAM 검증은 아직 하지 않았다.

```bash
lerobot-train \
  --dataset.repo_id=1ys1/areumii-real-pickplace-v3 \
  --policy.type=pi0 --policy.pretrained_path=lerobot/pi0_base \
  --policy.device=cuda --policy.dtype=bfloat16 \
  --policy.gradient_checkpointing=true --policy.compile_model=false \
  --policy.train_expert_only=true \
  --policy.push_to_hub=false --wandb.enable=false \
  --batch_size=1 --num_workers=2 --steps=100 --save_freq=100 \
  --output_dir=outputs/train/pi0-areumii-smoke --job_name=pi0-areumii-smoke
```

`policy.type` + `pretrained_path`는 데이터셋에서 입력/출력 feature를 구성하므로
이 예시에서는 원래 head/left_wrist/right_wrist 키와 16D action을 유지한다.
SmolVLA의 camera1/2/3 매핑을 여기에 그대로 붙이지 않는다. π₀ 내부 이미지
리사이즈는 모델 전처리이며 실제 카메라 설정을 바꾸지 않는다.
메모리가 충분하면 새 output_dir로 steps/save_freq를 늘려 본 학습을 시작한다.
부족하면 PEFT/LoRA 실험을 별도로 진행한다(`peft` extra는 이미 포함).
full fine-tuning, expert-only, LoRA는 서로 다른 학습 설정이며 결과를 구분한다.

설정 근거: [공식 π₀ 문서](https://huggingface.co/docs/lerobot/pi0),
[공식 PEFT 문서](https://huggingface.co/docs/lerobot/peft_training).
설치된 0.6.1의 `PI0Config`와 train CLI 옵션도 대조했다.
학습 완료 후 checkpoint의 `pretrained_model` 디렉터리 전체(가중치, config,
pre/postprocessor, 정규화 통계)를 함께 옮긴다. 가중치 파일 하나만 복사하지 않는다.
Hub 업로드는 위 데탑 예시에서 꺼져 있다. 업로드하려면 본인 `policy.repo_id`와
`policy.push_to_hub=true`를 명시한다.

### 데탑에서 π₀ 본 학습 및 Hub 업로드

100 step 시험 학습이 정상 완료된 뒤 실행한다. 이 설정은 시험과 같은
`train_expert_only=true`, batch size 1로 30,000 step 학습한다.
`DATASET_ID`는 **실제로 학습할 데이터셋 ID**로 바꾼다.

```bash
conda activate areumii-lerobot
cd /실제/areumii_lerobot_plugins/경로
hf auth whoami

DATASET_ID=1ys1/실제데이터셋ID
MODEL_ID=1ys1/areumii-pi0-expert-30k
mkdir -p logs
set -o pipefail

lerobot-train \
  --dataset.repo_id="$DATASET_ID" \
  --policy.type=pi0 --policy.pretrained_path=lerobot/pi0_base \
  --policy.device=cuda --policy.dtype=bfloat16 \
  --policy.gradient_checkpointing=true --policy.compile_model=false \
  --policy.train_expert_only=true \
  --policy.repo_id="$MODEL_ID" --policy.private=true --policy.push_to_hub=true \
  --wandb.enable=false \
  --batch_size=1 --num_workers=2 \
  --steps=30000 --log_freq=100 --save_freq=10000 \
  --output_dir=outputs/train/areumii-pi0-expert-30k \
  --job_name=areumii-pi0-expert-30k \
  2>&1 | tee logs/areumii-pi0-expert-30k.log &&
hf upload "$MODEL_ID" \
  logs/areumii-pi0-expert-30k.log \
  training_logs/areumii-pi0-expert-30k.log
```

100 step마다 학습 지표가 출력되고 로컬 `logs/`에 기록된다.
10,000 step마다 체크포인트가 로컬 `outputs/train/`에 저장된다.
정상 종료 시 최종 모델이 비공개 Hugging Face 모델 저장소에 올라가며,
그 후 텍스트 학습 로그가 같은 저장소의 `training_logs/`에 올라간다.
학습이 실패하면 `&&` 때문에 로그 업로드는 실행되지 않는다.
중간 체크포인트는 로컬에 보관된다.

다른 데이터셋이나 설정으로 다시 학습할 때는 `MODEL_ID`,
`output_dir`, `job_name`, 로그 파일명을 새 이름으로 바꾼다.

## CAN 저장소와 통신 규약

별도 `~/projects/Areum2_can`가 CAN 모터 제어와 공유 메모리를 소유한다.
그 안의 `areumii_udp_shm_bridge.cpp`, `areumii_move_pose.cpp`, `inc/Sharemem.hpp`를
함께 관리하며 여기로 복사하지 않는다. 실제 CAN 저장소 원격 주소와 빌드 환경은
팀원 저장소에서 확인한다.

```text
XR 또는 로컬 policy → Robot 플러그인 → UDP 127.0.0.1:5005
                                      ↓ C++ udp_shm_bridge
                                Sharemem.hpp / CAN 제어기
                                      ↓ UDP 127.0.0.1:5006
                         Robot → 프로세스 내부 snapshot → XR
```

`Sharemem.hpp`는 System V SHM ABI를 정의한다. `Control_Shm<16>`, key `13563267`,
control의 double pos/vel/Kp/Kd/ffTorque와 feedback의 float pos/vel/torque/temp,
원자적 sequence counter를 사용하는 읽기/쓰기 구조다. CAN 제어기와 두 C++ 도구는
같은 헤더/레이아웃을 사용해야 한다. Python은 SHM을 직접 읽지 않는다.
SHM index 0..6은 왼팔, 7은 왼손 gripper, 8..14는 오른팔, 15는 오른손 gripper다.
`areumii_move_pose`는 UDP를 거치지 않고 SHM에 ready/down 목표를 쓰는 별도
실물 구동 도구다. bridge와 동시 control writer로 사용하지 않는다.

### UDP action (5005)

UTF-8/ASCII CSV datagram이며 헤더·JSON·binary 배열이 아니다. Python은 한 tick에
다음 네 패킷을 각각 전송한다. 관절 및 gripper 모터 목표 위치 단위는 rad다.

```text
L,q0,q1,q2,q3,q4,q5,q6
R,q0,q1,q2,q3,q4,q5,q6
GL,left_gripper_motor_rad
GR,right_gripper_motor_rad
```

팔 순서는 shoulder pitch, shoulder roll, shoulder yaw, elbow, wrist roll,
wrist yaw, wrist pitch다. 현재 C++ 양팔 sign 배열은 모두 +1이다(과거 주석보다
실제 값을 기준으로 함). bridge는 팔 목표를 100Hz, 최대 0.60 rad/s로 보간하고
gripper 범위 [-1.25, 0.95] rad 밖의 명령은 거절한다. 이전 별도 safety 블록은
`#if 0`로 비활성 상태다. 이 저장소 정리에서 해당 제어 동작은 수정하지 않았다.

### UDP feedback (5006)

약 100Hz, 위치 32개를 comma로 구분한 문자열이다. 아래 범위는 Python slice 기준이다.

| packet index | 내용 (rad) |
| --- | --- |
| `[0:7]` / `[7:14]` | measured 왼팔 / 오른팔 |
| `[14:21]` / `[21:28]` | control target 왼팔 / 오른팔 |
| `[28]` / `[29]` | measured 왼 / 오른 gripper |
| `[30]` / `[31]` | control target 왼 / 오른 gripper |

데이터셋 `observation.state`는 measured 16D이고 `action`은 target 16D다.
둘 다 **왼팔 7 → 왼 gripper → 오른팔 7 → 오른 gripper** 순서다.
전체 feature 이름은 `shared_state.py`의 `FEATURE_KEYS`가 기준이다.
32D feedback 배열을 16D 학습 action 순서로 직접 취급하면 안 된다.

Robot만 5006을 bind하고 관측 시 snapshot을 갱신한다. XR은 같은 Python 프로세스의
snapshot을 읽어 IK seed와 target 연속성을 유지한다. XR가 포트를 추가 bind하지 않는다.
연결 시 32D 피드백을 요구하고 기본 freshness timeout은 0.20초다.

## Git 관리와 검증

처음 조사 시 `.git`은 없었고 추적된 생성 파일도 없었다. 이 폴더 자체에 Git을
초기화하고 `ys/repository-setup` 브랜치를 준비했다. 아직 commit/remote/push는 없다.
GitHub에서 빈 저장소를 만든 뒤 **본인 원격 URL**을 연결하면 된다.

```bash
git remote add origin https://github.com/YOUR_GITHUB_NAME/areumii_lerobot_plugins.git
git status --short
# 파일 검토 및 커밋 후에만: git push -u origin ys/repository-setup
```

`.gitignore`는 venv, cache/egg-info, 녹화 데이터, 캡처 이미지, checkpoint,
outputs/results, 로그, 토큰, `.env`/로컬 설정을 제외한다. 기존 `outputs/`와 이미지,
egg-info는 디스크에 그대로 있다. 추후 생성 파일을 잘못 추적했다면
`git rm --cached <file>` 또는 디렉터리에 `git rm -r --cached <directory>`를
사용하면 로컬 내용은 보존된다. `.env.example`에는 비밀값을 넣지 않는다.

문법 확인은 Python 파일을 compile/AST 검사하고 `bash -n`으로 shell을 검사한다.
두 독립 `pyproject.toml`이 설치 단위이며 루트의 `pip install -e .`는 지원하지 않는다.
패키지 검증은 임시 venv에 두 wheel을 `--no-deps`로 설치해 메타데이터와 entrypoint를
확인한다. 이는 데탑 전체 의존성 설치나 실제 모델 학습 성공을 의미하지 않는다.
이번 정리에서는 robot connect, UDP 송신, 카메라 연결, VR 실행, 실제 학습 및
Hub 업로드를 실행하지 않았다.

2026-09-28 검증 결과: Python 12개 파일 문법, shell 2개 문법, 두 wheel 빌드 및
임시 venv 설치, console entrypoint 메타데이터, 기존 환경 preflight import가 모두
통과했다. 카메라 설정 AST와 로봇 제어 구현을 수정 전과 비교해 동일함을 확인했고,
녹화 기본값 및 환경변수 override, rollout 매핑 전달과 기존 기본 명령 보존도 확인했다.
Git 포함 후보는 20개 텍스트 파일이며 기존 생성 파일은 제외되었다.

재점검에서는 `areumii-record`, `eval_areumii_smolvla.py`,
`areumii_rollout_prompt.py`, `lerobot-train`, `lerobot-rollout`의 `--help`도 모두
종료 코드 0을 확인했다(Hub offline 설정). 현재 소스로 wheel을 다시 빌드하고
임시 venv에 재설치했다. 기존 노트북 환경의 `pip check`는 위 `typeguard` 누락으로
실패했으며, 실제 녹화·모델 학습·추론 성능은 이번 검증 범위에 포함되지 않는다.
