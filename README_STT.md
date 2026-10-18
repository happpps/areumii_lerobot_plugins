Areumii 음성 task 입력
기존 텍스트 입력과 LeRobot의 RTC/sync·에피소드 녹화 흐름을 유지하고, 에피소드 시작 전 음성으로 작업을 선택합니다. VLA 재학습은 필요하지 않습니다.
1. 파일 배치
프로젝트 루트에 다음 파일을 함께 둡니다. 기존 두 Python 파일은 먼저 백업한 다음 교체하세요.
- areumii_rollout_prompt.py (수정본)
- eval_areumii_smolvla.py (수정본)
- areumii_speech.py (추가)
- speech_tasks.json (추가)
- requirements-stt.txt (추가)
STT 옵션 없이 실행하면 기존 텍스트 입력을 사용합니다. STT 의존성도 불러오지 않습니다.
2. 설치
평소 사용하는 LeRobot 가상환경을 활성화한 뒤 프로젝트 루트에서 실행합니다.
python -m pip install -r requirements-stt.txt
python -m pip check
Linux에서 PortAudio library not found가 나오면 시스템에 libportaudio2가 필요합니다. 다음 명령으로 설치한 뒤 다시 실행하세요.
sudo apt update
sudo apt install libportaudio2
STT는 faster-whisper의 다국어 small 모델을 CPU/int8로 실행합니다. VLA용 CUDA를 사용하지 않지만 CPU·RAM과 모델 캐시 공간은 사용합니다. 첫 실행은 인터넷으로 모델을 내려받으며, 이후 캐시를 사용합니다. 녹음은 메모리에서 처리하고 음성 파일은 따로 저장하지 않습니다.
3. 실제 학습 task 설정 — 먼저 수정
speech_tasks.json의 task는 반드시 학습에 사용한 정확한 문장으로 설정하세요.
- 보틀: 제공된 문장은 Pick up the pink bottle and place it in the box.입니다. 현재 학습 라벨과 같은지 확인하세요.
- 이클립스: 정확한 학습 문장을 확인할 수 없어 REPLACE_WITH_EXACT_ECLIPSE_TRAINING_TASK로 표시했습니다. 실제 문장으로 교체해야 합니다. 그대로 두면 시작 전에 오류로 종료합니다.
- aliases에는 음성 인식 결과에서 찾을 물체 이름을 넣습니다. 서로 다른 작업에 같은 별칭을 넣지 마세요.
마이크 입력은 단어 기반으로 하나의 작업을 선택합니다. 예: ‘분홍색 병을 박스에 넣어줘’ → 보틀의 task. 두 물체를 함께 언급하거나 ‘말고/취소/정지’ 등의 표현이 인식되면 실행하지 않습니다. 자유로운 지시 해석이나 새로운 작업 생성은 지원하지 않습니다. 인식이 잘못될 수 있으므로 최종 task를 확인하고 y를 입력해야 합니다.
4. 로봇 없이 마이크 테스트
입력 장치 목록:
python areumii_speech.py --speech.list_devices=true
오디오/STT만 테스트 (CAN·카메라·VLA를 시작하지 않음):
python areumii_speech.py \
  --speech.enabled=true \
  --speech.task_map=speech_tasks.json
v → Enter를 누르면 5초간 녹음합니다. ‘분홍색 병을 박스에 넣어줘’라고 말하고, 표시된 task가 맞으면 y를 입력합니다. 이 테스트는 선택 결과만 출력하고 로봇 명령을 보내지 않습니다.
원하는 마이크를 지정하려면 목록에 나온 입력 장치 번호를 추가합니다.
python areumii_speech.py \
  --speech.enabled=true \
  --speech.task_map=speech_tasks.json \
  --speech.input_device=3
3은 예시입니다. 실제 입력 장치 번호로 변경하세요.
5. 기존 추론 명령에 연결
기존 python areumii_rollout_prompt.py ... 명령에 아래 두 옵션만 추가하세요. --strategy.type=episodic이어야 합니다.
--speech.enabled=true
--speech.task_map=speech_tasks.json
예: 기존 SmolVLA RTC 평가 명령에 음성 추가
python areumii_rollout_prompt.py \
  --policy.path=1ys1/areumii-smolvla-doubletask-v2 \
  --robot.type=areumii \
  --robot.id=areumii_c1 \
  --robot.cmd_port=5005 \
  --robot.feedback_port=5006 \
  --robot.enable_command=true \
  --fps=30 \
  --play_sounds=false \
  --interpolation_multiplier=1 \
  --strategy.type=episodic \
  --dataset.repo_id=1ys1/rollout-smol-voice-test-v1 \
  --dataset.num_episodes=10 \
  --dataset.episode_time_s=50 \
  --dataset.reset_time_s=10 \
  --dataset.push_to_hub=true \
  --inference.type=rtc \
  --inference.rtc.execution_horizon=5 \
  --inference.rtc.max_guidance_weight=10.0 \
  --rename_map='{"observation.images.head":"observation.images.camera1","observation.images.left_wrist":"observation.images.camera2","observation.images.right_wrist":"observation.images.camera3"}' \
  --speech.enabled=true \
  --speech.task_map=speech_tasks.json
모델 ID는 평가하려는 모델로, 평가 데이터셋 ID는 아직 사용하지 않은 이름으로 변경하세요. π₀에도 같은 음성 옵션을 사용할 수 있습니다. π₀의 카메라 이름이 head/left_wrist/right_wrist라면 SmolVLA용 rename_map은 제외하세요. 실행 순서는 기존과 같이 CAN 프로그램 → move_pose → 브리지 → 추론입니다.
에피소드 시작 전 입력:
입력	동작
v + Enter	5초 동안 녹음하여 작업 선택
t + Enter	영어 task 문장 직접 입력
Enter만 입력	이전 task 선택 (첫 실행에서는 사용 불가)
q + Enter	종료
확인 화면에서 y + Enter	에피소드 시작
확인 화면에서 Enter만 입력	선택 취소 후 입력 화면으로 돌아가기


선택한 영어 task는 추론 엔진과 녹화 데이터셋의 해당 에피소드 라벨에 함께 전달합니다. STT는 에피소드 시작 전에만 실행합니다. 로봇이 작업하는 동안 음성으로 정지하는 기능은 지원하지 않습니다. 종료하려면 q 또는 Ctrl+C를 사용하세요. 확인 화면에서 Enter를 누르면 선택만 취소합니다.
6. 평가 실행 스크립트에서 사용하기
python eval_areumii_smolvla.py \
  --model 1ys1/areumii-smolvla-doubletask-v2 \
  --mode eval \
  --enable-command true \
  --repo-id 1ys1/eval-smol-voice-test-v1 \
  --inference rtc \
  --rtc-execution-horizon 5 \
  --rename-map '{"observation.images.head":"observation.images.camera1","observation.images.left_wrist":"observation.images.camera2","observation.images.right_wrist":"observation.images.camera3"}' \
  --speech true \
  --speech-task-map speech_tasks.json
음성 모드에서는 같은 폴더에 있는 areumii_rollout_prompt.py를 실행합니다. 기존 성공 판정 CSV는 고정된 task 문장을 기록하므로, 에피소드마다 작업을 바꾸는 음성 모드에서는 잘못된 라벨을 남기지 않도록 출력하지 않습니다. 각 에피소드의 실제 task는 녹화 데이터셋에 저장됩니다. --push-to-hub true를 추가하면 Hub 업로드도 활성화됩니다.
7. 조정 옵션
직접 실행 옵션	기본값	용도
--speech.model	small	다국어 STT 모델 이름 또는 로컬 모델 경로
--speech.language	ko	음성 인식 언어
--speech.seconds	5	녹음 시간 (최대 30초)
--speech.input_device	운영체제 기본 장치	마이크 번호 또는 이름
--speech.cpu_threads	4	STT에서 사용하는 CPU 스레드 수


마이크 번호는 장치 연결 상태에 따라 달라질 수 있습니다. 인식되지 않으면 입력 장치와 음량을 확인하세요. t를 선택하면 기존 텍스트 입력을 사용할 수 있습니다. 녹음이나 추론 도중에도 Ctrl+C로 중단할 수 있습니다.
8. 확인한 범위
작업 매핑, 모호한 입력 거부, 실행 확인, 기존 명령 옵션 유지, 추론 task와 저장 라벨 반영을 로봇 없이 테스트했습니다. 사용자의 노트북에서 음성을 인식하고 이클립스 task로 매핑하는 동작도 확인했습니다. 실제 로봇 추론 연결과 녹화 데이터셋의 Hub 업로드는 사용하는 PC에서 추가로 확인해야 합니다.