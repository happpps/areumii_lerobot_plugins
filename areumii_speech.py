"""Optional push-to-talk task selection. Audio/STT dependencies load only on demand."""
from __future__ import annotations

import argparse
import json
import math
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path


def as_bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in {"true", "1", "yes"}:
        return True
    if value.lower() in {"false", "0", "no"}:
        return False
    raise argparse.ArgumentTypeError("true 또는 false를 입력하세요.")


def parse_speech_args(argv):
    """Strip only our CLI options; leave LeRobot's dotted arguments intact."""
    p = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    p.add_argument("--speech.enabled", dest="enabled", type=as_bool, default=False)
    p.add_argument("--speech.task_map", dest="task_map")
    p.add_argument("--speech.model", dest="model", default="small")
    p.add_argument("--speech.language", dest="language", default="ko")
    p.add_argument("--speech.seconds", dest="seconds", type=float, default=5.0)
    p.add_argument("--speech.input_device", dest="input_device")
    p.add_argument("--speech.cpu_threads", dest="cpu_threads", type=int, default=4)
    p.add_argument("--speech.list_devices", dest="list_devices", type=as_bool, default=False)
    opts, remaining = p.parse_known_args(argv)
    if any(arg.startswith("--speech.") for arg in remaining):
        p.error("알 수 없는 --speech 옵션입니다.")
    if not math.isfinite(opts.seconds) or not 0 < opts.seconds <= 30:
        p.error("--speech.seconds는 0 초과 30 이하로 지정하세요.")
    if opts.cpu_threads < 1:
        p.error("--speech.cpu_threads는 1 이상이어야 합니다.")
    if opts.enabled and not opts.task_map and not opts.list_devices:
        p.error("음성 입력에는 --speech.task_map=파일.json이 필요합니다.")
    return opts, remaining


def normalize(text):
    return re.sub(r"[^\w]", "", unicodedata.normalize("NFKC", text).casefold())


@dataclass(frozen=True)
class TaskChoice:
    name: str
    task: str
    aliases: tuple[str, ...]


def load_task_choices(path):
    raw = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    if not isinstance(raw, list) or not raw:
        raise ValueError("task_map은 작업 목록을 담은 비어 있지 않은 JSON 배열이어야 합니다.")
    choices = []
    seen = set()
    for item in raw:
        if not isinstance(item, dict):
            raise ValueError("각 작업은 name, task, aliases를 가진 객체여야 합니다.")
        name, task, aliases = item.get("name"), item.get("task"), item.get("aliases")
        if not isinstance(name, str) or not name.strip():
            raise ValueError("각 작업에 name을 입력하세요.")
        if not isinstance(task, str) or not task.strip() or "REPLACE_" in task:
            raise ValueError(f"{name}: task를 실제 학습에 쓴 정확한 문장으로 수정하세요.")
        if not isinstance(aliases, list) or not aliases:
            raise ValueError(f"{name}: aliases 배열이 필요합니다.")
        clean = []
        for alias in aliases:
            if not isinstance(alias, str) or not normalize(alias):
                raise ValueError(f"{name}: 빈 별칭은 사용할 수 없습니다.")
            key = normalize(alias)
            if key in seen:
                raise ValueError(f"중복된 별칭: {alias}")
            seen.add(key)
            clean.append(key)
        choices.append(TaskChoice(name.strip(), task.strip(), tuple(clean)))
    return choices


def match_task(text, choices):
    key = normalize(text)
    if not key:
        raise ValueError("음성이 인식되지 않았습니다. 다시 입력하세요.")
    # Conservative rejection. This is keyword selection, not language reasoning.
    if any(word in key for word in ("말고", "아니", "취소", "정지", "멈춰", "하지마")):
        raise ValueError("취소·부정 표현이 있어 실행하지 않습니다. 한 작업만 다시 말하세요.")
    matches = [c for c in choices if any(alias in key for alias in c.aliases)]
    if len(matches) != 1:
        raise ValueError("물체를 하나로 구분하지 못했습니다. 한 물체만 지정하세요.")
    return matches[0]


def list_audio_devices():
    try:
        import sounddevice as sd
    except (ImportError, OSError) as exc:
        raise RuntimeError("sounddevice와 PortAudio 설치를 확인하세요. README_STT.md 참고.") from exc
    print(sd.query_devices())


class SpeechTaskPrompt:
    def __init__(self, opts):
        self.opts = opts
        self.choices = load_task_choices(opts.task_map)
        self.model = None
        self.sd = None
        self.device = None
        self.sample_rate = None

    def prepare(self):
        """Validate microphone and load STT before robot context construction."""
        try:
            import sounddevice as sd
            from faster_whisper import WhisperModel
        except (ImportError, OSError) as exc:
            raise RuntimeError("음성 의존성 설치가 필요합니다. README_STT.md를 확인하세요.") from exc
        device = self.opts.input_device
        if device is not None and device.isdecimal():
            device = int(device)
        info = sd.query_devices(device, "input")
        rate = int(info["default_samplerate"])
        sd.check_input_settings(device=device, channels=1, dtype="float32", samplerate=rate)
        print(f"[STT] 마이크: {info['name']} / {rate} Hz", flush=True)
        print(f"[STT] 모델 {self.opts.model} 로딩 (CPU int8, 첫 실행은 다운로드 필요)", flush=True)
        self.model = WhisperModel(
            self.opts.model, device="cpu", compute_type="int8",
            cpu_threads=self.opts.cpu_threads,
        )
        self.sd, self.device, self.sample_rate = sd, device, rate

    def transcribe(self):
        import numpy as np
        print(f"[STT] 지금 말하세요. {self.opts.seconds:g}초 동안 녹음합니다.", flush=True)
        try:
            audio = self.sd.rec(
                int(self.opts.seconds * self.sample_rate), samplerate=self.sample_rate,
                channels=1, dtype="float32", device=self.device,
            )
            self.sd.wait()
        finally:
            self.sd.stop()
        audio = audio[:, 0]
        if np.sqrt(np.mean(audio ** 2)) < 0.0001:
            raise ValueError("마이크 입력이 너무 작습니다. 입력 장치와 음량을 확인하세요.")
        if self.sample_rate != 16000:
            from scipy.signal import resample_poly
            divisor = math.gcd(self.sample_rate, 16000)
            audio = resample_poly(audio, 16000 // divisor, self.sample_rate // divisor).astype("float32")
        segments, _ = self.model.transcribe(
            audio, language=self.opts.language, beam_size=5,
            vad_filter=True, condition_on_previous_text=False,
        )
        return " ".join(segment.text.strip() for segment in segments).strip()

    def request(self, previous=None, shutdown_event=None):
        while True:
            if shutdown_event is not None and shutdown_event.is_set():
                raise KeyboardInterrupt
            mode = input("입력 [v=음성 / t=텍스트 / Enter=이전 task / q=종료] >>> ").strip().lower()
            if mode == "q":
                raise KeyboardInterrupt
            try:
                if mode == "v":
                    text = self.transcribe()
                    print(f"[인식] {text}")
                    choice = match_task(text, self.choices)
                    print(f"[작업] {choice.name}")
                    task = choice.task
                elif mode == "t":
                    task = input("학습에 사용한 task 문장 >>> ").strip()
                elif mode == "" and previous:
                    task = previous
                else:
                    print("v 또는 t를 입력하세요. 첫 에피소드는 이전 task가 없습니다.")
                    continue
                if not task:
                    print("빈 task는 실행하지 않습니다.")
                    continue
                print(f"[모델에 전달] {task}")
                answer = input("이 작업을 실행할까요? [y=실행 / Enter=취소 / q=종료] >>> ").strip().lower()
                if answer == "q":
                    raise KeyboardInterrupt
                if answer == "y":
                    if shutdown_event is not None and shutdown_event.is_set():
                        raise KeyboardInterrupt
                    return task
                print("취소했습니다. 다시 입력하세요.")
            except (ValueError, RuntimeError) as exc:
                print(f"[STT] {exc}")
            except Exception as exc:
                print(f"[STT 오류] {exc}\n마이크 설정을 확인하거나 t로 텍스트를 입력하세요.")


if __name__ == "__main__":
    import sys
    options, remaining = parse_speech_args(sys.argv[1:])
    if remaining:
        raise SystemExit(f"알 수 없는 옵션: {remaining}")
    if options.list_devices:
        list_audio_devices()
    elif options.enabled:
        prompt = SpeechTaskPrompt(options)
        prompt.prepare()
        try:
            print("선택된 task:", prompt.request())
        except (KeyboardInterrupt, EOFError):
            print("\n음성 입력 테스트 종료.")
    else:
        print("--speech.list_devices=true 또는 --speech.enabled=true를 지정하세요.")
