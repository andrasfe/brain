"""Voice — the brain's ears, wake-word gated.

This is NOT ambient transcription. The mic is only ever used to listen for the
wake phrase "hey brain"; nothing you say is recorded or transcribed until that
phrase fires. On a hit the brain speaks "I'm listening" (macOS `say`), captures
the single following utterance, transcribes it locally, and hands it to the
daemon's normal input pipeline as a direct task — so speaking to the brain is
identical to typing a task, just hands-free.

Privacy spine (same as screen/webcam):
  - transcription is LOCAL (mlx-whisper / faster-whisper on this machine); audio
    never leaves the box,
  - each short audio clip is written to a temp file ONLY to transcribe, then
    DELETED immediately,
  - a voice-energy gate skips silent windows so the recognizer isn't even run on
    silence,
  - committed default is OFF.

The recognizer is pluggable behind `SpeechTranscriber`; mlx-whisper is the
default backend (Apple-silicon native). Everything degrades to a no-op when the
backend / mic tool is absent. The pure pieces (`normalize_text`,
`detect_wake_word`) are dependency-free and unit-tested; the hardware/model
pieces are guarded.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
from typing import Optional

# Common Whisper mishearings of "brain" — used by the fuzzy wake matcher.
_DEFAULT_WAKE = "hey brain"


# ── text helpers (pure, testable) ────────────────────────────────────────────
def normalize_text(s: str) -> str:
    """Lowercase, drop punctuation, collapse whitespace."""
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", (s or "").lower())).strip()


def _levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1,
                           prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def detect_wake_word(text: str, wake: str = _DEFAULT_WAKE
                     ) -> Optional[str]:
    """If `text` contains the wake phrase, return the COMMAND that follows it
    (possibly empty when the wake phrase was spoken alone). Return None when no
    wake phrase is present.

    Tolerant of Whisper mishearings: an exact substring wins, else a fuzzy
    2-token match ("hey brain" ~ "hey brian", "hay brain", "hey brane", …)."""
    norm = normalize_text(text)
    wnorm = normalize_text(wake)
    if not norm or not wnorm:
        return None
    # exact substring
    idx = norm.find(wnorm)
    if idx != -1:
        return norm[idx + len(wnorm):].strip()
    # fuzzy 2-token match (only for the canonical 2-word wake phrase)
    wtoks = wnorm.split()
    toks = norm.split()
    if len(wtoks) == 2:
        for i in range(len(toks) - 1):
            if (_levenshtein(toks[i], wtoks[0]) <= 1
                    and _levenshtein(toks[i + 1], wtoks[1]) <= 2):
                return " ".join(toks[i + 2:]).strip()
    return None


# ── speech-to-text backends (pluggable) ──────────────────────────────────────
class SpeechTranscriber:
    def transcribe(self, wav_path: str) -> str:  # pragma: no cover
        raise NotImplementedError


class MLXWhisperTranscriber(SpeechTranscriber):
    """Local Whisper via Apple MLX. Model auto-downloads once from HF, then
    runs fully on-device. Optional dep — guarded by the factory."""

    def __init__(self, repo: str = "mlx-community/whisper-base.en-mlx"):
        import mlx_whisper  # noqa: F401  (import-time availability check)
        self._mlx_whisper = mlx_whisper
        self.repo = repo

    def transcribe(self, wav_path: str) -> str:
        try:
            r = self._mlx_whisper.transcribe(wav_path, path_or_hf_repo=self.repo)
            return str(r.get("text", "")).strip()
        except Exception:
            return ""


class FasterWhisperTranscriber(SpeechTranscriber):
    """Local Whisper via CTranslate2. Optional dep — guarded by the factory."""

    def __init__(self, model: str = "base.en"):
        from faster_whisper import WhisperModel
        self._model = WhisperModel(model, device="auto", compute_type="int8")

    def transcribe(self, wav_path: str) -> str:
        try:
            segments, _ = self._model.transcribe(wav_path)
            return " ".join(s.text for s in segments).strip()
        except Exception:
            return ""


_TRANSCRIBER_CACHE: dict[str, Optional[SpeechTranscriber]] = {}


def make_transcriber(cfg) -> Optional[SpeechTranscriber]:
    """Build (and cache) the configured local transcriber, or None when voice
    is disabled or no backend loads. Never raises."""
    vc = (getattr(cfg, "raw", {}) or {}).get("voice") or {}
    if not vc.get("enabled"):
        return None
    backend = str(vc.get("backend", "mlx_whisper")).lower()
    model = str(vc.get("model", ""))
    ck = f"{backend}:{model}"
    if ck in _TRANSCRIBER_CACHE:
        return _TRANSCRIBER_CACHE[ck]
    t: Optional[SpeechTranscriber] = None
    try:
        if backend in ("mlx_whisper", "mlx-whisper", "mlx"):
            t = MLXWhisperTranscriber(
                repo=model or "mlx-community/whisper-base.en-mlx")
        elif backend in ("faster_whisper", "faster-whisper"):
            t = FasterWhisperTranscriber(model=model or "base.en")
    except Exception:
        t = None
    _TRANSCRIBER_CACHE[ck] = t
    return t


# ── mic capture + voice-activity gate + speech output ─────────────────────────
def record_audio(dest: str, seconds: float, *, device: str = ":0") -> bool:
    """Record `seconds` of mic audio to a 16 kHz mono WAV via ffmpeg /
    avfoundation (same toolchain as the webcam path). Returns success."""
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return False
    try:
        r = subprocess.run(
            [ffmpeg, "-hide_banner", "-loglevel", "error",
             "-f", "avfoundation", "-i", str(device),
             "-t", f"{seconds:.2f}", "-ar", "16000", "-ac", "1", "-y", dest],
            capture_output=True, timeout=seconds + 15)
        return r.returncode == 0 and os.path.exists(dest)
    except Exception:
        return False


def audio_rms(wav_path: str) -> float:
    """RMS amplitude of a 16-bit WAV (stdlib only). ~<300 is silence, speech is
    typically several hundred to a few thousand. Used to skip the recognizer on
    silent windows."""
    try:
        import audioop
        import wave
        with wave.open(wav_path, "rb") as w:
            frames = w.readframes(w.getnframes())
            width = w.getsampwidth()
        if not frames:
            return 0.0
        return float(audioop.rms(frames, width))
    except Exception:
        return 0.0


def speak(text: str) -> None:
    """Speak `text` aloud via macOS `say` (local, no deps). Best-effort."""
    say = shutil.which("say")
    if not say or not text:
        return
    try:
        subprocess.Popen([say, text[:400]],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        pass


# ── CLI: prove the mic + STT once ────────────────────────────────────────────
def main() -> int:
    import sys
    from pathlib import Path
    repo = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(repo))
    from brain.config import load_config

    cfg = load_config()
    t = make_transcriber(cfg)
    if t is None:
        print("voice disabled or no local backend "
              "(set voice.enabled + `pip install mlx-whisper`)")
        return 1
    d = tempfile.mkdtemp()
    wav = os.path.join(d, "voice_test.wav")
    print("🎙  recording 4s — say something…")
    if not record_audio(wav, 4.0):
        print("mic capture failed (grant mic access to the terminal).")
        return 1
    try:
        print(f"   rms={audio_rms(wav):.0f}")
        text = t.transcribe(wav)
    finally:
        try:
            os.remove(wav)
        except OSError:
            pass
    print(f"heard: {text!r}")
    cmd = detect_wake_word(text)
    print(f"wake-word → {'no' if cmd is None else f'YES, command={cmd!r}'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
