"""VoiceAdapter — wake-word voice input ("hey brain").

A background thread listens to the mic in short windows. Each window is gated by
a cheap energy check (silence is skipped without running the recognizer), then
transcribed locally. Only when the wake phrase "hey brain" is heard does the
adapter act: it speaks "I'm listening", captures the following utterance,
transcribes it, and queues it as a DIRECT StreamItem — which the daemon then
processes exactly like a typed task. If the command was spoken inline ("hey
brain, what's on my calendar") it's used directly without a second capture.

Every audio clip is written to a temp file only to transcribe, then deleted.
Nothing is recorded or kept; the mic is a wake-word trigger, not a recorder.

The hardware/model calls are injectable (`record_fn`, `transcribe_fn`,
`speak_fn`, `energy_fn`) so `_listen_once()` can be unit-tested with no mic,
no model, and no threads.
"""
from __future__ import annotations

import os
import queue
import tempfile
import threading
from typing import Callable, Optional

from .base import InputAdapter, StreamItem
from ..voice import audio_rms, detect_wake_word, record_audio, speak


class VoiceAdapter(InputAdapter):
    name = "voice"
    default_channel = "direct"

    def __init__(self, transcriber=None, *, wake_word: str = "hey brain",
                 ack: str = "I'm listening", listen_seconds: float = 2.0,
                 command_seconds: float = 6.0, energy_threshold: float = 300.0,
                 device: str = ":0", max_queue: int = 100,
                 record_fn: Optional[Callable] = None,
                 transcribe_fn: Optional[Callable] = None,
                 speak_fn: Optional[Callable] = None,
                 energy_fn: Optional[Callable] = None):
        self.transcriber = transcriber
        self.wake_word = wake_word
        self.ack = ack
        self.listen_seconds = float(listen_seconds)
        self.command_seconds = float(command_seconds)
        self.energy_threshold = float(energy_threshold)
        self.device = device
        self._inbox: "queue.Queue[StreamItem]" = queue.Queue(maxsize=max_queue)
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        # injectable seams (default to the real local helpers)
        self._record = record_fn or (
            lambda dest, secs: record_audio(dest, secs, device=self.device))
        self._transcribe = transcribe_fn or (
            lambda wav: self.transcriber.transcribe(wav) if self.transcriber else "")
        self._speak = speak_fn or speak
        self._energy = energy_fn or audio_rms

    # ── one listen cycle (pure of threads; unit-tested) ──────────────────────
    def _listen_once(self) -> Optional[StreamItem]:
        """Listen one window. Returns a StreamItem when a wake-word command was
        captured, else None. Never raises through to the loop."""
        text = self._capture_and_transcribe(self.listen_seconds)
        if not text:
            return None
        cmd = detect_wake_word(text, self.wake_word)
        if cmd is None:
            return None
        # wake word heard → acknowledge, then get the command
        self._speak(self.ack)
        command = cmd.strip()
        if not command:                      # wake phrase spoken alone
            command = self._capture_and_transcribe(
                self.command_seconds, gate=False).strip()
        if not command:
            return None
        return StreamItem(source=self.name, kind="task", content=command,
                          channel="direct", sender="voice",
                          metadata={"voice": True})

    def _capture_and_transcribe(self, seconds: float, *, gate: bool = True
                                ) -> str:
        wav = os.path.join(tempfile.mkdtemp(), "voice.wav")
        try:
            if not self._record(wav, seconds):
                return ""
            if gate and self._energy(wav) < self.energy_threshold:
                return ""              # silence — don't even run the recognizer
            return self._transcribe(wav) or ""
        except Exception:
            return ""
        finally:
            try:
                os.remove(wav)         # drop the audio, always
            except OSError:
                pass

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                item = self._listen_once()
                if item is not None:
                    self._inbox.put(item)
            except Exception:
                pass

    # ── InputAdapter API ─────────────────────────────────────────────────────
    def start(self) -> None:
        if self._thread is not None or self.transcriber is None:
            return
        self._thread = threading.Thread(target=self._loop, name="brain-voice",
                                        daemon=True)
        self._thread.start()

    def poll(self) -> list[StreamItem]:
        out: list[StreamItem] = []
        while True:
            try:
                out.append(self._inbox.get_nowait())
            except queue.Empty:
                break
        return out

    def speak(self, text: str) -> None:
        """Speak a line aloud (used to read the brain's reply back)."""
        self._speak(text)

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
