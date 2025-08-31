"""Text-to-Speech (TTS) module using PiperVoice and a custom Player.

This module provides the TTS class for synthesizing and playing speech audio from text input.
"""

from collections.abc import Callable
from queue import Queue
from threading import Event, Thread
from typing import TYPE_CHECKING

from piper.voice import PiperVoice

from .player import Player

if TYPE_CHECKING:
    from piper import AudioChunk


class TTS(Thread):
    """Text-to-Speech (TTS) class using PiperVoice and a custom Player."""

    def __init__(self) -> None:
        """Initialize the TTS engine."""
        self._voice = PiperVoice.load("de_DE-thorsten-high.onnx")
        self._input_queue: Queue[str] = Queue()
        self._abort = Event()
        self._internal_running = Event()
        self._running = Event()
        self._player = Player(self._voice.config.sample_rate)
        self._player.start()
        self._player.on_finish(self._on_player_finish)

        self._on_finish: Callable[[], None] | None = None

        super().__init__(daemon=True)

    def run(self) -> None:
        """Run the TTS engine."""
        while True:
            self.run_once()

    def run_once(self) -> None:
        """Run a single iteration of the TTS engine."""
        sentence: str = self._input_queue.get()
        self._abort.clear()

        part: AudioChunk
        for part in self._voice.synthesize(sentence):
            if self._abort.is_set():
                break
            self._player.play(part.audio_int16_bytes)

        if self._input_queue.empty():
            self._internal_running.clear()
            if self._player.is_finished():
                self._running.clear()
                if self._on_finish:
                    self._on_finish()

    def _on_player_finish(self) -> None:
        """Handle the player finish event."""
        if not self._internal_running.is_set():
            self._running.clear()
            if self._on_finish:
                self._on_finish()


    def is_finished(self) -> bool:
        """Check if the TTS engine has finished."""
        return not self._running.is_set()

    def on_finish(self, callable: Callable[[], None]) -> None:  # noqa: A002
        """Set the on finish callback."""
        self._on_finish = callable

    def speak(self, text: str) -> None:
        """Speak the given text."""
        self._internal_running.set()
        self._running.set()
        self._input_queue.put(text)

    def abort(self) -> None:
        """Abort the current TTS operation."""
        with self._input_queue.mutex:
            self._input_queue.queue.clear()
        self._abort.set()
        self._player.abort()
