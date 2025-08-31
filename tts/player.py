"""Audio player module for streaming and controlling playback using PyAudio.

This module defines the Player class for playing audio data in a separate thread,
with support for aborting playback and handling finish events.
"""

import math
from collections.abc import Callable
from itertools import batched
from queue import Queue
from threading import Event, Thread

import pyaudio

CHUNK_TIME = 0.5

class Player(Thread):
    """Audio player class for streaming audio data using PyAudio."""

    def __init__(self, sample_rate: int) -> None:
        """Initialize the audio player."""
        self._pyaudio = pyaudio.PyAudio()
        self._input_queue: Queue[bytes] = Queue()
        self._audio_stream: pyaudio.Stream
        self._init_audio()
        self._sample_rate = sample_rate
        self._abort = Event()
        self._on_finish: Callable[[], None] | None = None
        self._running = Event()

        super().__init__(daemon=True)

    def _init_audio(self) -> None:
        """Initialize the audio stream for playback."""
        self._audio_stream = self._pyaudio.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=22050,
            output=True,
        )

    def run(self) -> None:
        """Start the audio playback thread."""
        while True:
            data = self._input_queue.get()
            self._abort.clear()

            n: int = math.ceil(CHUNK_TIME * self._sample_rate)
            if n % 2 == 1:
                n += 1
            for chunk in batched(data, n):
                if self._abort.is_set():
                    break
                self._running.set()
                self._audio_stream.write(bytes(chunk))

            if self._input_queue.empty():
                self._running.clear()
                if self._on_finish:
                    self._on_finish()

    def is_finished(self) -> bool:
        """Check if the audio playback has finished."""
        return not self._running.is_set()

    def on_finish(self, callable: Callable[[], None]) -> None:  # noqa: A002
        """Set the callback function to be called when playback finishes."""
        self._on_finish = callable

    def play(self, audio_data: bytes) -> None:
        """Start playback of the given audio data."""
        self._running.set()
        self._input_queue.put(audio_data)

    def abort(self) -> None:
        """Abort playback and clear the input queue."""
        with self._input_queue.mutex:
            self._input_queue.queue.clear()
        self._abort.set()
