"""Speech-to-text (STT) module using speech_recognition for audio transcription.

Provides the STT class for continuous or single-pass speech recognition
with callback support.
"""

from collections.abc import Callable
from threading import Thread

import speech_recognition as sr  # pyright: ignore[reportMissingTypeStubs]
from speech_recognition import AudioData  # pyright: ignore[reportMissingTypeStubs]


class STT(Thread):
    """Speech-to-text (STT) class using speech_recognition."""

    def __init__(self) -> None:
        """Initialize the speech-to-text recognizer and microphone."""
        self._recognizer = sr.Recognizer()
        self._microphone = sr.Microphone()
        self._on_recognize = None

        self._recognizer.pause_threshold = 2

        super().__init__(daemon=True)

    def run(self) -> None:
        """Continuously listen for audio input and processes it using run_once."""
        while True:
            self.run_once()

    def run_once(self) -> None:
        """Listen for a single audio input and process it."""
        with self._microphone as source:
            audio: AudioData = self._recognizer.listen(source) # pyright: ignore[reportUnknownMemberType, reportAssignmentType]

        try:
            text: str = self._recognizer.recognize_faster_whisper(audio) # pyright: ignore[reportUnknownVariableType, reportAttributeAccessIssue, reportUnknownMemberType]
            if self._on_recognize:
                self._on_recognize(text) # pyright: ignore[reportUnknownArgumentType]
        except sr.UnknownValueError:
            pass
        except sr.RequestError:
            pass

    def on_recognize(self, callback: Callable[[str], None]) -> None:
        """Set the callback function to be called when speech is recognized."""
        self._on_recognize = callback
