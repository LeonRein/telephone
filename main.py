"""Main module for the Telephone application.

This module initializes and runs the Telephone app, which uses speech-to-text (STT),
text-to-speech (TTS), and an assistant for interactive conversations.
"""

import time
from pathlib import Path

from rich import print  # noqa: A004

from assistant import OllamaAssistant as Assistant
from stt import STT
from tts import TTS


def read_system_prompt() -> str:
    """Read the system prompt from a file."""
    with Path("system_prompt.txt").open() as file:
        return file.read().strip()


class Telephone:
    """Main class for the Telephone application."""

    def __init__(self) -> None:
        """Initialize the Telephone application."""
        self.assistant = Assistant(read_system_prompt())
        self.tts = TTS()
        self.stt = STT()

        self.assistant.on_partial_response(lambda text: print(text, end="", flush=True))
        self.assistant.on_sentence_response(lambda text: self.tts.speak(text))
        self.assistant.on_finish(self.tts_next_sentence)
        self.tts.on_finish(self.tts_next_sentence)
        self.stt.on_recognize(self.on_stt_recognize)

        self.assistant_first_response = False

    def tts_next_sentence(self) -> None:
        """Proceed to the next sentence in the conversation."""
        if self.assistant.is_finished() and self.tts.is_finished():
            print("\n[green]You: [/green]", end="", flush=True)
            self.stt.run_once()

    def on_stt_recognize(self, text: str) -> None:
        """Handle recognized speech from the STT module."""
        self.tts.abort()
        self.assistant.abort()
        print(f"[green]{text}[/green]")
        self.assistant.chat(text)

    def run(self) -> None:
        """Start the Telephone application."""
        self.tts.start()
        self.assistant.start()

        self.assistant.chat(None)

        while True:
            time.sleep(100)


if __name__ == "__main__":
    telephone = Telephone()
    telephone.run()
