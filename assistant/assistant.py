from ollama import Client, ChatResponse
from .memory import Memory
from queue import Queue
from threading import Thread, Event
from typing import Callable
import re

# MODEL = "magistral"
MODEL = "gemma3:27b"

class Assistant(Thread):
    def __init__(self, system_prompt: str):
        self.memory = Memory(system_prompt)
        self.client : Client = Client()
        self._abort = Event()
        self._input_queue: Queue[str | None] = Queue()
        self._on_partial_response: Callable[[str], None] | None = None
        self._on_sentence_response: Callable[[str], None] | None = None
        self._on_finish: Callable[[], None] | None = None

        super().__init__(daemon=True)

    def run(self):
        while True:
            if message := self._input_queue.get():
                self._abort.clear()
                self.memory.add_user_message(message)

            response_message: str = ""
            sentence: str = ""

            for part in self.client.chat(model=MODEL, messages=self.memory.get_messages(), stream=True):  # type: ignore
                part: ChatResponse
                if part.message.role != "assistant":
                    continue

                if part.message.content:
                    response_message += part.message.content
                    sentence += part.message.content

                if any(sentence.endswith(p) for p in [".", "!", "?", ":", "\n"]):
                    sentence = sentence.replace("\n", " ")
                    if re.search(r"\w", sentence):
                        if self._on_sentence_response:
                            self._on_sentence_response(sentence)
                        sentence = ""

                if self._abort.is_set():
                    break

                if self._on_partial_response and part.message.content:
                    self._on_partial_response(part.message.content)

            if not self._abort.is_set():
                sentence = sentence.replace("\n", " ")
                if self._on_sentence_response and len(sentence) > 0 and re.search(r"\w", sentence):
                    self._on_sentence_response(sentence)

            if self._on_finish:
                self._on_finish()

            self.memory.add_assistant_message(response_message)

    def on_partial_response(self, callback: Callable[[str], None]):
        self._on_partial_response = callback

    def on_sentence_response(self, callback: Callable[[str], None]):
        self._on_sentence_response = callback
    
    def on_finish(self, callback: Callable[[], None]):
        self._on_finish = callback

    def chat(self, message: str | None = None):
        self._input_queue.put(message)

    def abort(self):
        with self._input_queue.mutex:
            self._input_queue.queue.clear()
        self._abort.set()