from ollama import ChatResponse, Client
from .assistant import Assistant
import re

# MODEL = "magistral"
MODEL = "gemma3:27b"

class OllamaAssistant(Assistant):
    def __init__(self, system_prompt: str):
        self._client : Client = Client()
        super().__init__(system_prompt=system_prompt)

    def run(self):
        while True:
            if message := self._input_queue.get():
                self._abort.clear()
                self.memory.add_user_message(message)

            response_message: str = ""
            sentence: str = ""

            for part in self._client.chat(model=MODEL, messages=self.memory.get_messages(), stream=True):  # type: ignore
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