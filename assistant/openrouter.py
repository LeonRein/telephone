from openai import OpenAI, Stream
from openai.types.chat import ChatCompletionChunk, ChatCompletionMessageParam
import typing
from rich import print

from .assistant import Assistant
import re
from utils.secrets import OPENROUTER_API_KEY

# MODEL = "magistral"
MODEL = "google/gemini-2.5-pro"

class OpenRouterAssistant(Assistant):
    def __init__(self, system_prompt: str):
        self._client : OpenAI = OpenAI(base_url="https://openrouter.ai/api/v1",
                                      api_key=OPENROUTER_API_KEY)
        super().__init__(system_prompt=system_prompt)

    # convert list[dict[str, str]] to typing.Generator[ChatCompletionMessageParam, None, None]
    @staticmethod
    def typed_messages(messages: list[dict[str, str]]) -> typing.Generator[ChatCompletionMessageParam, None, None]:
        for msg in messages:
            yield typing.cast(ChatCompletionMessageParam, msg)

    def run(self):
        while True:
            if message := self._input_queue.get():
                self._abort.clear()
                self.memory.add_user_message(message)

            response_message: str = ""
            sentence: str = ""

            thinking = False

            stream: Stream[ChatCompletionChunk] = self._client.chat.completions.create(model=MODEL, messages=self.typed_messages(self.memory.get_messages()), reasoning_effort="minimal", stream=True)

            for part in stream:
                delta = part.choices[0].delta
                if delta.role != "assistant":
                    continue
                reasoning: str = delta.model_extra.get("reasoning", None)
                respons_part: str = delta.content

                if "<think>" in respons_part:
                    thinking = True

                if "</think>" in respons_part:
                    thinking = False

                if thinking:
                    reasoning = respons_part
                    respons_part = ""

                if reasoning:
                    print(f"[magenta]{reasoning}[/magenta]", end="", flush=True)
                if respons_part == "":
                    continue

                response_message += respons_part
                sentence += respons_part

                if any(sentence.endswith(p) for p in [".", "!", "?", ":", "\n"]):
                    sentence = sentence.replace("\n", " ")
                    if re.search(r"\w", sentence):
                        if self._on_sentence_response:
                            self._on_sentence_response(sentence)
                        sentence = ""

                if self._on_partial_response:
                    self._on_partial_response(respons_part)

                if self._abort.is_set():
                    break

            if not self._abort.is_set():
                sentence = sentence.replace("\n", " ")
                if self._on_sentence_response and len(sentence) > 0 and re.search(r"\w", sentence):
                    self._on_sentence_response(sentence)

            if self._on_finish:
                self._on_finish()

            self.memory.add_assistant_message(response_message)

                