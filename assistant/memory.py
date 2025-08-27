from typing import List, Dict

class Memory:
    def __init__(self, system_prompt: str | None = None):
        self._messages : List[Dict[str, str]] = []
        if system_prompt:
            self.add_system_message(system_prompt)

    def add_user_message(self, message: str):
        self._messages.append({"role": "user", "content": message})

    def add_assistant_message(self, message: str):
        self._messages.append({"role": "assistant", "content": message})

    def add_system_message(self, message: str):
        self._messages.append({"role": "system", "content": message})

    def get_messages(self):
        return self._messages

    def clear(self):
        self._messages = []