from .memory import Memory
from queue import Queue
from threading import Thread, Event
from typing import Callable
from abc import abstractmethod, ABC

class Assistant(Thread, ABC):
    def __init__(self, system_prompt: str):
        self.memory = Memory(system_prompt)
        self._abort = Event()
        self._input_queue: Queue[str | None] = Queue()
        self._on_partial_response: Callable[[str], None] | None = None
        self._on_sentence_response: Callable[[str], None] | None = None
        self._on_finish: Callable[[], None] | None = None

        super().__init__(daemon=True)

    @abstractmethod
    def run(self):
        ...
            
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