"""Assistant module providing an abstract base class for threaded assistant implementations.

This module defines the Assistant class, which manages memory, threading, and callback mechanisms
for handling chat interactions and responses.
"""

from abc import ABC, abstractmethod
from collections.abc import Callable
from queue import Queue
from threading import Event, Thread

from .memory import Memory


class Assistant(Thread, ABC):
    """Abstract base class for threaded assistant implementations."""

    def __init__(self, system_prompt: str) -> None:
        """Initialize the Assistant."""
        self.memory = Memory(system_prompt)
        self._abort = Event()
        self._input_queue: Queue[str | None] = Queue()
        self._on_partial_response: Callable[[str], None] | None = None
        self._on_sentence_response: Callable[[str], None] | None = None
        self._on_finish: Callable[[], None] | None = None
        self._running = Event()

        super().__init__(daemon=True)

    @abstractmethod
    def run_once(self) -> None:
        """Run the assistant once."""
        ...

    def run(self) -> None:
        """Run the assistant."""
        while True:
            self.run_once()

    def on_partial_response(self, callback: Callable[[str], None]) -> None:
        """Set the callback for partial responses."""
        self._on_partial_response = callback

    def on_sentence_response(self, callback: Callable[[str], None]) -> None:
        """Set the callback for sentence responses."""
        self._on_sentence_response = callback

    def is_finished(self) -> bool:
        """Check if the assistant has finished running."""
        return not self._running.is_set()

    def on_finish(self, callback: Callable[[], None]) -> None:
        """Set the callback for when the assistant finishes."""
        self._on_finish = callback

    def chat(self, message: str | None = None) -> None:
        """Send a chat message to the assistant."""
        self._running.set()
        self._input_queue.put(message)

    def abort(self) -> None:
        """Abort the assistant's current operation."""
        with self._input_queue.mutex:
            self._input_queue.queue.clear()
        self._abort.set()
