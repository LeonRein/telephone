import pyaudio
from threading import Thread, Event
from queue import Queue
import math
from itertools import batched

CHUNK_TIME = 0.5

class Player(Thread):
    def __init__(self, sample_rate: int):
        self._pyaudio = pyaudio.PyAudio()
        self._input_queue: Queue[bytes] = Queue()
        self._audio_stream: pyaudio.Stream
        self._init_audio()
        self._sample_rate = sample_rate
        self._abort = Event()

        super().__init__(daemon=True)

    def _init_audio(self):
        self._audio_stream = self._pyaudio.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=22050,
            output=True
        )

    def run(self):
        while True:
            data = self._input_queue.get()
            self._abort.clear()

            n: int = math.ceil(CHUNK_TIME * self._sample_rate)
            if n % 2 == 1:
                n += 1
            for chunk in batched(data, n):
                if self._abort.is_set():
                    break
                self._audio_stream.write(bytes(chunk))

    def play(self, audio_data: bytes):
        self._input_queue.put(audio_data)

    def abort(self):
        with self._input_queue.mutex:
            self._input_queue.queue.clear()
        self._abort.set()
