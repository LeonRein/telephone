import pyaudio
from piper.voice import PiperVoice
from piper import AudioChunk
from threading import Thread, Event
from queue import Queue
from time import sleep
from itertools import batched
import math

CHUNK_TIME = 0.5

class TTS(Thread):
    def __init__(self):
        self._voice = PiperVoice.load("de_DE-thorsten-high.onnx")
        self._input_queue: Queue[str] = Queue()
        self._abort = Event()
        self._pyaudio = pyaudio.PyAudio()
        self._audio_stream: pyaudio.Stream
        self._init_audio()

        super().__init__(daemon=True)

    def _init_audio(self):
        self._audio_stream = self._pyaudio.open(
            format=pyaudio.paInt16,
            channels=getattr(self._voice.config, 'num_speakers', 1),
            rate=getattr(self._voice.config, 'sampling_rate', 22050),
            output=True
        )

    def run(self):
        while True:
            sentence: str = self._input_queue.get()
            self._abort.clear()

            part: AudioChunk
            for part in self._voice.synthesize(sentence):

                n: int = math.ceil(CHUNK_TIME * part.sample_rate)
                if n % 2 == 1:
                    n += 1
                for chunk in batched(part.audio_int16_bytes, n):
                    if self._abort.is_set():
                        with self._input_queue.mutex:
                            self._input_queue.queue.clear()
                        break
                    self._audio_stream.write(bytes(chunk))

    def speak(self, text: str):
        self._input_queue.put(text)

    def abort(self):
        print("Aborting TTS...")
        self._abort.set()