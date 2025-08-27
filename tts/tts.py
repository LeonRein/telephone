from piper.voice import PiperVoice
from piper import AudioChunk
from threading import Thread, Event
from queue import Queue
from .player import Player

class TTS(Thread):
    def __init__(self):
        self._voice = PiperVoice.load("de_DE-thorsten-high.onnx")
        self._input_queue: Queue[str] = Queue()
        self._abort = Event()
        self._player = Player(self._voice.config.sample_rate)
        self._player.start()

        super().__init__(daemon=True)

    def run(self):
        while True:
            sentence: str = self._input_queue.get()
            self._abort.clear()

            part: AudioChunk
            for part in self._voice.synthesize(sentence):
                if self._abort.is_set():
                    break
                self._player.play(part.audio_int16_bytes)

    def speak(self, text: str):
        self._input_queue.put(text)

    def abort(self):
        with self._input_queue.mutex:
            self._input_queue.queue.clear()
        self._abort.set()
        self._player.abort()