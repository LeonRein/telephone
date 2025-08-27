import pyaudio
from piper.voice import PiperVoice
from piper import AudioChunk
from threading import Thread, Event
from queue import Queue
class TTS(Thread):
    def __init__(self):
        self._voice = PiperVoice.load("de_DE-thorsten-high.onnx")
        self._input_queue: Queue[str] = Queue()
        self._abort = Event()
        self._audio_stream = pyaudio.PyAudio().open(
            format=pyaudio.paInt16,
            channels=getattr(self._voice.config, 'num_speakers', 1),
            rate=getattr(self._voice.config, 'sampling_rate', 22050),
            output=True
        )

        super().__init__(daemon=True)

    def run(self):
        while True:
            sentence: str = self._input_queue.get()
            self._abort.clear()

            part: AudioChunk
            for part in self._voice.synthesize(sentence):
                if self._abort.is_set():
                    return
                self._audio_stream.write(part.audio_int16_array.tobytes())

    def speak(self, text: str):
        self._input_queue.put(text)

    def abort(self):
        self._abort.set()