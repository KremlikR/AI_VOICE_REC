
import pyaudio
import time
import threading



import pyaudio
import time
import threading


class AudioListener:

    def __init__(self, rate=8000, duration=2):
        self.buffer_size = 1024
        self.sample_rate = rate
        self.record_seconds = duration
        self.audio_interface = pyaudio.PyAudio()
        self.stream = self.audio_interface.open(format=pyaudio.paInt16,
                                                channels=1,
                                                rate=self.sample_rate,
                                                input=True,
                                                output=True,
                                                frames_per_buffer=self.buffer_size)

    def capture_audio(self, queue):
        while True:
            audio_data = self.stream.read(self.buffer_size, exception_on_overflow=False)
            queue.append(audio_data)
            time.sleep(0.01)

    def start(self, queue):
        listener_thread = threading.Thread(target=self.capture_audio, args=(queue,), daemon=True)
        listener_thread.start()
        print("Audio capture has started...")
        