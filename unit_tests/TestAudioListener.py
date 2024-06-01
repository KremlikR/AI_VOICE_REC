import unittest
from unittest.mock import patch, MagicMock
from collections import deque
import time
import pyaudio
import time
import threading

class TestAudioListener(unittest.TestCase):

    @patch('pyaudio.PyAudio')
    def setUp(self, MockPyAudio):
        # Mocking PyAudio and stream
        self.mock_pyaudio = MockPyAudio.return_value
        self.mock_stream = self.mock_pyaudio.open.return_value
        self.mock_stream.read.return_value = b'audio_data'

        # Initialize AudioListener
        self.audio_listener = TestAudioListener(rate=8000, duration=2)

    def test_init(self):
        self.assertEqual(self.audio_listener.sample_rate, 8000)
        self.assertEqual(self.audio_listener.record_seconds, 2)
        self.assertEqual(self.audio_listener.buffer_size, 1024)
        self.assertIsNotNone(self.audio_listener.audio_interface)
        self.assertIsNotNone(self.audio_listener.stream)
        self.mock_pyaudio.open.assert_called_once_with(format=pyaudio.paInt16,
                                                       channels=1,
                                                       rate=8000,
                                                       input=True,
                                                       output=True,
                                                       frames_per_buffer=1024)

    def test_capture_audio(self):
        test_queue = deque()
        with patch.object(self.audio_listener, 'stream', self.mock_stream):
            with patch('time.sleep', return_value=None):  # Skip actual sleep to speed up the test
                thread = threading.Thread(target=self.audio_listener.capture_audio, args=(test_queue,), daemon=True)
                thread.start()
                time.sleep(0.05)  # Give some time for the thread to run
                self.assertTrue(len(test_queue) > 0)
                self.assertEqual(test_queue[0], b'audio_data')
                thread.join(timeout=0.1)

    def test_start(self):
        test_queue = deque()
        with patch.object(self.audio_listener, 'capture_audio') as mock_capture_audio:
            self.audio_listener.start(test_queue)
            self.assertTrue(mock_capture_audio.called)
            self.assertTrue(mock_capture_audio.call_count, 1)

if __name__ == '__main__':
    unittest.main()
