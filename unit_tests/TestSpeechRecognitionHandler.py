import unittest
from script.SpeechRecognizer import SpeechRecognizer
from mymodule import SpeechRecognitionHandler  

class TestSpeechRecognitionHandler(unittest.TestCase):

    def setUp(self):
        self.handler = SpeechRecognitionHandler()

    def test_initialization(self):
        self.assertEqual(self.handler.complete_transcription, "")
        self.assertEqual(self.handler.current_transcription, "")

    def test_call_method_short_context(self):
        result = ("hello world", 5)
        self.handler(result)
        self.assertEqual(self.handler.current_transcription, "hello world")
        self.assertEqual(self.handler.complete_transcription, "")

    def test_call_method_long_context(self):
        result = ("hello world", 15)
        self.handler(result)
        self.assertEqual(self.handler.current_transcription, "hello world")
        self.assertEqual(self.handler.complete_transcription, "hello world")

    def test_call_method_accumulate_transcriptions(self):
        result1 = ("hello", 15)
        result2 = ("world", 15)
        self.handler(result1)
        self.handler(result2)
        self.assertEqual(self.handler.current_transcription, "world")
        self.assertEqual(self.handler.complete_transcription, "hello world")

    def test_call_method_partial_context_update(self):
        result1 = ("hello", 15)
        result2 = ("beautiful", 5)
        result3 = ("world", 15)
        self.handler(result1)
        self.handler(result2)
        self.handler(result3)
        self.assertEqual(self.handler.current_transcription, "world")
        self.assertEqual(self.handler.complete_transcription, "hello beautiful world")

if __name__ == '__main__':
    unittest.main()
