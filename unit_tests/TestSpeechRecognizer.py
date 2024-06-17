import unittest
from unittest.mock import patch, MagicMock, call
import torch
import torchaudio
import wave
import pyaudio
from script.AudioListener import AudioListener
from script.BeamSearchDecoder import BeamSearchDecoder
from script.LogMelSpectogram import create_featurizer
from script.SpeechRecognizer import SpeechRecognizer

class TestSpeechRecognizer(unittest.TestCase):

    @patch('script.AudioListener.AudioListener')
    @patch('torch.jit.load')
    @patch('script.LogMelSpectograrm.create_featurizer')
    @patch('script.BeamSearchDecoder')
    def setUp(self, MockBeamSearchDecoder, MockCreateFeaturizer, MockJitLoad, MockAudioListener):
        self.mock_audio_listener = MockAudioListener.return_value
        self.mock_recognition_model = MockJitLoad.return_value
        self.mock_recognition_model.eval.return_value = self.mock_recognition_model
        self.mock_featurizer = MockCreateFeaturizer.return_value
        self.mock_beam_decoder = MockBeamSearchDecoder.return_value

        self.model_path = "dummy_model_path"
        self.lm_path = "dummy_lm_path"
        self.speech_recognizer = SpeechRecognizer(self.model_path, self.lm_path)

    @patch('wave.open')
    @patch('pyaudio.PyAudio')
    def test_save_audio(self, MockPyAudio, MockWaveOpen):
        mock_wave_write = MagicMock()
        MockWaveOpen.return_value = mock_wave_write
        mock_audio_interface = MockPyAudio.return_value
        mock_audio_interface.get_sample_size.return_value = 2

        audio_data = [b'audio_frame1', b'audio_frame2']
        filename = self.speech_recognizer.save_audio(audio_data)

        MockWaveOpen.assert_called_once_with(filename, "wb")
        mock_wave_write.setnchannels.assert_called_once_with(1)
        mock_wave_write.setsampwidth.assert_called_once_with(2)
        mock_wave_write.setframerate.assert_called_once_with(8000)
        mock_wave_write.writeframes.assert_called_once_with(b'audio_frame1audio_frame2')
        self.assertEqual(filename, "audio_temp")

    @patch('torchaudio.load')
    @patch('torch.no_grad')
    @patch('torch.jit.load')
    @patch('torch.nn.functional.softmax')
    def test_transcribe(self, MockSoftmax, MockJitLoad, MockNoGrad, MockTorchaudioLoad):
        mock_waveform = torch.rand(1, 16000)
        MockTorchaudioLoad.return_value = (mock_waveform, 8000)
        mock_log_mel_spectrogram = torch.rand(1, 1, 128, 201)
        self.mock_featurizer.return_value = mock_log_mel_spectrogram

        mock_model_output = torch.rand(1, 201, 29)
        self.mock_recognition_model.return_value = (mock_model_output, self.speech_recognizer.hidden_state)
        MockSoftmax.return_value = mock_model_output

        audio_data = [b'audio_frame1', b'audio_frame2']
        transcription, context_duration = self.speech_recognizer.transcribe(audio_data)

        self.assertTrue(MockNoGrad.called)
        MockTorchaudioLoad.assert_called_once_with("audio_temp")
        self.mock_featurizer.assert_called_once_with(mock_waveform)
        self.mock_recognition_model.assert_called_once_with(mock_log_mel_spectrogram.unsqueeze(1), self.speech_recognizer.hidden_state)
        MockSoftmax.assert_called_once_with(mock_model_output, dim=2)
        self.mock_beam_decoder.assert_called_once_with(mock_model_output.transpose(0, 1))
        self.assertIsNotNone(transcription)
        self.assertGreater(context_duration, 0)

    @patch('threading.Thread')
    def test_start(self, MockThread):
        mock_callback = MagicMock()

        self.speech_recognizer.start(mock_callback)

        self.mock_audio_listener.start.assert_called_once_with(self.speech_recognizer.audio_buffer)
        MockThread.assert_called_once_with(target=self.speech_recognizer.process_audio, args=(mock_callback,), daemon=True)
        MockThread.return_value.start.assert_called_once()

if __name__ == '__main__':
    unittest.main()
