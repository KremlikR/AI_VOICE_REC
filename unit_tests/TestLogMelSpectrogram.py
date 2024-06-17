import unittest
from unittest.mock import patch, MagicMock
import torch
import torchaudio
from mymodule import LogMelSpectrogram, create_featurizer  

class TestLogMelSpectrogram(unittest.TestCase):

    @patch('torchaudio.transforms.MelSpectrogram')
    def setUp(self, MockMelSpectrogram):
        self.mock_mel_spectrogram = MockMelSpectrogram.return_value
        self.sample_rate = 8000
        self.n_mels = 128
        self.win_length = 160
        self.hop_length = 80
        self.log_mel_spec = LogMelSpectrogram(
            sample_rate=self.sample_rate,
            n_mels=self.n_mels,
            win_length=self.win_length,
            hop_length=self.hop_length
        )

    def test_init(self):
        self.mock_mel_spectrogram.__init__.assert_called_once_with(
            sample_rate=self.sample_rate,
            n_mels=self.n_mels,
            win_length=self.win_length,
            hop_length=self.hop_length
        )

    @patch('torch.log')
    def test_forward(self, mock_log):
        x = torch.rand(1, self.sample_rate * 2)  # A random tensor simulating a waveform
        mel_spec_output = torch.rand(1, self.n_mels, 201)  # Simulated output from MelSpectrogram
        self.mock_mel_spectrogram.return_value = mel_spec_output

        result = self.log_mel_spec(x)
        
        self.mock_mel_spectrogram.assert_called_once_with(x)
        mock_log.assert_called_once_with(mel_spec_output + 1e-14)
        self.assertEqual(result, mock_log.return_value)

class TestCreateFeaturizer(unittest.TestCase):

    @patch('mymodule.LogMelSpectrogram')
    def test_create_featurizer(self, MockLogMelSpectrogram):
        sample_rate = 8000
        n_mels = 81
        featurizer = create_featurizer(sample_rate, n_mels)
        
        MockLogMelSpectrogram.assert_called_once_with(
            sample_rate=sample_rate,
            n_mels=n_mels,
            win_length=160,
            hop_length=80
        )
        self.assertIsInstance(featurizer, MockLogMelSpectrogram)

if __name__ == '__main__':
    unittest.main()
