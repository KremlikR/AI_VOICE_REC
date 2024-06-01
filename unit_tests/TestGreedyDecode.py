import unittest
from unittest.mock import patch, MagicMock
import torch
import numpy as np
from script.TextProcess import TextProcess
from mymodule import GreedyDecode, BeamSearchDecoder  # Adjust the import based on your module's actual name and structure

class TestGreedyDecode(unittest.TestCase):

    @patch('script.TextProcess.TextProcess.sequence_to_text')
    def test_greedy_decode(self, mock_sequence_to_text):
        mock_sequence_to_text.return_value = 'hello'
        
        output = torch.tensor([[
            [0.1, 0.9, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 
             0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 
             0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
        ]])
        decoded_text = GreedyDecode(output)
        
        mock_sequence_to_text.assert_called_once()
        self.assertEqual(decoded_text, 'hello')

class TestBeamSearchDecoder(unittest.TestCase):

    @patch('ctcdecode.CTCBeamDecoder')
    def setUp(self, MockCTCBeamDecoder):
        self.mock_decoder = MockCTCBeamDecoder.return_value
        self.mock_decoder.decode.return_value = (
            np.array([[[1, 2, 3]]]),  # decoded_output
            np.array([0.9]),  # scores
            np.array([10]),  # time_steps
            np.array([3])  # sequence_lengths
        )
        self.labels = [
            "'", " ", "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l",
            "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z", "_"
        ]
        self.beam_search_decoder = BeamSearchDecoder(
            beam_width=100, blank_symbol='_', lm_path=None
        )

    def test_beam_search_decoder_initialization(self):
        self.mock_decoder.__init__.assert_called_once_with(
            self.labels, alpha=0.522729216841, beta=0.96506699808,
            beam_width=100, blank_id=self.labels.index('_'),
            model_path=None
        )

    def test_beam_search_decoder_call(self):
        output = torch.tensor([[[0.1, 0.2, 0.7], [0.3, 0.4, 0.3], [0.5, 0.1, 0.4]]])
        decoded_string = self.beam_search_decoder(output)
        self.assertEqual(decoded_string, 'abc')
        self.mock_decoder.decode.assert_called_once_with(output)

    def test_tokens_to_string(self):
        tokens = [2, 3, 4]
        length = 3
        expected_string = 'abc'
        result = self.beam_search_decoder._tokens_to_string(tokens, self.labels, length)
        self.assertEqual(result, expected_string)

if __name__ == '__main__':
    unittest.main()
