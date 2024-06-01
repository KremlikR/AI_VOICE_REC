import unittest
import torch
from mymodule import TextProcess, GreedyDecoder  # Adjust the import based on your module's actual name and structure

class TestTextProcess(unittest.TestCase):

    def setUp(self):
        self.text_processor = TextProcess()

    def test_initialization(self):
        expected_char_to_int = {
            "'": 0,
            "<SPACE>": 1,
            "a": 2, 
            "b": 3, 
            "c": 4, 
            "d": 5, 
            "e": 6,
            "f": 7, 
            "g": 8, 
            "h": 9, 
            "i": 10, 
            "j": 11, 
            "k": 12, 
            "l": 13,
            "m": 14, 
            "n": 15, 
            "o": 16, 
            "p": 17, 
            "q": 18, 
            "r": 19, 
            "s": 20,
            "t": 21, 
            "u": 22, 
            "v": 23, 
            "w": 24, 
            "x": 25, 
            "y": 26, 
            "z": 27
        }
        expected_int_to_char = {v: k for k, v in expected_char_to_int.items()}
        expected_int_to_char[1] = ' '

        self.assertEqual(self.text_processor.char_to_int, expected_char_to_int)
        self.assertEqual(self.text_processor.int_to_char, expected_int_to_char)

    def test_text_to_sequence(self):
        text = "hello world"
        expected_sequence = [9, 6, 13, 13, 16, 1, 24, 16, 19, 13, 5]
        sequence = self.text_processor.text_to_sequence(text)
        self.assertEqual(sequence, expected_sequence)

    def test_sequence_to_text(self):
        sequence = [9, 6, 13, 13, 16, 1, 24, 16, 19, 13, 5]
        expected_text = "hello world"
        text = self.text_processor.sequence_to_text(sequence)
        self.assertEqual(text, expected_text)

class TestGreedyDecoder(unittest.TestCase):

    def setUp(self):
        self.text_processor = TextProcess()

    def test_greedy_decoder(self):
        output = torch.tensor([[[0.1, 0.9, 0.0, 0.0], [0.9, 0.1, 0.0, 0.0], [0.1, 0.8, 0.1, 0.0], [0.9, 0.1, 0.0, 0.0]]])
        labels = torch.tensor([[2, 1, 2, 1]])
        label_lengths = torch.tensor([4])
        blank_label = 3

        decoded_sequences, target_sequences = GreedyDecoder(output, labels, label_lengths, blank_label)

        expected_decoded_sequences = [" "]
        expected_target_sequences = ["a a "]

        self.assertEqual(decoded_sequences, expected_decoded_sequences)
        self.assertEqual(target_sequences, expected_target_sequences)

if __name__ == '__main__':
    unittest.main()
