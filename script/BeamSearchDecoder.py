
import ctcdecode
import torch
textprocess = TextProcess()

labels = [
    "'",  # 0
    " ",  # 1
    "a",  # 2
    "b",
    "c",
    "d",
    "e",
    "f",
    "g",
    "h",
    "i",
    "j",
    "k",
    "l",
    "m",
    "n",
    "o",
    "p",
    "q",
    "r",
    "s",
    "t",
    "u",
    "v",
    "w",
    "x",
    "y",
    "z",  # 27
    "_",  # 28, blank
]

def DecodeGreedy(output, blank_label=28, collapse_repeated=True):
	arg_maxes = torch.argmax(output, dim=2).squeeze(1)
	decode = []
	for i, index in enumerate(arg_maxes):
		if index != blank_label:
			if collapse_repeated and i != 0 and index == arg_maxes[i -1]:
				continue
			decode.append(index.item())
	return textprocess.int_to_text_sequence(decode)

import ctcdecode
import torch
from script.TextProcess import TextProcess
TextProcess = TextProcess()
labels=[
    "'",  
    " ",  
    "a",  
    "b",
    "c",
    "d",
    "e",
    "f",
    "g",
    "h",
    "i",
    "j",
    "k",
    "l",
    "m",
    "n",
    "o",
    "p",
    "q",
    "r",
    "s",
    "t",
    "u",
    "v",
    "w",
    "x",
    "y",
    "z",  
    "_",  
    ]
def GreedyDecode(output, blank_label=28, collapse_repeated=True):
    arg_maxes = torch.argmax(output, dim=2).squeeze(1)
    decoded_sequence = []
    for i, index in enumerate(arg_maxes):
        if index != blank_label:
            if collapse_repeated and i > 0 and index == arg_maxes[i - 1]:
                continue
            decoded_sequence.append(index.item())
    return TextProcess.sequence_to_text(decoded_sequence)



class BeamSearchDecoder:


    def __init__(self, beam_width=100, blank_symbol='_', lm_path=None):
        print("Initializing beam search with language model...")
        blank_index = labels.index(blank_symbol)
        self.decoder = ctcdecode.CTCBeamDecoder(
            labels, alpha=0.522729216841, beta=0.96506699808,
            beam_width=beam_width, blank_id=blank_index,
            model_path=lm_path)
        print("Beam search initialization complete")

    def __call__(self, output):
        decoded_output, scores, time_steps, sequence_lengths = self.decoder.decode(output)
        return self._tokens_to_string(decoded_output[0][0], labels, sequence_lengths[0][0])

    def _tokens_to_string(self, tokens, vocabulary, length):
        return ''.join([vocabulary[token] for token in tokens[:length]])
