import torch
class TextProcess:
    def __init__(self):
        char_mapping = """
        ' 0
        <SPACE> 1
        a 2
        b 3
        c 4
        d 5
        e 6
        f 7
        g 8
        h 9
        i 10
        j 11
        k 12
        l 13
        m 14
        n 15
        o 16
        p 17
        q 18
        r 19
        s 20
        t 21
        u 22
        v 23
        w 24
        x 25
        y 26
        z 27
        """
        self.char_to_int = {}
        self.int_to_char = {}
        for line in char_mapping.strip().split('\n'):
            char, idx = line.split()
            self.char_to_int[char] = int(idx)
            self.int_to_char[int(idx)] = char
        self.int_to_char[1] = ' '

    def text_to_sequence(self, text):
        """Convert text into a sequence of integers using the character map."""
        sequence = []
        for char in text:
            if char == ' ':
                mapped_char = self.char_to_int['<SPACE>']
            else:
                mapped_char = self.char_to_int[char]
            sequence.append(mapped_char)
        return sequence

    def sequence_to_text(self, sequence):
        """Convert a sequence of integers back into text using the character map."""
        text = []
        for idx in sequence:
            text.append(self.int_to_char[idx])
        return ''.join(text).replace('<SPACE>', ' ')

text_processor = TextProcess()

def GreedyDecoder(output, labels, label_lengths, blank_label=28, collapse_repeated=True):
    arg_maxes = torch.argmax(output, dim=2)
    decoded_sequences = []
    target_sequences = []
    for i, arg in enumerate(arg_maxes):
        decoded = []
        target_sequences.append(
            text_processor.sequence_to_text(labels[i][:label_lengths[i]].tolist())
        )
        for j, index in enumerate(arg):
            if index != blank_label:
                if collapse_repeated and j != 0 and index == arg[j - 1]:
                    continue
                decoded.append(index.item())
        decoded_sequences.append(text_processor.sequence_to_text(decoded))
    return decoded_sequences, target_sequences

