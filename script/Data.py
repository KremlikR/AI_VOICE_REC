import torch
import torchaudio
import torch.nn as nn
import pandas as pd
from script.LogMelSpec import LogMelSpec
from script.TextProcess import  TextProcess
from script.SpecAugment import  SpecAugment

class Data(torch.utils.data.Dataset):
    """
    Dataset class for loading and processing data.
    """
    parameters = {
        "sample_rate": 8000, "n_feats": 81,
        "specaug_rate": 0.5, "specaug_policy": 3,
        "time_mask": 70, "freq_mask": 15 
    }

    def __init__(self, json_path, sample_rate, n_feats, specaug_rate, specaug_policy,
                 time_mask, freq_mask, valid=False, shuffle=True, text_to_int=True, log_ex=True):
        self.log_ex = log_ex
        self.text_process = TextProcess()

        print("Loading data from", json_path)
        self.data = pd.read_json(json_path, lines=True)

        if valid:
            self.audio_transforms = nn.Sequential(
                LogMelSpec(sample_rate=sample_rate, n_mels=n_feats, win_length=160, hop_length=80)
            )
        else:
            self.audio_transforms = nn.Sequential(
                LogMelSpec(sample_rate=sample_rate, n_mels=n_feats, win_length=160, hop_length=80),
                SpecAugment(specaug_rate, specaug_policy, freq_mask, time_mask)
            )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.item()

        try:
            file_path = self.data['key'].iloc[idx]
            waveform, _ = torchaudio.load(file_path)
            label = self.text_process.text_to_int_sequence(self.data['text'].iloc[idx])
            spectrogram = self.audio_transforms(waveform)  # (channel, feature, time)
            spec_len = spectrogram.shape[-1] // 2
            label_len = len(label)

            if spec_len < label_len:
                raise ValueError('Spectrogram length is smaller than label length')
            if spectrogram.shape[0] > 1:
                raise ValueError(f'Dual channel, skipping audio file {file_path}')
            if spectrogram.shape[2] > 1650:
                raise ValueError(f'Spectrogram too large. Size: {spectrogram.shape[2]}')
            if label_len == 0:
                raise ValueError(f'Label length is zero, skipping {file_path}')
        except Exception as e:
            if self.log_ex:
                print(str(e), file_path)
            return self.__getitem__(idx - 1 if idx != 0 else idx + 1)

        return spectrogram, label, spec_len, label_len

    def describe(self):
        return self.data.describe()

def collate_fn_padd(data):
    """
    Pads batch of variable length.
    """
    spectrograms = []
    labels = []
    input_lengths = []
    label_lengths = []

    for (spectrogram, label, input_length, label_length) in data:
        if spectrogram is None:
            continue
        spectrograms.append(spectrogram.squeeze(0).transpose(0, 1))
        labels.append(torch.tensor(label))
        input_lengths.append(input_length)
        label_lengths.append(label_length)

    spectrograms = nn.utils.rnn.pad_sequence(spectrograms, batch_first=True).unsqueeze(1).transpose(2, 3)
    labels = nn.utils.rnn.pad_sequence(labels, batch_first=True)

    return spectrograms, labels, input_lengths, label_lengths