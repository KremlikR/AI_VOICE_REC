import torchaudio
import torch 
class LogMelSpectrogram(torch.Module):
    def __init__(self, sample_rate=8000, n_mels=128, win_length=160, hop_length=80):
        super(LogMelSpectrogram, self).__init__()
        self.mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate, n_mels=n_mels,
            win_length=win_length, hop_length=hop_length
        )

    def forward(self, x):
        mel_spec = self.mel_spectrogram(x)
        log_mel_spec = torch.log(mel_spec + 1e-14)  # logarithmic, add small value to avoid inf
        return log_mel_spec


def create_featurizer(sample_rate, n_mels=81):
    return LogMelSpectrogram(sample_rate=sample_rate, n_mels=n_mels, win_length=160, hop_length=80)
