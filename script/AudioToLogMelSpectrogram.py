import torch
import torchaudio.transforms as T

class AudioToLogMelSpectrogram(torch.nn.Module):
    def __init__(self, sample_rate=16000, n_mels=128, n_fft=400, hop_length=160):
        super(AudioToLogMelSpectrogram, self).__init__()
        self.mel_spectrogram = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_mels=n_mels,
            n_fft=n_fft,
            hop_length=hop_length
        )

    def forward(self, waveform):
        mel_spec = self.mel_spectrogram(waveform)
        log_mel_spec = torch.log(mel_spec + 1e-9)  # Adding a small value to avoid log(0)
        return log_mel_spec
