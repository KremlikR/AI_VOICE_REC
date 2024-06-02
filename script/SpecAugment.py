import torch
import torchaudio
import torch.nn as nn
class SpecAugment(nn.Module):
    """
    SpecAugment module for data augmentation.
    """
    def __init__(self, rate, policy=3, freq_mask=15, time_mask=35):
        super(SpecAugment, self).__init__()
        self.rate = rate
        self.specaug = nn.Sequential(
            torchaudio.transforms.FrequencyMasking(freq_mask_param=freq_mask),
            torchaudio.transforms.TimeMasking(time_mask_param=time_mask)
        )
        self.specaug2 = nn.Sequential(
            torchaudio.transforms.FrequencyMasking(freq_mask_param=freq_mask),
            torchaudio.transforms.TimeMasking(time_mask_param=time_mask),
            torchaudio.transforms.FrequencyMasking(freq_mask_param=freq_mask),
            torchaudio.transforms.TimeMasking(time_mask_param=time_mask)
        )
        policies = {1: self.policy1, 2: self.policy2, 3: self.policy3}
        self._forward = policies[policy]

    def forward(self, x):
        return self._forward(x)

    def policy1(self, x):
        if self.rate > torch.rand(1).item():
            return self.specaug(x)
        return x

    def policy2(self, x):
        if self.rate > torch.rand(1).item():
            return self.specaug2(x)
        return x

    def policy3(self, x):
        if torch.rand(1).item() > 0.5:
            return self.policy1(x)
        return self.policy2(x)