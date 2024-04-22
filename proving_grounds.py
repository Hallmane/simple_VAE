import librosa
from matplotlib import pyplot as plt
import numpy as np
import torch
from torch import nn

#signal = librosa.load('./dataset/spectrograms/OttoTito-EscapeArtist.mp3.npy', sr=22050, duration=3, mono=True)
#signal = np.load('./dataset/spectrograms/OttoTito-EscapeArtist.mp3.npy')
signal = np.load('./dataset/spectrograms/LɅVΣN-IThoughtYouWereReal.mp3.npy')
#signal = np.load('./dataset/spectrograms/Balrog-TotalEcstasy-OriginalMix.mp3.npy')
print(signal[240])

conv_block = nn.Sequential(
    nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1),
    nn.LeakyReLU(0.2),
    nn.BatchNorm2d(64)
)


# (batch_size, num_channels, height, width)
#print(conv_block(torch.randn(1, 1, 256, 256)))
#print(conv_block(torch.randn(1, 1, 256, 256)).shape)



