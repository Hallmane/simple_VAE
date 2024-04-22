import librosa
from matplotlib import pyplot as plt
import numpy as np

#plt.figure(figsize=(25,10))

#signal = librosa.load('./dataset/spectrograms/OttoTito-EscapeArtist.mp3.npy', sr=22050, duration=3, mono=True)
#signal = np.load('./dataset/spectrograms/OttoTito-EscapeArtist.mp3.npy')
#signal = np.load('./dataset/spectrograms/Balrog-TotalEcstasy-OriginalMix.mp3.npy')
signal = np.load('./dataset/spectrograms/LɅVΣN-IThoughtYouWereReal.mp3.npy')
#print(np.shape(signal))

librosa.display.specshow(signal, 
                         sr=22050, 
                         hop_length=256,
                         x_axis="time",
                         y_axis="log")

plt.colorbar(format="%+2.f")
plt.show()
