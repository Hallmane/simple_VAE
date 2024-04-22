"""
preprocessing pipeline for audio to log spectrograms

1. load audio file
2. pad the file if necessary
3. extracting log spectrogram from signal
4. normalize the spectrogram
5. save the normalized spectrogram
"""

import librosa
import numpy as np
import os
import pickle



class Loader:
    def __init__(self, sample_rate, duration=5, mono=True):
        self.sample_rate = sample_rate
        self.duration = duration
        self.mono = mono
        #self.num_samples_expected = sample_rate*duration

    def load(self, file_path):
        signal = librosa.load(file_path, sr=self.sample_rate, duration=self.duration, mono=self.mono)[0]
        return signal


class Padder:
    def __init__(self, mode="constant", side="right"):
        self.mode = mode
        self.side = side

    def pad(self, array, steps, side):
        padded = np.pad(array, 0, steps, mode=self.mode) if side=="right" else np.pad(array, steps, 0, mode=self.mode)
        return padded

class LogSpectrogramExtractor:
    """extracting log spectrograms (dB) from time-series signal"""
    def __init__(self, frame_size, hop_length):
        self.frame_size = frame_size # 
        self.hop_length = hop_length

    def extract(self, signal):
        # sliced to remove the last frequency bin
        stft = librosa.stft(signal, n_fft=self.frame_size, hop_length=self.hop_length)[:-1]
        return librosa.amplitude_to_db(np.abs(stft))


class MinMaxNormalizer:
    def __init__(self, min_val, max_val):
        self.min = min_val
        self.max = max_val

    def normalize(self, array):
        norm_array = (array - array.min()) / (array.max() - array.min())
        return norm_array * (self.max - self.min) + self.min

    def denormalize(self, norm_array, orig_min, orig_max):
        array = (norm_array - self.min) / (self.max - self.min) * (orig_max - orig_min) + orig_min
        return array
        


class Saver:
    """saving features and min/max values"""
    def __init__(self, features_save_dir, min_max_values_save_dir ):
        self.features_save_dir = features_save_dir
        self.min_max_values_save_dir = min_max_values_save_dir

    def save_feature(self, feature, file_path):
        save_path = self._generate_save_path(file_path)
        np.save(save_path, feature)

    def save_min_max_values(self, min_max_values):
        save_path = os.path.join(self.min_max_values_save_dir, "min_max_values.pkl") #pickle :/
        with open(save_path, "wb") as f:
            pickle.dump(min_max_values, f)

    def _generate_save_path(self, file_path):
        file_name = os.path.split(file_path)[1]
        save_path = os.path.join(self.features_save_dir, file_name + ".npy")
        return save_path

class Preprocessor:
    """
        audio file dir -> normalized spectrograms of files and their min/max values stored
    """

    def __init__(self):
        self.padder = None
        self.extractor = None
        self.normalizer = None
        self.saver = None
        self.min_max_values = {}
        self._num_samples_expected = None

    @property
    def loader(self):
        return self._loader

    @loader.setter
    def loader(self, loader):
        self._loader = loader
        self._num_samples_expected = int(loader.sample_rate * loader.duration)

    def process(self, audio_files_dir):
        for root, _, files in os.walk(audio_files_dir):
            for file in files:
                file_path = os.path.join(root, file)
                self._process_file(file_path)
                print(f"processing {file_path}")
        self.saver.save_min_max_values(self.min_max_values)

    def _process_file(self, file_path):
        signal = self.loader.load(file_path)
        if self._needs_padding(signal):
            signal = self._pad(signal)

        feature = self.extractor.extract(signal)
        norm_feature = self.normalizer.normalize(feature)
        save_path = self.saver.save_feature(norm_feature, file_path)
        self._store_min_max_value(save_path, feature.min, feature.max)

    def _needs_padding(self, signal):
        return len(signal) < self._num_samples_expected

    def _pad(self, signal):
        num_missing_samples = self._num_samples_expected - len(signal)
        return self.padder(signal, num_missing_samples)

    def _store_min_max_value(self, save_path, min_val, max_val):
        self.min_max_values[save_path] = {
            "min":min_val,
            "max":max_val
        }



if __name__ == "__main__":
    FRAME_SIZE = 512
    HOP_LENGTH = 256
    DURATION = 2.97 # experiment with this 
    SR = 22050
    MONO = True

    AUDIO_FILES_DIR = './dataset/audio_files/mp3/'
    SPECTROGRAM_SAVE_DIR = './dataset/spectrograms/'
    MIN_MAX_VALUS_SAVES_DIR = './dataset/min_max_values/'

    loader = Loader(SR, DURATION)
    padder = Padder()
    log_spectrogram_extractor = LogSpectrogramExtractor(FRAME_SIZE, HOP_LENGTH)
    min_max_normalizer = MinMaxNormalizer(0, 1)
    saver = Saver(SPECTROGRAM_SAVE_DIR, MIN_MAX_VALUS_SAVES_DIR)

    preprocessor = Preprocessor()
    preprocessor.loader = loader
    preprocessor.padder = padder
    preprocessor.extractor = log_spectrogram_extractor
    preprocessor.normalizer = min_max_normalizer
    preprocessor.saver = saver


    preprocessor.process(AUDIO_FILES_DIR)










