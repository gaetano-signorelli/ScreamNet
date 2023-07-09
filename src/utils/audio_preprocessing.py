import numpy as np
import librosa
from PIL import Image

from src.config import *

class AudioFile:

    def __init__(self, path):

        self.wave = self.__read_file(path)
        self.spectrogram = None

    def preprocess(self, no_silence=True, filter_acceptability=True, pad=False):

        if no_silence:
            self.wave = self.__remove_silence()

        if TRANSFORM_METHOD == "stft":
            self.spectrogram = self.__get_short_fourier()
        elif TRANSFORM_METHOD == "mel":
            self.spectrogram = self.__get_mel_spectrogram()
        elif TRANSFORM_METHOD == "mfcc":
            self.spectrogram = self.__get_mfcc()

        else:
            raise Exception("Invalid transform method")

        segments = self.__split_spectrogram(filter_acceptability, pad)

        return segments

    def __read_file(self, path):

        wave, sampling_rate = librosa.load(path, sr=SAMPLING_RATE)

        return wave

    def __remove_silence(self):

        intervals_indices = librosa.effects.split(self.wave, top_db=TOP_DECIBEL)
        intervals = []

        for indices in intervals_indices:
            start = indices[0]
            end = indices[1]
            interval = self.wave[start:end]
            intervals.append(interval)

        silence_wave = np.concatenate(intervals)

        return silence_wave

    def __get_short_fourier(self):

        spectrogram = np.abs(librosa.stft(y=self.wave,
                                        hop_length=HOP_LENGTH,
                                        win_length = WINDOW_SIZE))

        return spectrogram

    def __get_mel_spectrogram(self):

        mel_spectrogram = librosa.feature.melspectrogram(y=self.wave,
                                                        sr=SAMPLING_RATE,
                                                        hop_length=HOP_LENGTH,
                                                        win_length=WINDOW_SIZE)

        return mel_spectrogram

    def __get_mfcc(self):

        mfcc = librosa.feature.mfcc(y=self.wave,
                                    sr=SAMPLING_RATE,
                                    hop_length=HOP_LENGTH,
                                    win_length=WINDOW_SIZE)

        return mfcc

    def __split_spectrogram(self, _filter, pad):

        is_acceptable = lambda x: (x > 0.01).sum() / (N_FRAMES*self.spectrogram.shape[0]) > ACCEPTABLE_RATE

        if pad:
            to_pad = N_FRAMES - (self.spectrogram.shape[1] % N_FRAMES)
            if to_pad!=N_FRAMES:
                self.spectrogram = np.pad(self.spectrogram, ((0,0),(0,to_pad)))

        else:
            to_remove = self.spectrogram.shape[1] % N_FRAMES
            if to_remove!=0:
                self.spectrogram = self.spectrogram[:, :-to_remove]

        length = self.spectrogram.shape[1] / N_FRAMES

        if length!=0:
            segments = np.array_split(self.spectrogram, length, axis=1)
            if _filter:
                segments = list(filter(is_acceptable, segments))
        else:
            segments = None

        return segments

#-------------------------------------------------------------------------------

def save_spectrogram(spectrogram, path):

    im = Image.fromarray(spectrogram)
    im.save(path)
