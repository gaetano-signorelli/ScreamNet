import os
import numpy as np
import librosa
import soundfile as sf

from src.config import *

class AudioFile:

    def __init__(self, path, name, format="mp3"):

        self.wave = self.__read_file(path)

        self.name = name
        self.format = format

    def preprocess(self):

        self.wave = self.__remove_silence()

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

    def save(self, output_path):

        path = os.path.join(output_path, self.name + "_silence" + "." + self.format)

        if self.wave is not None:
            sf.write(path, self.wave, SAMPLING_RATE, format=self.format)
