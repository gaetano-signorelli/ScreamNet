import random
import librosa
import numpy as np
import tensorflow as tf
from tf import keras

from src.config import *

class DatasetLoader(keras.utils.Sequence):

    def __init__(self):

        self.whispers = self.__get_files(WHISPERS_PATH)
        self.screams = self.__get_files(SCREAMS_PATH)

    def on_epoch_end(self):
        pass

    def __getitem__(self, idx):

        whispers_batch_audios = np.random.choice(self.whispers, size=BATCH_SIZE)
        screams_batch_audios = np.random.choice(self.screams, size=BATCH_SIZE)

        batch_whispers = []
        batch_screams = []

        for whisper_audio, scream_audio in zip(whispers_batch_audios, screams_batch_audios):
            batch_whispers.append(self.__load_segment(whisper_audio))
            batch_screams.append(self.__load_segment(scream_audio))

        batch_whispers = np.array(batch_whispers)
        batch_screams = np.array(batch_screams)

        return batch_whispers, batch_screams

    def __len__(self):

        return EPOCH_LEN

    def __get_files(self, root):

        paths = []

        for dirname, dirnames, filenames in os.walk(root):
            for filename in filenames:
                paths.append(os.path.join(dirname, filename))

        return paths

    def __load_audio(self, audio_file):

        wave, sampling_rate = librosa.load(audio_file, sr=SAMPLING_RATE)

        if NORMALIZE:
            wave = librosa.util.normalize(wave) * 0.95 # *0.95 comes from the official MelGan implementation

        if AUGMENT:
            amplitude = np.random.uniform(low=0.3, high=1.0)
            wave = wave * amplitude

        return wave

    def __load_segment(self, audio_file):

        wave = self.__load_audio(audio_file)

        if wave.shape[0] >= SEGMENT_LENGTH:
            max_audio_start = wave.shape[0] - SEGMENT_LENGTH
            audio_start = random.randint(0, max_audio_start)
            segment = wave[audio_start : audio_start + SEGMENT_LENGTH]
        else:
            diff = SEGMENT_LENGTH - wave.shape[0]
            segment = np.pad(wave, (0, diff))

        return segment
