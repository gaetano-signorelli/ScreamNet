import threading

import random
import librosa
import numpy as np
import tensorflow as tf
from tensorflow import keras

from src.config import *

class DatasetLoader(keras.utils.Sequence):

    def __init__(self):

        print("Loading files...")

        whispers_files = self.__get_files(WHISPERS_PATH)
        screams_files = self.__get_files(SCREAMS_PATH)

        self.whispers = [self.__load_audio(whisper_file) for whisper_file in whispers_files]
        self.screams = [self.__load_audio(scream_file) for scream_file in screams_files]

        print("Files loaded!")

    def on_epoch_end(self):
        pass

    def __getitem__(self, idx):

        batch_whispers = []
        batch_screams = []

        threads = []

        for _ in range(BATCH_SIZE):

            t = threading.Thread(target=self.__prepare_batch_sample, args=(batch_whispers, batch_screams))
            threads.append(t)
            t.start()

        for thread in threads:
            thread.join()

        batch_whispers = np.array(batch_whispers)
        batch_screams = np.array(batch_screams)

        return batch_whispers, batch_screams

    def __len__(self):

        return EPOCH_LEN

    def __prepare_batch_sample(self, batch_whispers, batch_screams):

        whisper_audio = random.choice(self.whispers)
        scream_audio = random.choice(self.screams)

        batch_whispers.append(self.__load_segment(whisper_audio))
        batch_screams.append(self.__load_segment(scream_audio))

    def __get_files(self, root):

        paths = []

        for dirname, dirnames, filenames in os.walk(root):
            for filename in filenames:
                paths.append(os.path.join(dirname, filename))

        return paths

    def __load_audio(self, audio_file):

        wave, sampling_rate = librosa.load(audio_file, sr=SAMPLING_RATE)

        if AUGMENT:
            amplitude = np.random.uniform(low=0.3, high=1.0)
            wave = wave * amplitude

        return wave

    def __load_segment(self, wave):

        if wave.shape[0] >= SEGMENT_LENGTH:
            max_audio_start = wave.shape[0] - SEGMENT_LENGTH
            audio_start = random.randint(0, max_audio_start)
            segment = wave[audio_start : audio_start + SEGMENT_LENGTH]
        else:
            diff = SEGMENT_LENGTH - wave.shape[0]
            segment = np.pad(wave, (0, diff))

        return segment
