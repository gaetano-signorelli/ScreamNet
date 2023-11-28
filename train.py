import os
import numpy as np
import librosa
import soundfile as sf
from tqdm import tqdm

from src.utils.file_manager import get_files_names

import tensorflow as tf
from tensorflow.keras import Input, callbacks

from src.model.loader import DatasetLoader
from src.model.gan import ScreamGAN

from src.config import *

RANDOM_SEED = 24
COMPLETED = 0

SAVE_WEIGHTS = True
SAVE_SAMPLES = True
LOAD_WEIGHTS = True

CHECKPOINTS_DISTANCE = 50

class SaveCallback(callbacks.Callback):

    def __init__(self, save_weights, save_samples, cp_distance):

        super().__init__()

        self.save_weights = save_weights
        self.save_samples = save_samples
        self.checkpoint_distance = cp_distance
        self.test_samples = get_files_names(os.path.join(TRAINING_PATH, "test"))

    def on_epoch_end(self, epoch, logs=None):

        if (epoch+1) % self.checkpoint_distance == 0:

            if self.save_weights:
                print()
                print("Saving checkpoint...")
                self.model.save_weights(WEIGHTS_PATH)

            if self.save_samples:
                print("Generating and saving samples...")
                for test_name in self.test_samples:
                    self.__save_sample(test_name, epoch+COMPLETED+1)
                print("Samples saved successfully!")

    def __save_sample(self, file_name, epoch):

        #Load file with librosa
        input_path = os.path.join(TRAINING_PATH, "test", file_name + ".mp3")
        original_wave, sr = librosa.load(input_path, sr=SAMPLING_RATE)

        #Split in 1sec segments (22050 samples)
        diff = SEGMENT_LENGTH - (original_wave.shape[0] % SEGMENT_LENGTH)
        original_wave = np.pad(original_wave, (0, diff))
        n_chunks = original_wave.shape[0] // SEGMENT_LENGTH
        segments = np.split(original_wave, n_chunks)
        batch = np.array(segments)

        #Predict (transform to screams)
        results = self.model.generator.predict(batch)

        #Concatenate (flatten)
        scream = np.concatenate(results)

        #Save
        output_path = os.path.join(TRAIN_SAMPLES_PATH, file_name + "_" + str(epoch) + ".mp3")
        sf.write(output_path, scream, SAMPLING_RATE, format="mp3")

if __name__ == '__main__':

    tf.keras.utils.set_random_seed(RANDOM_SEED)

    input_shape = (SEGMENT_LENGTH)
    warmup_input = Input(shape=input_shape)

    model = ScreamGAN(use_weight_norm=USE_WEIGHT_NORMALIZATION)
    model(warmup_input)

    model.compile(run_eagerly=RUN_EAGERLY)
    model.summary()

    if LOAD_WEIGHTS:
        if os.path.exists(WEIGHTS_PATH + ".index"):
            print("Loading model's weights...")
            model.load_weights(WEIGHTS_PATH)
            print("Model's weights successfully loaded!")

        else:
            print("WARNING: model's weights not found, the model will be executed with initialized random weights.")
            print("Ignore this warning if it is a test, or the first training.")

    dataset_loader = DatasetLoader()

    save_callback = SaveCallback(SAVE_WEIGHTS, SAVE_SAMPLES, CHECKPOINTS_DISTANCE)

    history = model.fit(x=dataset_loader, callbacks=[save_callback], epochs=EPOCHS)

    if SAVE_WEIGHTS:
        model.save_weights(WEIGHTS_PATH)

    print("Train completed!")
