import os
import numpy as np
import librosa
import soundfile as sf
from tqdm import tqdm

import tensorflow as tf
from tf.keras.optimizers import Adam
from tf.keras import Input, callbacks.Callback

from src.model.loader import DatasetLoader
from src.model.gan import ScreamGAN

from src.config import *

RANDOM_SEED = 24

SAVE_WEIGHTS = True
SAVE_SAMPLES = True
LOAD_WEIGHTS = False

CHECKPOINTS_DISTANCE = 50

TEST_SAMPLES = [
"test",
"test2",
"test3"
]

class SaveCallback(Callback):

    def __init__(self, save_weights, save_samples, cp_distance, test_samples):

        super().__init__()

        self.save_weights = save_weights
        self.save_samples = save_samples
        self.checkpoint_distance = cp_distance
        self.test_samples = test_samples

    def on_epoch_end(self, epoch, logs=None):

        if if epoch % self.checkpoint_distance == 0:

            if self.save_weights:
                print("Saving checkpoint...")
                self.model.save_weights(WEIGHTS_PATH)

            if self.save_samples:
                print("Generating and saving samples...")
                for test_name in self.test_samples:
                    self.save_sample(test_name, epoch)
                print("Samples saved successfully!")

    def __save_sample(self, file_name, epoch):

        #Load file with librosa
        input_path = os.path.join(WHISPERS_PATH, file_name + ".mp3")
        original_wave = librosa.load(input_path, sr=SAMPLING_RATE)

        #Normalize
        original_wave = librosa.util.normalize(wave) * 0.95

        #Split in 1sec segments (22050 samples)
        diff = original_wave.shape[0] % SEGMENT_LENGTH
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
        sf.write(output_path, scream, SAMPLING_RATE, "mp3")

if __name__ == '__main__':

    tf.keras.utils.set_random_seed(RANDOM_SEED)

    input_shape = (SEGMENT_LENGTH)
    warmup_input = Input(shape=input_shape)

    model = ScreamGAN()
    model(warmup_input)

    model.compile(optimizer=Adam(LEARNING_RATE), run_eagerly=CLASSIFICATION_RUN_EAGERLY)
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

    save_callback = SaveCallback(SAVE_WEIGHTS, SAVE_SAMPLES, CHECKPOINTS_DISTANCE, TEST_SAMPLES)

    history = model.fit(x=dataset_loader, callbacks=[save_callback], epochs=EPOCHS)

    if SAVE_WEIGHTS:
        model.save_weights(WEIGHTS_PATH)
