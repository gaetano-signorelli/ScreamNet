import os
import numpy as np
from PIL import Image
from tqdm import tqdm

import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.metrics import BinaryAccuracy
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import Input

from src.models.noise_detector import NoiseDetector
from src.config import *

SAVE_WEIGHTS = True
LOAD_WEIGHTS = False
RANDOM_SEED = 24

def load_dataset_class(root, class_value):

    samples = []

    print("Loading files from " + root)

    for dirname, dirnames, filenames in os.walk(root):
        for filename in tqdm(filenames):
            file_path = os.path.join(dirname, filename)
            sample = Image.open(file_path)
            sample = np.expand_dims(sample, axis=-1)
            samples.append(sample)

    labels = [class_value] * len(samples)

    return samples, labels

def load_dataset():

    screams, scream_labels = load_dataset_class(os.path.join(DETECTOR_TRAIN_PATH,"Screams"), 0)
    whispers, whisper_labels = load_dataset_class(os.path.join(DETECTOR_TRAIN_PATH,"Whispers"), 0)
    noise, noise_labels = load_dataset_class(os.path.join(DETECTOR_TRAIN_PATH,"Noisy screams"), 1)

    x = np.array(screams + whispers + noise)
    y = np.array(scream_labels + whisper_labels + noise_labels)

    if DETECTOR_NORMALIZE:
        max_val = np.max(x)
        x = x / max_val

    print("{} samples in the dataset".format(len(x)))

    return x, y

if __name__ == '__main__':

    tf.keras.utils.set_random_seed(RANDOM_SEED)

    input_shape = (128, N_FRAMES, 1)
    warmup_input = Input(shape=input_shape)

    model = NoiseDetector()
    model(warmup_input)

    model.compile(optimizer=Adam(DETECTOR_LEARNING_RATE),
                loss=BinaryCrossentropy(),
                metrics = [BinaryAccuracy()],
                run_eagerly=DETECTOR_RUN_EAGERLY)

    model.summary()

    if LOAD_WEIGHTS:
        if os.path.exists(DETECTOR_WEIGHTS_PATH + ".index"):
            print("Loading model's weights...")
            model.load_weights(DETECTOR_WEIGHTS_PATH)
            print("Model's weights successfully loaded!")

        else:
            print("WARNING: model's weights not found, the model will be executed with initialized random weights.")
            print("Ignore this warning if it is a test, or the first training.")

    x, y = load_dataset()
    early_stopping = EarlyStopping(monitor='loss', patience=5)

    history = model.fit(x=x,
                        y=y,
                        batch_size=DETECTOR_BATCH_SIZE,
                        epochs=DETECTOR_EPOCHS,
                        callbacks=[early_stopping],
                        validation_split=DETECTOR_VALIDATION_SPLIT,
                        shuffle=True,
                        class_weight={0:1.8, 1:1.0})

    if SAVE_WEIGHTS:
        model.save_weights(DETECTOR_WEIGHTS_PATH)
