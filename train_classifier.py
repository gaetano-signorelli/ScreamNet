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

from src.models.scream_classifier import ScreamClassifier
from src.config import *

SAVE_WEIGHTS = True
LOAD_WEIGHTS = False

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

    sings, sings_labels = load_dataset_class(os.path.join(CLASSIFICATION_TRAIN_PATH,"Sings"), 0)
    whisps, whisps_labels = load_dataset_class(os.path.join(CLASSIFICATION_TRAIN_PATH,"Whispers"), 0)
    screams, scream_labels = load_dataset_class(os.path.join(CLASSIFICATION_TRAIN_PATH,"Screams"), 1)

    x = np.array(sings + whisps + screams)
    y = np.array(sings_labels + whisps_labels + scream_labels)

    print("{} samples in the dataset".format(len(x)))

    return x, y

if __name__ == '__main__':

    input_shape = (128, N_FRAMES, 1)
    warmup_input = Input(shape=input_shape)

    model = ScreamClassifier()
    model(warmup_input)

    model.compile(optimizer=Adam(CLASSIFICATION_LEARNING_RATE),
                loss=BinaryCrossentropy(),
                metrics = [BinaryAccuracy()],
                run_eagerly=CLASSIFICATION_RUN_EAGERLY)

    model.summary()

    if LOAD_WEIGHTS:
        if os.path.exists(CLASSIFICATION_WEIGHTS_PATH + ".index"):
            print("Loading model's weights...")
            model.load_weights(CLASSIFICATION_WEIGHTS_PATH)
            print("Model's weights successfully loaded!")

        else:
            print("WARNING: model's weights not found, the model will be executed with initialized random weights.")
            print("Ignore this warning if it is a test, or the first training.")

    x, y = load_dataset()
    early_stopping = EarlyStopping(monitor='loss', patience=5)

    history = model.fit(x=x,
                        y=y,
                        batch_size=CLASSIFICATION_BATCH_SIZE,
                        epochs=CLASSIFICATION_EPOCHS,
                        callbacks=[early_stopping],
                        validation_split=CLASSIFICATION_VALIDATION_SPLIT,
                        shuffle=True,
                        class_weight=None)

    if SAVE_WEIGHTS:
        model.save_weights(CLASSIFICATION_WEIGHTS_PATH)
