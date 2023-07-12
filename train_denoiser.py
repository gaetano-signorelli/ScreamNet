import os
import numpy as np
from PIL import Image
from tqdm import tqdm

import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredLogarithmicError
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import Input

from src.models.denoise_transformer import DenoiseTransformer
from src.config import *

SAVE_WEIGHTS = True
LOAD_WEIGHTS = False
RANDOM_SEED = 24

def load_dataset_class(root):

    samples = []

    print("Loading files from " + root)

    for dirname, dirnames, filenames in os.walk(root):
        for filename in tqdm(filenames):
            file_path = os.path.join(dirname, filename)
            sample = Image.open(file_path)
            sample = np.expand_dims(sample, axis=-1)
            samples.append(sample)

    return samples

def load_dataset():

    x = load_dataset_class(DENOISER_TRAIN_PATH_X)
    y = load_dataset_class(DENOISER_TRAIN_PATH_Y)

    x = np.array(x)
    y = np.array(y)

    assert len(x)==len(y)

    if DETECTOR_NORMALIZE:
        max_val_x = np.max(x)
        max_val_y = np.max(y)
        max_val = max(max_val_x, max_val_y)
        x = x / max_val
        y = y / max_val

    print("{} samples in the dataset".format(len(x)))

    return x, y


if __name__ == '__main__':

    tf.keras.utils.set_random_seed(RANDOM_SEED)

    input_shape = (128, N_FRAMES, 1)
    warmup_input = Input(shape=input_shape)

    model = DenoiseTransformer()
    model(warmup_input)

    model.compile(optimizer=Adam(DENOISER_LEARNING_RATE),
                loss=MeanSquaredLogarithmicError(),
                run_eagerly=DENOISER_RUN_EAGERLY)

    model.summary()

    if LOAD_WEIGHTS:
        if os.path.exists(DENOISER_WEIGHTS_PATH + ".index"):
            print("Loading model's weights...")
            model.load_weights(DENOISER_WEIGHTS_PATH)
            print("Model's weights successfully loaded!")

        else:
            print("WARNING: model's weights not found, the model will be executed with initialized random weights.")
            print("Ignore this warning if it is a test, or the first training.")

    x, y = load_dataset()
    early_stopping = EarlyStopping(monitor='loss', patience=3)

    history = model.fit(x=x,
                        y=y,
                        batch_size=DENOISER_BATCH_SIZE,
                        epochs=DENOISER_EPOCHS,
                        callbacks=[early_stopping],
                        validation_split=DENOISER_VALIDATION_SPLIT,
                        shuffle=True)

    if SAVE_WEIGHTS:
        model.save_weights(DENOISER_WEIGHTS_PATH)
