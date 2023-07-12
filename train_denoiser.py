import os
import numpy as np

import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import Input

from src.models.denoise_transformer import DenoiseGenerator, DenoiseTransformer
from src.config import *

SAVE_WEIGHTS = True
LOAD_WEIGHTS = False
RANDOM_SEED = 24

if __name__ == '__main__':

    tf.keras.utils.set_random_seed(RANDOM_SEED)

    noise_path = os.path.join("data","Dataset","Noisy screams extended")
    normal_path = os.path.join("data","Dataset","Full screams extended")
    generator = DenoiseGenerator()

    input_shape = (128, N_FRAMES, 1)
    warmup_input = Input(shape=input_shape)

    model = DenoiseTransformer()
    model(warmup_input)

    model.compile(optimizer=Adam(DENOISER_LEARNING_RATE),
                loss=BinaryCrossentropy(),
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

    early_stopping = EarlyStopping(monitor='loss', patience=3)

    history = model.fit(generator,
                        callbacks=[early_stopping],
                        epochs=DENOISER_EPOCHS)

    if SAVE_WEIGHTS:
        model.save_weights(DENOISER_WEIGHTS_PATH)
