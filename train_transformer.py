import os
import numpy as np
from PIL import Image
from tqdm import tqdm

import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import Input

from src.models.scream_classifier import ScreamClassifier
from src.models.scream_transformer import ScreamTransformer
from src.config import *

SAVE_WEIGHTS = True
LOAD_WEIGHTS = False
RANDOM_SEED = 24

def load_dataset(root):

    samples = []

    print("Loading files from " + root)

    for dirname, dirnames, filenames in os.walk(root):
        for filename in tqdm(filenames):
            file_path = os.path.join(dirname, filename)
            sample = Image.open(file_path)
            sample = np.expand_dims(sample, axis=-1)
            samples.append(sample)

    dataset = np.array(samples)

    if TRANSFORMATION_NORMALIZE:
        max_val = np.max(dataset)
        dataset = dataset / max_val

    return dataset

def load_classifier():

    classifier = ScreamClassifier()

    if os.path.exists(CLASSIFICATION_WEIGHTS_PATH + ".index"):
        classifier.load_weights(CLASSIFICATION_WEIGHTS_PATH)

    else:
        raise Exception("No weights found for classifiers: consider to train it first.")

    return classifier

if __name__ == '__main__':

    tf.keras.utils.set_random_seed(RANDOM_SEED)

    classifier = load_classifier()

    input_shape = (128, N_FRAMES, 1)
    warmup_input = Input(shape=input_shape)

    model = ScreamTransformer(classifier)
    model(warmup_input)

    model.compile(optimizer=Adam(TRANSFORMATION_LEARNING_RATE),
                run_eagerly=TRANSFORMATION_RUN_EAGERLY)

    model.summary()

    if LOAD_WEIGHTS:
        if os.path.exists(TRANSFORMATION_WEIGHTS_PATH + ".index"):
            print("Loading model's weights...")
            model.load_weights(TRANSFORMATION_WEIGHTS_PATH)
            print("Model's weights successfully loaded!")

        else:
            print("WARNING: model's weights not found, the model will be executed with initialized random weights.")
            print("Ignore this warning if it is a test, or the first training.")

    dataset = load_dataset()
    early_stopping = EarlyStopping(monitor='loss', patience=5)

    history = model.fit(x=dataset,
                        batch_size=TRANSFORMATION_BATCH_SIZE,
                        epochs=TRANSFORMATION_EPOCHS,
                        callbacks=[early_stopping],
                        shuffle=True)

    if SAVE_WEIGHTS:
        model.save_weights(TRANSFORMATION_WEIGHTS_PATH)
