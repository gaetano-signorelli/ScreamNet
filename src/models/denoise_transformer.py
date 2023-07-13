import random
import numpy as np
from PIL import Image
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras import layers, Model, metrics
from tensorflow.keras.utils import Sequence

from src.config import *

class DenoiseGenerator(Sequence):

    def __init__(self):

        self.noise_spectrograms = self.load_dataset_class(DENOISER_TRAIN_PATH_X)
        self.normal_spectrograms = self.load_dataset_class(DENOISER_TRAIN_PATH_Y)

        assert len(self.noise_spectrograms)==len(self.normal_spectrograms)

        self.n_spectrograms = len(self.noise_spectrograms)

    def load_dataset_class(self, root):

        samples = []

        print("Loading files from " + root)

        for dirname, dirnames, filenames in os.walk(root):
            for filename in tqdm(filenames):
                file_path = os.path.join(dirname, filename)
                sample = Image.open(file_path)
                sample = np.expand_dims(sample, axis=-1)
                samples.append(sample)

        return samples

    def on_epoch_end(self):
        pass

    def __getitem__(self, idx):

        x = []
        y = []

        while len(x) < DENOISER_BATCH_SIZE:
            random_image_index = random.randrange(self.n_spectrograms)
            random_image = self.noise_spectrograms[random_image_index]
            random_point = random.randrange(random_image.shape[1])

            sample_y = self.normal_spectrograms[random_image_index][:,random_point,:]
            #if (sample_y > 1).sum() / 128 < 0.05:
            #    continue

            start = random_point - (N_FRAMES//2)
            end = random_point + (N_FRAMES//2)
            pad_left = 0
            pad_right = 0

            if start<0:
                pad_left = -start
                start = 0
            if end>random_image.shape[1]:
                pad_right = end - random_image.shape[1]
                end = random_image.shape[1]

            random_frame = random_image[:, start:end, :]
            random_frame = np.pad(random_frame, ((0,0),(pad_left,pad_right),(0,0)))

            x.append(random_frame)
            y.append(sample_y)

        return np.array(x), np.array(y)

    def __len__(self):

        return DENOISER_EPOCH_LEN

class DenoiseEncoder(layers.Layer):

    def __init__(self):

        super().__init__()

        self.conv_1 = layers.Conv2D(16,
                                    kernel_size=3,
                                    padding="same",
                                    strides=(4,2),
                                    activation=layers.LeakyReLU())

        self.conv_2 = layers.Conv2D(16,
                                    kernel_size=3,
                                    padding="same",
                                    activation=layers.LeakyReLU())

        self.conv_3 = layers.Conv2D(32,
                                    kernel_size=3,
                                    padding="same",
                                    strides=(4,2),
                                    activation=layers.LeakyReLU())

        self.conv_4 = layers.Conv2D(32,
                                    kernel_size=3,
                                    padding="same",
                                    activation=layers.LeakyReLU())

        self.conv_5 = layers.Conv2D(64,
                                    kernel_size=3,
                                    padding="same",
                                    strides=2,
                                    activation=layers.LeakyReLU())

        self.conv_6 = layers.Conv2D(64,
                                    kernel_size=3,
                                    padding="same",
                                    activation=layers.LeakyReLU())

    def call(self, x):

        x = self.conv_1(x)
        x = self.conv_2(x)
        #x = layers.Dropout(0.25)(x)
        x = self.conv_3(x)
        #x = layers.Dropout(0.25)(x)
        x = self.conv_4(x)
        #x = layers.Dropout(0.25)(x)
        x = self.conv_5(x)
        #x = layers.Dropout(0.25)(x)
        x = self.conv_6(x)

        return x

class DenoiseTransformer(Model):

    def __init__(self):

        super().__init__()

        self.encoder = DenoiseEncoder()

        self.flatten_layer = layers.Flatten()
        self.dense_1 = layers.Dense(256, activation=layers.LeakyReLU())
        self.dense_2 = layers.Dense(128, activation="relu")

        self.cosine_similarity = layers.Dot(axes=(1), normalize=True)

        self.total_loss_tracker = metrics.Mean(name="loss")

    @tf.function
    def call(self, x):

        x = self.encoder(x)

        x = self.flatten_layer(x)
        x = self.dense_1(x)
        x = self.dense_2(x)

        x = tf.expand_dims(x, axis=-1)

        return x

    @tf.function
    def correlation_loss(self, input, output):

        correlation = self.cosine_similarity([input, output])

        norm_correlation =  (2.0 - (correlation + 1.0)) / 2.0

        return norm_correlation

    @tf.function
    def train_step(self, data):

        x, y = data

        with tf.GradientTape() as tape:

            pred = self(x)
            loss = self.correlation_loss(pred, y)
            loss = tf.math.reduce_mean(loss)

        grads = tape.gradient(loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        self.total_loss_tracker.update_state(loss)

        return {
            "loss": self.total_loss_tracker.result()
        }
