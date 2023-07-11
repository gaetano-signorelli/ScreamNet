import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model, metrics

from src.config import *

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

class DenoiseDecoder(layers.Layer):

    def __init__(self):

        super().__init__()

        self.conv_1 = layers.Conv2D(64,
                                    kernel_size=3,
                                    padding="same",
                                    activation=layers.LeakyReLU())

        self.conv_2 = layers.Conv2D(32,
                                    kernel_size=3,
                                    padding="same",
                                    activation=layers.LeakyReLU())

        self.conv_3 = layers.Conv2D(32,
                                    kernel_size=3,
                                    padding="same",
                                    activation=layers.LeakyReLU())

        self.conv_4 = layers.Conv2D(16,
                                    kernel_size=3,
                                    padding="same",
                                    activation=layers.LeakyReLU())

        self.conv_5 = layers.Conv2D(16,
                                    kernel_size=3,
                                    padding="same",
                                    activation=layers.LeakyReLU())

        self.conv_6 = layers.Conv2D(1,
                                    kernel_size=3,
                                    padding="same",
                                    activation=layers.LeakyReLU())

    def call(self, x):

        x = self.conv_1(x)
        x = layers.UpSampling2D()(x)

        x = self.conv_2(x)
        #x = layers.Dropout(0.25)(x)
        x = self.conv_3(x)
        x = layers.UpSampling2D(size=(4,2))(x)

        #x = layers.Dropout(0.25)(x)
        x = self.conv_4(x)
        #x = layers.Dropout(0.25)(x)
        x = self.conv_5(x)
        x = layers.UpSampling2D(size=(4,2))(x)

        #x = layers.Dropout(0.25)(x)
        x = self.conv_6(x)

        return x

class DenoiseTransformer(Model):

    def __init__(self):

        super().__init__()


        self.encoder = DenoiseEncoder()
        self.decoder = DenoiseDecoder()

    @tf.function
    def call(self, x, training=False):

        encoded_x = self.encoder(x)
        decoded_x = self.decoder(encoded_x)

        return decoded_x
