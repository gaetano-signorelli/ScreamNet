import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model

class NoiseDetector(Model):

    def __init__(self):

        super().__init__()

        self.global_avg_pool = layers.GlobalAveragePooling2D()
        self.flatten = layers.Flatten()

        self.conv_1 = layers.Conv2D(32,
                                    kernel_size=5,
                                    padding="same",
                                    strides=(4,2),
                                    activation=layers.LeakyReLU())

        self.conv_2 = layers.Conv2D(64,
                                    kernel_size=5,
                                    padding="same",
                                    strides=(4,2),
                                    activation=layers.LeakyReLU())

        self.conv_3 = layers.Conv2D(16,
                                    kernel_size=3,
                                    padding="same",
                                    activation=layers.LeakyReLU())

        self.conv_4 = layers.Conv2D(32,
                                    kernel_size=3,
                                    padding="same",
                                    strides=2,
                                    activation=layers.LeakyReLU())

        self.conv_5 = layers.Conv2D(16,
                                    kernel_size=3,
                                    padding="same",
                                    activation=layers.LeakyReLU())

        self.conv_6 = layers.Conv2D(32,
                                    kernel_size=3,
                                    padding="same",
                                    strides=2,
                                    activation=layers.LeakyReLU())

        self.conv_7 = layers.Conv2D(16,
                                    kernel_size=3,
                                    padding="same",
                                    activation=layers.LeakyReLU())

        self.classification_head = layers.Dense(1, activation="sigmoid")

    def call(self, x):

        x = self.conv_1(x)
        x = self.conv_2(x)
        x = layers.Dropout(0.25)(x)
        x = self.conv_3(x)
        x = layers.Dropout(0.25)(x)
        x = self.conv_4(x)
        x = layers.Dropout(0.25)(x)
        x = self.conv_5(x)
        x = layers.Dropout(0.25)(x)
        x = self.conv_6(x)
        x = layers.Dropout(0.25)(x)
        x = self.conv_7(x)

        x = self.global_avg_pool(x)

        x = self.flatten(x)

        x = self.classification_head(x)

        return x
