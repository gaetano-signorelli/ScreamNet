import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model

from src.utils.weight_normalization import WeightNormalization

from src.config import *

class DiscriminatorBlock(layers.Layer):

    def __init__(self, use_weight_norm):

        super().__init__()

        self.paddings = tf.constant([[0, 0], [7, 7], [0, 0]])

        self.conv_1 = layers.Conv1D(filters=16,
                                    kernel_size=15,
                                    strides=1,
                                    data_format="channels_last",
                                    activation=layers.LeakyReLU(0.2))

        self.downsample_1 = layers.Conv1D(filters=64,
                                        kernel_size=41,
                                        strides=4,
                                        groups=4,
                                        data_format="channels_last",
                                        padding="same",
                                        activation=layers.LeakyReLU(0.2))

        self.downsample_2 = layers.Conv1D(filters=256,
                                        kernel_size=41,
                                        strides=4,
                                        groups=16,
                                        data_format="channels_last",
                                        padding="same",
                                        activation=layers.LeakyReLU(0.2))

        self.downsample_3 = layers.Conv1D(filters=1024,
                                        kernel_size=41,
                                        strides=4,
                                        groups=64,
                                        data_format="channels_last",
                                        padding="same",
                                        activation=layers.LeakyReLU(0.2))

        self.downsample_4 = layers.Conv1D(filters=1024,
                                        kernel_size=41,
                                        strides=4,
                                        groups=256,
                                        data_format="channels_last",
                                        padding="same",
                                        activation=layers.LeakyReLU(0.2))

        self.conv_2 = layers.Conv1D(filters=1024,
                                    kernel_size=5,
                                    strides=1,
                                    data_format="channels_last",
                                    padding="same",
                                    activation=layers.LeakyReLU(0.2))

        self.conv_3 = layers.Conv1D(filters=1,
                                    kernel_size=3,
                                    strides=1,
                                    padding="same",
                                    data_format="channels_last")

        if use_weight_norm:
            self.conv_1 = WeightNormalization(self.conv_1)
            self.conv_2 = WeightNormalization(self.conv_2)
            self.conv_3 = WeightNormalization(self.conv_3)

            self.downsample_1 = WeightNormalization(self.downsample_1)
            self.downsample_2 = WeightNormalization(self.downsample_2)
            self.downsample_3 = WeightNormalization(self.downsample_3)
            self.downsample_4 = WeightNormalization(self.downsample_4)

    @tf.function
    def call(self, x):

        x = tf.expand_dims(x, axis=-1)

        x = tf.pad(x, self.paddings, "REFLECT")
        x = self.conv_1(x)

        x = self.downsample_1(x)
        x = self.downsample_2(x)
        x = self.downsample_3(x)
        x = self.downsample_4(x)

        x = self.conv_2(x)
        x = self.conv_3(x)

        x = tf.squeeze(x, axis=-1)

        return x

class Discriminator(Model):

    def __init__(self, use_weight_norm):

        super().__init__()

        self.block_1 = DiscriminatorBlock(use_weight_norm)
        self.block_2 = DiscriminatorBlock(use_weight_norm)
        self.block_3 = DiscriminatorBlock(use_weight_norm)

        self.avg_pool_layer = layers.AveragePooling1D(pool_size=4,
                                                    strides=2,
                                                    data_format="channels_last")

    @tf.function
    def call(self, x):

        score_1 = self.block_1(x)

        x = tf.expand_dims(x, axis=-1)
        x = self.avg_pool_layer(x)
        x = tf.squeeze(x, axis=-1)
        score_2 = self.block_2(x)

        x = tf.expand_dims(x, axis=-1)
        x = self.avg_pool_layer(x)
        x = tf.squeeze(x, axis=-1)
        score_3 = self.block_3(x)

        scores = [score_1, score_2, score_3]

        return scores
