import numpy as np
import tensorflow as tf
from tf.keras import layers, Model

from src.config import *

class DiscriminatorBlock(layers.Layer):

    def __init__(self):

        super().__init__()

        self.conv_1 = layers.Conv1D(filters=16,
                                    kernel_size=15,
                                    strides=1,
                                    data_format="channels_first",
                                    activation=layers.LeakyReLU(0.2))

        self.downsample_1 = layers.Conv1D(filters=64,
                                        kernel_size=41,
                                        strides=4,
                                        groups=4,
                                        data_format="channels_first",
                                        activation=layers.LeakyReLU(0.2))

        self.downsample_2 = layers.Conv1D(filters=256,
                                        kernel_size=41,
                                        strides=4,
                                        groups=16,
                                        data_format="channels_first",
                                        activation=layers.LeakyReLU(0.2))

        self.downsample_3 = layers.Conv1D(filters=1024,
                                        kernel_size=41,
                                        strides=4,
                                        groups=64,
                                        data_format="channels_first",
                                        activation=layers.LeakyReLU(0.2))

        self.downsample_4 = layers.Conv1D(filters=1024,
                                        kernel_size=41,
                                        strides=4,
                                        groups=256,
                                        data_format="channels_first",
                                        activation=layers.LeakyReLU(0.2))

        self.conv_2 = layers.Conv1D(filters=1024,
                                    kernel_size=5,
                                    strides=1,
                                    data_format="channels_first",
                                    activation=layers.LeakyReLU(0.2))

        self.conv_3 = layers.Conv1D(filters=1,
                                    kernel_size=3,
                                    strides=1,
                                    data_format="channels_first")

    def call(self, x):

        #TODO: check for paddings

        x = self.conv_1(x)

        x = self.downsample_1(x)
        x = self.downsample_2(x)
        x = self.downsample_3(x)
        x = self.downsample_4(x)

        x = self.conv_2(x)
        x = self.conv_3(x)

        return x

class Discriminator(Model):

    def __init__(self):

        self.block_1 = DiscriminatorBlock()
        self.block_2 = DiscriminatorBlock()
        self.block_3 = DiscriminatorBlock()

        self.avg_pool_layer = layers.AveragePooling1D(pool_size=4,
                                                    strides=2,
                                                    data_format="channels_last")

    def call(self, x):

        score_1 = self.block_1(x)

        x = self.avg_pool_layer(x)
        score_2 = self.block_2(x)

        x = self.avg_pool_layer(x)
        score_3 = self.block_3(x)

        scores = [score_1, score_2, score_3]

        return scores
