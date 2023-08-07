import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model

from src.model.mel import WavToMel

from src.utils.weight_normalization import WeightNormalization

from src.config import *

class ResidualLayer(layers.Layer):

    def __init__(self, dim, dilation=1):

        super().__init__()

        self.paddings = tf.constant([[0, 0], [0, 0], [dilation, dilation]])

        self.dilated_conv = layers.Conv1D(filters=dim,
                                        kernel_size=3,
                                        strides=1,
                                        data_format="channels_first",
                                        dilation_rate=dilation,
                                        activation=layers.LeakyReLU(0.2))

        self.feed_forward_1 = layers.Conv1D(filters=dim,
                                        kernel_size=1,
                                        strides=1,
                                        data_format="channels_first")

        self.feed_forward_2 = layers.Conv1D(filters=dim,
                                        kernel_size=1,
                                        strides=1,
                                        data_format="channels_first")

        self.leaky_layer = layers.LeakyReLU(0.2)

        if USE_WEIGHT_NORMALIZATION:
            self.dilated_conv = WeightNormalization(self.dilated_conv)
            self.feed_forward_1 = WeightNormalization(self.feed_forward_1)
            self.feed_forward_2 = WeightNormalization(self.feed_forward_2)

    def call(self, x):

        dilated_x = tf.pad(x, self.paddings, "REFLECT")
        dilated_x = self.dilated_conv(dilated_x)
        dilated_x = self.feed_forward_1(dilated_x)

        x = self.feed_forward_2(x)

        res_out = dilated_x + x
        res_out = self.leaky_layer(res_out)

        return res_out

class Generator(Model):

    def __init__(self, tflite=False):

        super().__init__()

        self.mel_layer = WavToMel(tflite)

        self.upsample_rates = [8,8,2,2]
        self.n_residual_layers = 3

        n_filters = 512

        self.paddings_start = tf.constant([[0, 0], [0, 0], [3, 3]])
        self.paddings_end = tf.constant([[0, 0], [0, 0], [0, 34]])

        self.conv_1 = layers.Conv1D(filters=n_filters,
                                    kernel_size=7,
                                    strides=1,
                                    data_format="channels_first",
                                    activation=layers.LeakyReLU(0.2))

        n_filters = n_filters // 2

        self.upsample_layers = []
        self.residual_layers = []

        for rate in self.upsample_rates:
            upsample_layer = layers.Conv1DTranspose(filters=n_filters,
                                                    kernel_size=rate*2,
                                                    strides=rate,
                                                    padding="same",
                                                    data_format="channels_first",
                                                    activation=layers.LeakyReLU(0.2))

            if USE_WEIGHT_NORMALIZATION:
                upsample_layer = WeightNormalization(upsample_layer)

            self.upsample_layers.append(upsample_layer)

            for i in range(self.n_residual_layers):
                residual_layer = ResidualLayer(n_filters, dilation=3**i)
                self.residual_layers.append(residual_layer)

            n_filters = n_filters // 2

        self.conv_2 = layers.Conv1D(filters=1,
                                    kernel_size=7,
                                    strides=1,
                                    data_format="channels_first",
                                    activation="tanh")

        if USE_WEIGHT_NORMALIZATION:
            self.conv_1 = WeightNormalization(self.conv_1)
            self.conv_2 = WeightNormalization(self.conv_2)

    def call(self, x):

        x = self.mel_layer(x)

        x = tf.pad(x, self.paddings_start, "REFLECT")
        x = self.conv_1(x)

        i = 0
        for upsample_layer in self.upsample_layers:
            x = upsample_layer(x)
            for _ in range(self.n_residual_layers):
                x = self.residual_layers[i](x)
                i += 1

        x = tf.pad(x, self.paddings_start, "REFLECT")
        x = self.conv_2(x)
        x = tf.pad(x, self.paddings_end, "REFLECT")

        x = tf.squeeze(x, axis=1)

        return x
