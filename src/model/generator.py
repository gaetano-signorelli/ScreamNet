import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model

from src.model.mel import WavToMel

from src.utils.weight_normalization import WeightNormalization
from src.utils.conv_1d_transpose import TFConvTranspose1d

from src.config import *

class NormalizationLayer(layers.Layer):

    def __init__(self):

        super().__init__()

    @tf.function
    def call(self, x):

        max_values = tf.math.reduce_max(tf.math.abs(x), axis=-1, keepdims=True)

        x = tf.math.divide(x, max_values)
        x = x * 0.95 # *0.95 comes from the official MelGan implementation

        return x

class ResidualLayer(layers.Layer):

    def __init__(self, use_weight_norm, dim, dilation=1):

        super().__init__()

        self.paddings = tf.constant([[0, 0], [dilation, dilation], [0, 0]])

        self.dilated_conv = layers.Conv1D(filters=dim,
                                        kernel_size=3,
                                        strides=1,
                                        data_format="channels_last",
                                        dilation_rate=dilation,
                                        activation=layers.LeakyReLU(0.2))

        self.feed_forward_1 = layers.Conv1D(filters=dim,
                                        kernel_size=1,
                                        strides=1,
                                        data_format="channels_last")

        self.feed_forward_2 = layers.Conv1D(filters=dim,
                                        kernel_size=1,
                                        strides=1,
                                        data_format="channels_last")

        self.leaky_layer = layers.LeakyReLU(0.2)

        if use_weight_norm:
            self.dilated_conv = WeightNormalization(self.dilated_conv)
            self.feed_forward_1 = WeightNormalization(self.feed_forward_1)
            self.feed_forward_2 = WeightNormalization(self.feed_forward_2)

    @tf.function
    def call(self, x):

        dilated_x = tf.pad(x, self.paddings, "REFLECT")
        dilated_x = self.dilated_conv(dilated_x)
        dilated_x = self.feed_forward_1(dilated_x)

        x = self.feed_forward_2(x)

        res_out = dilated_x + x
        res_out = self.leaky_layer(res_out)

        return res_out

class Generator(Model):

    def __init__(self, tflite, use_weight_norm):

        super().__init__()

        self.norm_layer = NormalizationLayer()

        self.mel_layer = WavToMel(tflite)

        self.upsample_rates = [8,8,2,2]
        self.n_residual_layers = 3

        n_filters = 512

        self.paddings_start = tf.constant([[0, 0], [3, 3], [0, 0]])
        self.paddings_end = tf.constant([[0, 0], [0, 34], [0, 0]])

        self.conv_1 = layers.Conv1D(filters=n_filters,
                                    kernel_size=7,
                                    strides=1,
                                    data_format="channels_last",
                                    activation=layers.LeakyReLU(0.2))

        n_filters = n_filters // 2

        self.upsample_layers = []
        self.residual_layers = []

        for rate in self.upsample_rates:
            upsample_layer = TFConvTranspose1d(filters=n_filters,
                                                kernel_size=rate*2,
                                                strides=rate,
                                                padding="same",
                                                data_format="channels_last",
                                                activation=layers.LeakyReLU(0.2),
                                                use_weight_norm=use_weight_norm)

            self.upsample_layers.append(upsample_layer)

            for i in range(self.n_residual_layers):
                residual_layer = ResidualLayer(use_weight_norm, n_filters, dilation=3**i)
                self.residual_layers.append(residual_layer)

            n_filters = n_filters // 2

        self.conv_2 = layers.Conv1D(filters=1,
                                    kernel_size=7,
                                    strides=1,
                                    data_format="channels_last",
                                    activation="tanh")

        if use_weight_norm:
            self.conv_1 = WeightNormalization(self.conv_1)
            self.conv_2 = WeightNormalization(self.conv_2)

    @tf.function
    def call(self, x):

        x = self.norm_layer(x)

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

        x = tf.squeeze(x, axis=-1)

        return x
