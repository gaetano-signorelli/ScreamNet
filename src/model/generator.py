import numpy as np
import tensorflow as tf
from tf.keras import layers, Model

from src.config import *

class ResidualLayer(layers.Layer):

    def __init__(self, dim, dilation=1):

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

    def call(self, x):

        dilated_x = self.dilated_conv(x)
        dilated_x = self.feed_forward_1(dilated_x)

        x = self.feed_forward_2(x)

        res_out = dilated_x + x
        res_out = self.leaky_layer(res_out)

        return res_out

class Generator(Model):

    def __init__(self):

        super().__init__()

        upsample_rates = [8,8,2,2]
        self.n_residual_layers = 3

        n_filters = 512

        self.conv_1 = layers.Conv1D(filters=n_filters,
                                    kernel_size=7,
                                    strides=1,
                                    data_format="channels_first",
                                    activation=layers.LeakyReLU(0.2))

        n_filters = n_filters // 2

        self.upsample_layers = []
        self.residual_layers = []

        for rate in upsample_rates:
            upsample_layer = layers.Conv1DTranspose(filters=n_filters,
                                                    kernel_size=rate*2,
                                                    strides=rate,
                                                    data_format="channels_first",
                                                    activation=layers.LeakyReLU(0.2))
            upsample_layers.append(upsample_layer)

            for i in range(self.n_residual_layers):
                residual_layer = ResidualLayer(n_filters, dilation=3**i)
                residual_layers.append(residual_layer)

            n_filters = n_filters // 2

        self.conv_2 = layers.Conv1D(filters=1,
                                    kernel_size=7,
                                    strides=1,
                                    data_format="channels_first",
                                    activation="tanh")

    def call(self, x):

        #TODO: check for using reflection paddings

        x = self.conv_1(x)

        i = 0
        for upsample_layer in upsample_layers:
            x = upsample_layer(x)
            for _ in range(self.n_residual_layers):
                x = self.residual_layers[i](x)
                i += 1

        x = self.conv_2(x)

        return x
