import tensorflow as tf
from tensorflow.keras import layers

from src.config import *

from src.utils.weight_normalization import WeightNormalization

class TFConvTranspose1d(layers.Layer):

    def __init__(
        self,
        filters,
        kernel_size,
        strides,
        padding,
        data_format,
        activation,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.conv1d_transpose = layers.Conv2DTranspose(
            filters=filters,
            kernel_size=(kernel_size, 1),
            strides=(strides, 1),
            padding=padding,
            data_format=data_format,
            activation=activation
        )

        if USE_WEIGHT_NORMALIZATION:
            self.conv1d_transpose = WeightNormalization(self.conv1d_transpose)

    def call(self, x):
        """Calculate forward propagation.
        Args:
            x (Tensor): Input tensor (B, T, C).
        Returns:
            Tensor: Output tensor (B, T', C').
        """

        x = tf.expand_dims(x, axis=2)
        x = self.conv1d_transpose(x)
        x = tf.squeeze(x, axis=2)

        return x
