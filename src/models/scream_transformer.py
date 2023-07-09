import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model, metrics

from src.config import *

class ScreamEncoder(layers.Layer):

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

class ScreamDecoder(layers.Layer):

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

class ScreamTransformer(Model):

    def __init__(self, classifier=None, detector=None):

        super().__init__()

        self.classifier = classifier
        if self.classifier is not None:
            self.classifier.trainable = False

        self.detector = detector
        if self.detector is not None:
            self.detector.trainable = False

        self.encoder = ScreamEncoder()
        self.decoder = ScreamDecoder()

        self.flatten_layer = layers.Flatten()
        self.cosine_similarity = layers.Dot(axes=(1), normalize=True)


        self.total_loss_tracker = metrics.Mean(name="loss")
        self.scream_loss_tracker = metrics.Mean(name="scream")
        self.correlation_loss_tracker = metrics.Mean(name="correlation")
        self.denoise_loss_tracker = metrics.Mean(name="denoise")

    @tf.function
    def scream_loss(self, output):

        pred = self.classifier(output)

        return 1.0 - pred

    @tf.function
    def denoise_loss(self, output):

        pred = self.detector(output)

        return pred

    @tf.function
    def correlation_loss(self, input, output):

        embedding_input = self.flatten_layer(input)
        embedding_output = self.flatten_layer(output)

        correlation = self.cosine_similarity([embedding_input, embedding_output])

        norm_correlation =  (2.0 - (correlation + 1.0)) / 2.0

        return norm_correlation

    @tf.function
    def call(self, x, training=False):

        encoded_x = self.encoder(x)
        decoded_x = self.decoder(encoded_x)

        if training:
            loss_scream = self.scream_loss(decoded_x)
            loss_corr = self.correlation_loss(x, decoded_x)
            loss_deno = self.denoise_loss(decoded_x)

            return decoded_x, loss_scream, loss_corr, loss_deno

        else:
            return decoded_x

    @tf.function
    def train_step(self, input):

        with tf.GradientTape() as tape:

            output, loss_scream, loss_corr, loss_deno = self(input, training=True)

            #Weight losses
            loss_scream = loss_scream * WEIGHT_SCREAM_LOSS
            loss_corr = loss_corr * WEIGHT_CORRELATION_LOSS
            loss_deno = loss_deno * WEIGHT_DENOISE_LOSS

            total_loss = loss_scream + loss_corr + loss_deno

            loss = tf.math.reduce_mean(total_loss)

        grads = tape.gradient(loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        self.total_loss_tracker.update_state(loss)
        self.scream_loss_tracker.update_state(tf.math.reduce_mean(loss_scream)/WEIGHT_SCREAM_LOSS)
        self.correlation_loss_tracker.update_state(tf.math.reduce_mean(loss_corr)/WEIGHT_CORRELATION_LOSS)
        self.denoise_loss_tracker.update_state(tf.math.reduce_mean(loss_deno)/WEIGHT_DENOISE_LOSS)

        return {
            "loss": self.total_loss_tracker.result(),
            "scream": self.scream_loss_tracker.result(),
            "correlation": self.correlation_loss_tracker.result(),
            "denoise": self.denoise_loss_tracker.result(),
        }
