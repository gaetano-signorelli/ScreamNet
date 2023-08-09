import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model, activations, metrics, mixed_precision
from tensorflow.keras.optimizers import Adam

from src.model.generator import Generator
from src.model.discriminator import Discriminator

from src.config import *

class ScreamGAN(Model):

    def __init__(self, tflite=False, use_weight_norm=True):

        super().__init__()

        self.discriminator_optimizer=Adam(LEARNING_RATE, beta_1=BETA_1, beta_2=BETA_2)
        self.generator_optimizer=Adam(LEARNING_RATE, beta_1=BETA_1, beta_2=BETA_2)

        self.generator = Generator(tflite, use_weight_norm)

        if USE_MIXED_PRECISION:
            mixed_precision.set_global_policy('mixed_float16')

        self.discriminator = Discriminator(use_weight_norm)

        self.cosine_similarity = layers.Dot(axes=(1), normalize=True, dtype='float32')
        self.flatten_layer = layers.Flatten(dtype='float32')

        self.gen_loss_tracker = metrics.Mean(name="generator")
        self.corr_loss_tracker = metrics.Mean(name="correlation")
        self.total_gen_loss_tracker = metrics.Mean(name="total generator")
        self.disc_loss_tracker = metrics.Mean(name="discriminator")

    @tf.function
    def call(self, x):

        wave = self.generator(x)
        output = self.discriminator(wave)

        return wave, output

    @tf.function
    def __generator_loss(self, scores_fake):

        score_fake_1 = -tf.math.reduce_mean(scores_fake[0])
        score_fake_2 = -tf.math.reduce_mean(scores_fake[1])
        score_fake_3 = -tf.math.reduce_mean(scores_fake[2])

        loss = score_fake_1 + score_fake_2 + score_fake_3

        return loss

    @tf.function
    def __correlation_loss(self, original_waves, generated_waves):

        original_waves = self.generator.mel_layer(original_waves)
        generated_waves = self.generator.mel_layer(generated_waves)

        original_waves = self.flatten_layer(original_waves)
        generated_waves = self.flatten_layer(generated_waves)

        correlation = self.cosine_similarity([original_waves, generated_waves])
        correlation = tf.math.reduce_mean(correlation)

        loss =  (2.0 - (correlation + 1.0)) / 2.0

        return loss

    @tf.function
    def __discriminator_loss(self, scores_fake, scores_real):

        score_real_1 = tf.math.reduce_mean(activations.relu(1 - scores_real[0]))
        score_real_2 = tf.math.reduce_mean(activations.relu(1 - scores_real[1]))
        score_real_3 = tf.math.reduce_mean(activations.relu(1 - scores_real[2]))

        loss_real = score_real_1 + score_real_2 + score_real_3

        score_fake_1 = tf.math.reduce_mean(activations.relu(1 + scores_fake[0]))
        score_fake_2 = tf.math.reduce_mean(activations.relu(1 + scores_fake[1]))
        score_fake_3 = tf.math.reduce_mean(activations.relu(1 + scores_fake[2]))

        loss_fake = score_fake_1 + score_fake_2 + score_fake_3

        loss = loss_real + loss_fake

        return loss

    @tf.autograph.experimental.do_not_convert
    def train_step(self, data):

        whispers, screams = data

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_waves = self.generator(whispers)

            scores_fake = self.discriminator(generated_waves)
            scores_real = self.discriminator(screams)

            gen_loss = self.__generator_loss(scores_fake)
            corr_loss = self.__correlation_loss(whispers, generated_waves)

            total_gen_loss = gen_loss + CORRELATION_WEIGHT * corr_loss

            disc_loss = self.__discriminator_loss(scores_fake, scores_real)

        gradients_of_discriminator = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)
        gradients_of_generator = gen_tape.gradient(total_gen_loss, self.generator.trainable_variables)

        self.discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, self.discriminator.trainable_variables))
        self.generator_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))

        self.gen_loss_tracker.update_state(gen_loss)
        self.corr_loss_tracker.update_state(corr_loss)
        self.total_gen_loss_tracker.update_state(total_gen_loss)
        self.disc_loss_tracker.update_state(disc_loss)

        return {
            "generator": self.gen_loss_tracker.result(),
            "correlation": self.corr_loss_tracker.result(),
            "total generator": self.total_gen_loss_tracker.result(),
            "discriminator": self.disc_loss_tracker.result()
        }
