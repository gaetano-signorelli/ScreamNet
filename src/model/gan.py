import numpy as np
import tensorflow as tf
from tf.keras import layers, Model, activations

from src.model.generator import Generator
from src.model.discriminator import Discriminator

from src.config import *

class ScreamGAN(Model):

    def __init__(self):

        super().__init__()

        self.generator = Generator()
        self.discriminator = Discriminator()

        self.cosine_similarity = layers.Dot(axes=(1), normalize=True)

        self.gen_loss_tracker = metrics.Mean(name="generator")
        self.corr_loss_tracker = metrics.Mean(name="correlation")
        self.total_gen_loss_tracker = metrics.Mean(name="total generator")
        self.disc_loss_tracker = metrics.Mean(name="discriminator")

    def call(self, x):

        wave = self.generator(x)
        output = self.discriminator(wave)

        return output

    @tf.function
    def __generator_loss(self, scores_fake):

        loss = 0
        for scores in scores_fake:
            loss -= tf.math.reduce_mean(scores)

        return loss

    @tf.function
    def __correlation_loss(self, original_waves, generated_waves):

        correlation = self.cosine_similarity([original_waves, generated_waves])
        correlation = tf.math.reduce_mean(correlation)

        loss =  (2.0 - (correlation + 1.0)) / 2.0

        return loss

    @tf.function
    def __discriminator_loss(self, scores_fake, scores_real):

        loss_real = 0
        for scores in scores_real:
            norm_scores = activations.relu(1 - scores)
            loss_real += tf.math.reduce_mean(norm_scores)

        loss_fake = 0
        for scores in scores_fake:
            norm_scores = activations.relu(1 + scores)
            loss_fake += tf.math.reduce_mean(norm_scores)

        loss = loss_real + loss_fake

        return loss

    @tf.function
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

        discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, self.discriminator.trainable_variables))
        generator_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))

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
