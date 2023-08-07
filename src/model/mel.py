import numpy as np
import librosa
import tensorflow as tf
from tensorflow.keras import layers, Model
from kapre import STFT, Magnitude, STFTTflite, MagnitudeTflite

from src.config import *

class WavToMel(Model):

    def __init__(self, tflite=False):

        super().__init__()

        if not tflite:
            self.stft_layer = STFT(n_fft=N_FFT, win_length=WINDOW_SIZE, hop_length=HOP_LENGTH,
                                   window_name=None, pad_end=False, pad_begin=False,
                                   input_data_format='channels_last', output_data_format='channels_last')

            self.magnitude_layer = Magnitude()

        else:
            self.stft_layer = STFTTflite(n_fft=N_FFT, win_length=WINDOW_SIZE, hop_length=HOP_LENGTH,
                                   window_name=None, pad_end=False, pad_begin=False,
                                   input_data_format='channels_last', output_data_format='channels_last')

            self.magnitude_layer = MagnitudeTflite()

        mel_filters = librosa.filters.mel(sr=SAMPLING_RATE, n_fft=N_FFT, n_mels=MEL_CHANNELS)
        self.mel_basis = tf.convert_to_tensor(mel_filters, dtype=tf.float32)
        self.mel_basis = tf.expand_dims(self.mel_basis, axis=0)

        self.dot_layer = layers.Dot(axes=(1,2))

    def call(self, x):

        batch_size = tf.shape(x)[0]
        mel_basis = tf.repeat(self.mel_basis, repeats=[batch_size], axis=0)

        p = (N_FFT - HOP_LENGTH) // 2
        padding = tf.constant([[0, 0], [p, p]])
        x = tf.pad(x, padding, "REFLECT")

        x = tf.expand_dims(x, axis=-1) #Add channel (mono)

        x = self.stft_layer(x) #Compure Short Fourier Transform
        x = self.magnitude_layer(x) #Get Magnitude

        x = tf.squeeze(x, axis=-1) #Remove channels
        x = tf.transpose(x, perm=[0, 2, 1]) #(B, Frequency, Time)

        x = self.dot_layer([x, mel_basis]) #(B, Time, Mel_Channels)
        x = tf.transpose(x, perm=[0, 2, 1]) #(B, Mel_Channels, Time)

        x = tf.clip_by_value(x, clip_value_min=1e-5, clip_value_max=tf.float32.max) #Clamp
        x = tf.math.log(x) / tf.math.log(tf.constant(10, dtype=tf.float32)) #Move to decibels (log10)

        return x
