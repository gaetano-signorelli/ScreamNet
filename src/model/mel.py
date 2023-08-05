import numpy as np
import librosa
import tensorflow as tf
from tf.keras import layers, Model
from kapre import STFT, Magnitude, STFTTflite, MagnitudeTflite

from src.config import *

class WavToMel(Model):

    def __init__(self):

        super().__init__()

        self.stft_layer = STFT(n_fft=N_FFT, win_length=WINDOW_SIZE, hop_length=HOP_LENGTH,
                               window_name=None, pad_end=True, pad_begin=True,
                               input_data_format='channels_last', output_data_format='channels_last')

        self.magnitude_layer = Magnitude()

        mel_filters = librosa.filters.mel(SAMPLING_RATE, N_FFT, n_mels=MEL_CHANNELS)
        self.mel_basis = tf.convert_to_tensor(mel_filters, dtype=tf.float32)

        self.dot_layer = layers.Dot(axes=(1,2))

    def call(self, x):

        x = tf.expand_dims(x, axis=-1) #Add channel (mono)

        x = self.stft_layer(x) #Compure Short Fourier Transform
        x = self.magnitude_layer(x) #Get Magnitude

        x = tf.squeeze(x, axis=-1) #Remove channels
        x = tf.transpose(x, perm=[0, 2, 1]) #(B, Frequency, Time)

        x = self.dot_layer([self.mel_basis, x]) #(B, Mel_channels, Time)

        x = tf.clip_by_value(x, clip_value_min=1e-5, clip_value_max=tf.float32.max) #Clamp
        x = tf.math.log(x) / tf.math.log(tf.constant(10, dtype=tf.float32)) #Move to decibels (log10)

        return x

class WavToMelLite(Model):

    def __init__(self):

        super().__init__()

        self.stft_layer = STFTTflite(n_fft=N_FFT, win_length=WINDOW_SIZE, hop_length=HOP_LENGTH,
                               window_name=None, pad_end=True, pad_begin=True,
                               input_data_format='channels_last', output_data_format='channels_last')

        self.magnitude_layer = MagnitudeTflite()

        mel_filters = librosa.filters.mel(SAMPLING_RATE, N_FFT, n_mels=MEL_CHANNELS)
        self.mel_basis = tf.convert_to_tensor(mel_filters, dtype=tf.float32)

        self.dot_layer = layers.Dot(axes=(1,2))

    def call(self, x):

        x = tf.expand_dims(x, axis=-1) #Add channel (mono)

        x = self.stft_layer(x) #Compure Short Fourier Transform
        x = self.magnitude_layer(x) #Get Magnitude

        x = tf.squeeze(x, axis=-1) #Remove channels
        x = tf.transpose(x, perm=[0, 2, 1]) #(B, Frequency, Time)

        x = self.dot_layer([self.mel_basis, x]) #(B, Mel_channels, Time)

        x = tf.clip_by_value(x, clip_value_min=1e-5, clip_value_max=tf.float32.max) #Clamp
        x = tf.math.log(x) / tf.math.log(tf.constant(10, dtype=tf.float32)) #Move to decibels (log10)

        x = tf.expand_dims(x, axis=1)

        return x
