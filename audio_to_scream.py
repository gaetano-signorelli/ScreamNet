import os
import librosa
import numpy as np
import tqdm as tqdm
import soundfile as sf
from tensorflow.keras import Input

from src.utils.audio_preprocessing import AudioFile, save_spectrogram
from src.models.scream_transformer import ScreamTransformer
from src.models.denoise_transformer import DenoiseTransformer

from src.config import *

FILE = "aaaaaaahhh"
PATH = os.path.join("data","Whispers", FILE + ".mp3")
RESULT_PATH = os.path.join(RESULTS_PATH, FILE + ".mp3")

EMPOWER = 50
INTERPOLATION_RATE = 1.0

def build_denoise_set(spectrogram):

    frames = []

    for i in range(spectrogram.shape[1]):
        start = i - (N_FRAMES//2)
        end = i + (N_FRAMES//2)
        pad_left = 0
        pad_right = 0

        if start<0:
            pad_left = -start
            start = 0
        if end>spectrogram.shape[1]:
            pad_right = end - spectrogram.shape[1]
            end = spectrogram.shape[1]

        frame = spectrogram[:, start:end, :]
        frame = np.pad(frame, ((0,0),(pad_left,pad_right),(0,0)))

        frames.append(frame)

    return np.array(frames)

def screamify(transformer, denoiser, input_path, output_path):

    print("Preprocessing file " + input_path + "...")

    audio_file = AudioFile(input_path)
    segments = audio_file.preprocess(no_silence=False,
                                    filter_acceptability=False,
                                    pad=True)
    segments = np.expand_dims(segments, axis=-1)

    print("Preprocessing completed!")

    print("Transforming to screams...")

    output_segments = transformer.predict(segments)

    print("Transformations completed!")
    print("Denoising...")

    scream_spectrogram = np.concatenate(output_segments, axis=-2)

    frames = build_denoise_set(scream_spectrogram)
    denoised_frames = denoiser.predict(frames)

    denoised_frames = np.squeeze(denoised_frames)
    denoised_scream_spectrogram = np.transpose(denoised_frames)

    scream_spectrogram = (1.0-INTERPOLATION_RATE) * np.squeeze(scream_spectrogram) + INTERPOLATION_RATE * denoised_scream_spectrogram

    scream_spectrogram*= EMPOWER
    #scream_spectrogram[96:,:]=0

    print("Denoising completed!")

    print("Saving...")

    save_spectrogram(scream_spectrogram, os.path.join(RESULTS_PATH, FILE + ".tif"))

    scream_audio = librosa.feature.inverse.mel_to_audio(scream_spectrogram,
                                                        sr=SAMPLING_RATE,
                                                        hop_length=HOP_LENGTH,
                                                        win_length=WINDOW_SIZE)

    sf.write(output_path, scream_audio, SAMPLING_RATE, format="mp3")

    print("File converted successfully!")

if __name__ == '__main__':

    input_shape = (128, N_FRAMES, 1)
    warmup_input = Input(shape=input_shape)

    transformer = ScreamTransformer()
    denoiser = DenoiseTransformer()
    transformer(warmup_input)
    denoiser(warmup_input)

    #transformer.summary()
    #denoiser.summary()

    if os.path.exists(TRANSFORMATION_WEIGHTS_PATH + ".index"):
        print("Loading Transformer's weights...")
        transformer.load_weights(TRANSFORMATION_WEIGHTS_PATH)
        print("Transformer's weights successfully loaded!")

    else:
        raise Exception("Transformer's weights not found.")

    if os.path.exists(DENOISER_WEIGHTS_PATH + ".index"):
        print("Loading Denoiser's weights...")
        denoiser.load_weights(DENOISER_WEIGHTS_PATH)
        print("Denoiser's weights successfully loaded!")

    else:
        raise Exception("Denoiser's weights not found.")


    screamify(transformer, denoiser, PATH, RESULT_PATH)
