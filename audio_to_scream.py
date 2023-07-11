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

FILE = "test2"
PATH = os.path.join("data","Whispers", FILE + ".mp3")
RESULT_PATH = os.path.join(RESULTS_PATH, FILE + ".mp3")

def screamify(transformer, denoiser, input_path, output_path):

    print("Preprocessing file " + input_path + "...")

    audio_file = AudioFile(input_path)
    segments = audio_file.preprocess(no_silence=False,
                                    filter_acceptability=False,
                                    pad=True)
    segments = np.expand_dims(segments, axis=-1)

    print("Preprocessing completed!")

    print("Transforming to screams...")

    #segments[:, 96:, :, :] = 0
    output_segments = transformer.predict(segments)

    print("Transformations completed!")
    print("Denoising...")

    output_segments = denoiser.predict(output_segments)
    output_segments = np.squeeze(output_segments)

    scream_spectrogram = np.concatenate(output_segments, axis=-1)
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
