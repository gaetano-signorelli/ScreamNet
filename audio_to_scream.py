import os
import librosa
import numpy as np
import tqdm as tqdm
import soundfile as sf
from tensorflow.keras import Input

from src.utils.audio_preprocessing import AudioFile, save_spectrogram
from src.models.scream_transformer import ScreamTransformer

from src.config import *

PATH = os.path.join("data","Whispers","2 lug, 19.20ÔÇï.mp3")
RESULT_PATH = os.path.join(RESULTS_PATH, "test.mp3")

def screamify(model, input_path, output_path):

    print("Preprocessing file " + input_path + "...")

    audio_file = AudioFile(input_path)
    segments = audio_file.preprocess(no_silence=False,
                                    filter_acceptability=False,
                                    pad=True)
    segments = np.expand_dims(segments, axis=-1)

    print("Preprocessing completed!")

    print("Transforming to screams...")

    output_segments = model.predict(segments)
    output_segments = np.squeeze(output_segments)

    scream_spectrogram = np.concatenate(output_segments, axis=-1)

    print("Transformations completed!")

    print("Saving...")

    save_spectrogram(scream_spectrogram, os.path.join(RESULTS_PATH,"test_spectrogram.tif"))

    scream_audio = librosa.feature.inverse.mel_to_audio(scream_spectrogram,
                                                        sr=SAMPLING_RATE,
                                                        hop_length=HOP_LENGTH,
                                                        win_length=WINDOW_SIZE)

    sf.write(output_path, scream_audio, SAMPLING_RATE, format="mp3")

    print("File converted successfully!")

if __name__ == '__main__':

    input_shape = (128, N_FRAMES, 1)
    warmup_input = Input(shape=input_shape)

    model = ScreamTransformer()
    model(warmup_input)

    #model.summary()

    if os.path.exists(TRANSFORMATION_WEIGHTS_PATH + ".index"):
        print("Loading model's weights...")
        model.load_weights(TRANSFORMATION_WEIGHTS_PATH)
        print("Model's weights successfully loaded!")

    else:
        raise Exception("Model's weights not found.")

    screamify(model, PATH, RESULT_PATH)

    '''
    input_path = os.path.join("data","Whispers")
    output_path = os.path.join("data","Metal","Noisy vocals")
    count = 2

    paths = []
    names = []

    for dirname, dirnames, filenames in os.walk(input_path):
        for filename in filenames:
            paths.append(os.path.join(dirname, filename))
            names.append(os.path.splitext(filename)[0])

    for path, name in zip(paths, names):
        out_path = os.path.join(output_path, name+str(count)+".mp3")
        screamify(model, path, out_path)

    print("Ended!")
    '''
