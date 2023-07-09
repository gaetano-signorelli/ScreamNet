import os
import librosa
import numpy as np
import soundfile as sf
from tensorflow.keras import Input

from src.utils.audio_preprocessing import AudioFile
from src.models.scream_transformer import ScreamTransformer

from src.config import *

PATH = os.path.join("data","Whispers","2 lug, 18.2ÔÇï.mp3")
RESULT_PATH = os.path.join(RESULTS_PATH, "test.mp3")

if __name__ == '__main__':

    print("Preprocessing...")

    audio_file = AudioFile(PATH)
    segments = audio_file.preprocess(no_silence=False)
    segments = np.expand_dims(segments, axis=-1)

    print("Preprocessing completed!")

    input_shape = (128, N_FRAMES, 1)
    warmup_input = Input(shape=input_shape)

    model = ScreamTransformer()
    model(warmup_input)

    model.summary()

    if os.path.exists(TRANSFORMATION_WEIGHTS_PATH + ".index"):
        print("Loading model's weights...")
        model.load_weights(TRANSFORMATION_WEIGHTS_PATH)
        print("Model's weights successfully loaded!")

    else:
        raise Exception("Model's weights not found.")

    print("Transforming to screams...")

    output_segments = model.predict(segments)
    output_segments = np.squeeze(output_segments)

    scream_spectrogram = np.concatenate(output_segments, axis=-1)

    print("Transformations completed!")

    print("Saving...")

    scream_audio = librosa.feature.inverse.mel_to_audio(scream_spectrogram,
                                                        sr=SAMPLING_RATE,
                                                        hop_length=HOP_LENGTH,
                                                        win_length=WINDOW_SIZE)

    sf.write(RESULT_PATH, scream_audio, SAMPLING_RATE, format="mp3")

    print("File converted successfully!")
