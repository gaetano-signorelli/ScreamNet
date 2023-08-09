import os
import librosa
import numpy as np
import tqdm as tqdm
import soundfile as sf
from tensorflow.keras import Input

from src.model.gan import ScreamGAN

from src.config import *

FILE = "it works"
PATH = os.path.join("data","Whispers", FILE + ".mp3")
RESULT_PATH = os.path.join(RESULTS_PATH, FILE + ".mp3")

def screamify(model, input_path, output_path):

    #Load file with librosa
    original_wave, sr = librosa.load(input_path, sr=SAMPLING_RATE)

    #Split in 1sec segments (22050 samples)
    diff = SEGMENT_LENGTH - (original_wave.shape[0] % SEGMENT_LENGTH)
    original_wave = np.pad(original_wave, (0, diff))
    n_chunks = original_wave.shape[0] // SEGMENT_LENGTH
    segments = np.split(original_wave, n_chunks)
    batch = np.array(segments)

    #Predict (transform to screams)
    results = model.generator.predict(batch)

    #Concatenate (flatten)
    scream = np.concatenate(results)

    #Save
    sf.write(output_path, scream, SAMPLING_RATE, format="mp3")

    print("Conversion completed!")

if __name__ == '__main__':

    input_shape = (SEGMENT_LENGTH)
    warmup_input = Input(shape=input_shape)

    gan = ScreamGAN(tflite=False, use_weight_norm=False)
    gan(warmup_input)

    if os.path.exists(WEIGHTS_PATH + ".index"):
        print("Loading GAN's weights...")
        gan.load_weights(WEIGHTS_PATH)
        print("GAN's weights successfully loaded!")

    else:
        raise Exception("GAN's weights not found.")

    screamify(gan, PATH, RESULT_PATH)
