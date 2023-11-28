import os
import argparse
import librosa
import numpy as np
import tqdm as tqdm
import soundfile as sf
from tensorflow.keras import Input

from src.model.gan import ScreamGAN

from src.config import *

NORMALIZE = False

def parse_arguments():

    parser = argparse.ArgumentParser(description='Voice to scream')
    parser.add_argument('input_path', type=str, help='Path to the input voice')

    args = parser.parse_args()

    return args

def normalize(wave, target_db=-1.0):

    target_peak = np.power(10, target_db / 20)
    current_peak = np.max(target_peak)
    gain = target_peak / current_peak
    normalized_wave = wave * gain

    return normalized_wave

def screamify(model, input_path, output_path):

    #Load file with librosa
    original_wave, sr = librosa.load(input_path, sr=SAMPLING_RATE, res_type='soxr_lq')

    if NORMALIZE:
        original_wave = normalize(original_wave)

    #Split in 1sec segments (22050 samples)
    diff = SEGMENT_LENGTH - (original_wave.shape[0] % SEGMENT_LENGTH)
    original_wave = np.pad(original_wave, (0, diff))
    n_chunks = original_wave.shape[0] // SEGMENT_LENGTH
    segments = np.split(original_wave, n_chunks)
    batch = np.array(segments)

    #Predict (transform to screams)
    results = model.generator.predict(batch)

    #Concatenate (flatten) and remove padding
    scream = np.concatenate(results)
    scream = scream[:-diff]

    #Save
    sf.write(output_path, scream, SAMPLING_RATE, format="mp3")

    print("Result saved in: " + output_path)
    print("Conversion completed!")

if __name__ == '__main__':

    args = parse_arguments()
    name = os.path.basename(args.input_path).split(".")[0]
    output_path = os.path.join(RESULTS_PATH, name + "_scream.mp3")

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

    screamify(gan, args.input_path, output_path)
