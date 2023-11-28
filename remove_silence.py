import os
import argparse
import numpy as np
from tqdm import tqdm

from src.utils.audio_preprocessing import AudioFile

def parse_arguments():

    parser = argparse.ArgumentParser(description='Remove silence')
    parser.add_argument('input_path', type=str, help='Location of the audio files')
    parser.add_argument('output_path', type=str, help='Location for saving the output files')

    args = parser.parse_args()

    return args

def get_audio_paths(root):

    paths = []
    names = []

    for dirname, dirnames, filenames in os.walk(root):
        for filename in filenames:
            paths.append(os.path.join(dirname, filename))
            names.append(os.path.splitext(filename)[0])

    return paths, names

if __name__ == '__main__':

    args = parse_arguments()

    if not os.path.exists(args.output_path):
        os.mkdir(args.output_path)

    path_files, path_names = get_audio_paths(args.input_path)

    for path_file, path_name in tqdm(zip(path_files, path_names)):
        audio_file = AudioFile(path_file, path_name)
        audio_file.preprocess()
        audio_file.save(args.output_path)

    print("Files saved in: " + args.output_path)
