import os
import numpy as np
from tqdm import tqdm

from src.utils.audio_preprocessing import AudioFile

INPUT_PATH = os.path.join("data","GTZAN","Vocals")

OUTPUT_PATH = os.path.join("data","GTZAN","Vocals no silence")

def get_audio_paths(root):

    paths = []
    names = []

    for dirname, dirnames, filenames in os.walk(root):
        for filename in filenames:
            paths.append(os.path.join(dirname, filename))
            names.append(os.path.splitext(filename)[0])

    return paths, names

if __name__ == '__main__':

    if not os.path.exists(OUTPUT_PATH):
        os.mkdir(OUTPUT_PATH)

    path_files, path_names = get_audio_paths(INPUT_PATH)

    for path_file, path_name in tqdm(zip(path_files, path_names)):
        audio_file = AudioFile(path_file, path_name)
        audio_file.preprocess()
        audio_file.save(OUTPUT_PATH)
