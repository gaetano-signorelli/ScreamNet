import os
from tqdm import tqdm

from src.utils.audio_preprocessing import AudioFile, save_spectrogram

INPUT_PATHS = [
#os.path.join("data","GTZAN","Vocals"),
#os.path.join("data","Metal","Vocals"),
os.path.join("data","Metal","Noisy vocals"),
#os.path.join("data","Whispers")
]

OUTPUT_PATHS = [
#os.path.join("data","Dataset","Sings"),
#os.path.join("data","Dataset","Screams"),
os.path.join("data","Dataset","Noisy screams"),
#os.path.join("data","Dataset","Whispers"),
]

def get_audio_paths(root):

    paths = []
    names = []

    for dirname, dirnames, filenames in os.walk(root):
        for filename in filenames:
            paths.append(os.path.join(dirname, filename))
            names.append(os.path.splitext(filename)[0])

    return paths, names

if __name__ == '__main__':

    for in_path, out_path in zip(INPUT_PATHS,OUTPUT_PATHS):
        print("Extracting songs from " + in_path)
        path_files, path_names = get_audio_paths(in_path)

        for path_file, path_name in tqdm(zip(path_files, path_names)):
            audio_file = AudioFile(path_file)
            segments = audio_file.preprocess()

            if segments is not None:
                for i, segment in enumerate(segments):
                    save_path = os.path.join(out_path, path_name + "_"+str(i)+".tif")
                    save_spectrogram(segment, save_path)
