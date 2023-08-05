import os

#Preprocessing
SAMPLING_RATE = 22050 # 8192
WINDOW_SIZE = 1024
HOP_LENGTH = 256 # 768
N_FRAMES = 32 # 25
TRANSFORM_METHOD = "mel" #One between "mel", "mfcc" and "stft"
TOP_DECIBEL = 60
ACCEPTABLE_RATE = 0.05

#Training
NORMALIZE = False
RUN_EAGERLY = True #False
LEARNING_RATE = 1e-3
EPOCHS = 100
BATCH_SIZE = 32
SCREAMS_PATH = os.path.join("data","Dataset","Screams")
WHISPERS_PATH = os.path.join("data","Dataset","Whispers")
WEIGHTS_PATH = os.path.join("weights","weights.ckpt")

#Others
RESULTS_PATH = "results"
