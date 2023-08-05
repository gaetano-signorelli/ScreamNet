import os

#Preprocessing
SAMPLING_RATE = 22050
N_FFT = 1024
WINDOW_SIZE = 1024
HOP_LENGTH = 256
MEL_CHANNELS = 80
TOP_DECIBEL = 30

#Training
NORMALIZE = True
AUGMENT = False
RUN_EAGERLY = True #False
LEARNING_RATE = 1e-4
SEGMENT_LENGTH = 22050
EPOCHS = 3000
EPOCH_LEN = 1000
BATCH_SIZE = 16
CORRELATION_WEIGHT = 10

#Paths
SCREAMS_PATH = os.path.join("data","Metal","Vocals no silence")
WHISPERS_PATH = os.path.join("data","Dataset","Whispers")
WEIGHTS_PATH = os.path.join("weights","weights.ckpt")
RESULTS_PATH = "results"
