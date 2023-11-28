import os

DATASET_NAME = "Vocals"

#Preprocessing
SAMPLING_RATE = 22050
N_FFT = 1024
WINDOW_SIZE = 1024
HOP_LENGTH = 256
MEL_CHANNELS = 80
TOP_DECIBEL = 30

#Training
AUGMENT = False
USE_WEIGHT_NORMALIZATION = False
USE_MIXED_PRECISION = True
RUN_EAGERLY = False
LEARNING_RATE = 1e-4
BETA_1 = 0.5
BETA_2 = 0.9
SEGMENT_LENGTH = 22050
EPOCHS = 1000
EPOCH_LEN = 100
BATCH_SIZE = 8
CORRELATION_WEIGHT = 150 #250

#Paths
SCREAMS_PATH = os.path.join("data","Dataset","Screams")
TRAINING_PATH = os.path.join("data","Dataset", DATASET_NAME)
WEIGHTS_BASE_PATH = os.path.join("weights", DATASET_NAME)
WEIGHTS_PATH = os.path.join(WEIGHTS_BASE_PATH, "weights.ckpt")
RESULTS_PATH = "results"
TRAIN_SAMPLES_PATH = os.path.join("results","{} test samples".format(DATASET_NAME))

if not os.path.exists(TRAIN_SAMPLES_PATH):
    os.mkdir(TRAIN_SAMPLES_PATH)

if not os.path.exists(WEIGHTS_BASE_PATH):
    os.mkdir(WEIGHTS_BASE_PATH)
