import os

#Preprocessing
SAMPLING_RATE = 22050
WINDOW_SIZE = 1024
HOP_LENGTH = 256
MEL_CHANNELS = 80
TOP_DECIBEL = 30

#Training
NORMALIZE = False
AUGMENT = False
RUN_EAGERLY = True #False
LEARNING_RATE = 1e-4
EPOCHS = 100
BATCH_SIZE = 32
SCREAMS_PATH = os.path.join("data","Metal","Vocals no silence")
WHISPERS_PATH = os.path.join("data","Dataset","Whispers")
WEIGHTS_PATH = os.path.join("weights","weights.ckpt")

#Others
RESULTS_PATH = "results"
