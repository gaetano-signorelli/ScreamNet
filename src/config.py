import os

#Preprocessing
SAMPLING_RATE = 22050 # 8192
WINDOW_SIZE = 1024
HOP_LENGTH = 256 # 768
N_FRAMES = 32 # 25
TRANSFORM_METHOD = "mel" #One between "mel", "mfcc" and "stft"
TOP_DECIBEL = 60
ACCEPTABLE_RATE = 0.05

#Classification training
CLASSIFICATION_NORMALIZE = False
CLASSIFICATION_RUN_EAGERLY = True #False
CLASSIFICATION_LEARNING_RATE = 1e-3
CLASSIFICATION_EPOCHS = 7
CLASSIFICATION_BATCH_SIZE = 64
CLASSIFICATION_VALIDATION_SPLIT = 0.2
CLASSIFICATION_TRAIN_PATH = os.path.join("data","Dataset")
CLASSIFICATION_WEIGHTS_PATH = os.path.join("weights","classifier","weights.ckpt")

#Transformation training
TRANSFORMATION_NORMALIZE = False
TRANSFORMATION_RUN_EAGERLY = True #False
TRANSFORMATION_LEARNING_RATE = 1e-4
TRANSFORMATION_EPOCHS = 100
TRANSFORMATION_BATCH_SIZE = 32
TRANSFORMATION_TRAIN_PATH = os.path.join("data","Dataset","Whispers")
TRANSFORMATION_WEIGHTS_PATH = os.path.join("weights","transformer","weights.ckpt")
WEIGHT_SCREAM_LOSS = 0.5
WEIGHT_CORRELATION_LOSS = 1.0

#Others
RESULTS_PATH = "results"
