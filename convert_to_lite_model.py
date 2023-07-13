import os
import tensorflow as tf
from tensorflow.keras import Input

from src.models.scream_transformer import ScreamTransformer

from src.config import *

OUTPUT_PATH = os.path.join("weights","lite models","screamer.tflite")

if __name__ == '__main__':

    input_shape = (128, N_FRAMES, 1)
    warmup_input = Input(shape=input_shape)

    transformer = ScreamTransformer()
    transformer(warmup_input)

    if os.path.exists(TRANSFORMATION_WEIGHTS_PATH + ".index"):
        print("Loading Transformer's weights...")
        transformer.load_weights(TRANSFORMATION_WEIGHTS_PATH)
        print("Transformer's weights successfully loaded!")

    else:
        raise Exception("Transformer's weights not found.")

    # Convert the model
    converter = tf.lite.TFLiteConverter.from_keras_model(transformer)
    tflite_model = converter.convert()

    # Save the model.
    with open(OUTPUT_PATH, 'wb') as f:
      f.write(tflite_model)

    print("Conversion completed!")
