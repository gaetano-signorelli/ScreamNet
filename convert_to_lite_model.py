import os
import tensorflow as tf
from tensorflow.keras import Input

from src.model.gan import ScreamGAN

from src.config import *

OUTPUT_PATH = os.path.join("weights","lite model","screamer.tflite")

if __name__ == '__main__':

    input_shape = (SEGMENT_LENGTH)
    warmup_input = Input(shape=input_shape)

    gan = ScreamGAN(tflite=True)
    gan(warmup_input)

    if os.path.exists(WEIGHTS_PATH + ".index"):
        print("Loading GAN's weights...")
        gan.load_weights(WEIGHTS_PATH)
        print("GAN's weights successfully loaded!")

    else:
        raise Exception("GAN's weights not found.")

    # Convert the model
    converter = tf.lite.TFLiteConverter.from_keras_model(gan.generator)

    converter.target_spec.supported_ops = [
      tf.lite.OpsSet.TFLITE_BUILTINS, # enable TensorFlow Lite ops.
      tf.lite.OpsSet.SELECT_TF_OPS # enable TensorFlow ops.
    ]

    tflite_model = converter.convert()

    # Save the model.
    with open(OUTPUT_PATH, 'wb') as f:
      f.write(tflite_model)

    print("Conversion completed!")
