import os
import tensorflow as tf


if os.path.exists('saved_model'):
    print("Loaded previously saved model")
    our_model = tf.keras.models.load_model('saved_model', custom_objects={'psnr': psnr})

def psnr(y_true, y_pred):
    return tf.image.psnr(y_true, y_pred, max_val=255)

converter = tf.lite.TFLiteConverter.from_keras_model(our_model)

tflite_model = converter.convert()

open("converted_model.tflite", "wb").write(tflite_model)
