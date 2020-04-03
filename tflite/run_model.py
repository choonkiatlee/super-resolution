import os
import tensorflow as tf
import numpy as np

from PIL import Image
import matplotlib.pyplot as plt

def load_image(path):
    return np.array(Image.open(path))

def psnr(y_true, y_pred):
    return tf.image.psnr(y_true, y_pred, max_val=255)

def plot_sample(lr, sr):
    fig = plt.figure(figsize=(20, 10))

    images = [lr, sr]
    titles = ['LR', f'SR (x{sr.shape[0] // lr.shape[0]})']

    for i, (img, title) in enumerate(zip(images, titles)):
        plt.subplot(1, 2, i+1)
        plt.imshow(img)
        plt.title(title)
        plt.xticks([])
        plt.yticks([])

    return fig


interpreter = tf.lite.Interpreter(model_path="converted_model.tflite")
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Test model on random input data.
input_shape = input_details[0]['shape']

dataset_path = "/home/finderr/super-resolution/.div2k/images/DIV2K_train_LR_bicubic/X4"

input_pic = load_image(os.path.join(dataset_path, '0293x4.png'))
input_data = tf.expand_dims(input_pic, axis=0)
input_data = tf.cast(input_pic, tf.float32)

interpreter.set_tensor(input_details[0]['index'], input_data)

interpreter.invoke()

# The function `get_tensor()` returns a copy of the tensor data.
# Use `tensor()` in order to get a pointer to the tensor.
output_data = interpreter.get_tensor(output_details[0]['index'])[0]

output_data = tf.clip_by_value(output_data, 0, 255)
output_data = tf.round(output_data)
output_data = tf.cast(output_data, tf.uint8)

fig = plot_sample(input_data, output_data)

with open("output.png","wb+") as outfile:
    plt.savefig(outfile, format='png')
plt.close(fig)


