import numpy as np
import matplotlib.pyplot as plt

from PIL import Image


def load_image(path):
    return np.array(Image.open(path))


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


def plot_samples(samples):
    fig = plt.figure(figsize=(20, 10))

    for i, (lr,sr) in enumerate(samples):

        titles = ['LR', f'SR (x{sr.shape[0] // lr.shape[0]})']

        plt.subplot(i, 2, 1)
        plt.imshow(lr)
        plt.title(titles[0])

        plt.subplot(i,2,2)
        plt.imshow(sr)
        plt.title(titles[1])

        plt.xticks([])
        plt.yticks([])
    return fig
