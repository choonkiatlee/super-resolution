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


def plot_samples(samples, interpolate_lr=False):
    fig = plt.figure(figsize=(20, 10))

    for i, (lr,sr) in enumerate(samples):

        titles = ['LR', f'SR (x{sr.shape[0] // lr.shape[0]})']

        plt.subplot(len(samples), 2, 2*i + 1)

        if interpolate_lr:
            lr_img = Image.fromarray(lr)

            print(sr.shape)
            interpolated = lr_img.resize(sr.shape[0:1], resample=Image.BILINEAR)
            lr = np.array(interpolated)
            
        plt.imshow(lr)
        plt.title(titles[0])

        plt.subplot(len(samples), 2, 2*i + 2)
        plt.imshow(sr)
        plt.title(titles[1])

        plt.xticks([])
        plt.yticks([])
    return fig
