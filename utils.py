import numpy as np
import matplotlib.pyplot as plt

from PIL import Image


def load_image(path, make_input_img_bw=False):

    img = Image.open(path)

    if make_input_img_bw:
        img = img.convert('L')

    return np.array(img)


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


def plot_samples(samples, interpolate_lr=False, input_img_bw=False):
    fig = plt.figure(figsize=(20, 10))

    for i, (lr,sr) in enumerate(samples):

        titles = ['LR', f'SR (x{sr.shape[0] // lr.shape[0]})']

        plt.subplot(len(samples), 2, 2*i + 1)

        if interpolate_lr:
            lr_img = Image.fromarray(lr)
            interpolated = lr_img.resize(sr.shape[0:2], resample=Image.BILINEAR)
            lr = np.array(interpolated)
        
        if input_img_bw:
            plt.imshow(lr, cmap='gray', vmin=0, vmax=255)
        else:
            plt.imshow(lr)

        plt.title(titles[0])

        plt.subplot(len(samples), 2, 2*i + 2)
        plt.imshow(sr)
        plt.title(titles[1])

        plt.xticks([])
        plt.yticks([])
    return fig
