import os
import matplotlib.pyplot as plt

from data import DIV2K
import model.wdsr as wdsr

import model.dscn

import train
# from train import WdsrTrainer

import tensorflow as tf
import datetime

LOAD_SAVED_MODEL = True

# Number of residual blocks
depth = 32

# Super-resolution factor
scale = 4

# Downgrade operator
downgrade = 'bicubic'

SAVED_MODEL_DIR = 'saved_model_dscn'

############################################### Load Data Set ####################################################

# Location of model weights (needed for demo)
weights_dir = f'weights/wdsr-b-{depth}-x{scale}'
weights_file = os.path.join(weights_dir, 'weights.h5')

os.makedirs(weights_dir, exist_ok=True)

div2k_train = DIV2K(scale=scale, subset='train', downgrade=downgrade)
div2k_valid = DIV2K(scale=scale, subset='valid', downgrade=downgrade)

train_ds = div2k_train.dataset(batch_size=256, random_transform=True)
valid_ds = div2k_valid.dataset(batch_size=1, random_transform=False, repeat_count=1)

############################################### Setup Training stuff ####################################################

checkpoint_path = "training_1/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

# Create a callback that saves the model's weights
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)

log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
tb_file_writer = tf.summary.create_file_writer(log_dir)

STEPS_PER_EPOCH = 800//256

lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
  0.001,
  decay_steps=STEPS_PER_EPOCH*1000,
  decay_rate=1,
  staircase=False)

def get_optimizer():
  return tf.keras.optimizers.Adam(lr_schedule)

def psnr(y_true, y_pred):
    return tf.image.psnr(y_true, y_pred, max_val=255)

from model import resolve_single
from utils import load_image, plot_samples
import io

def resolve_and_tensorboard_plot(our_model, lr_image_paths, title=''):

    samples = []

    for lr_image_path in lr_image_paths:

        lr = load_image(lr_image_path)
        sr = resolve_single(our_model, lr)
        samples.append((lr,sr))

    fig = plot_samples(samples, interpolate_lr=True)

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    # Closing the figure prevents it from being displayed directly inside
    # the notebook.
    plt.close(fig)
    buf.seek(0)
    # Convert PNG buffer to TF image
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    # Add the batch dimension
    image = tf.expand_dims(image, 0)

    with tb_file_writer.as_default():
        tf.summary.image(title, image, step=0)


############################################### Load Model and Train ####################################################


if os.path.exists(SAVED_MODEL_DIR) and LOAD_SAVED_MODEL:
    
    print("Loaded previously saved model")
    our_model = tf.keras.models.load_model(SAVED_MODEL_DIR, custom_objects={'psnr': psnr})

else:
    our_model = model.dscn.dscn(scale=4, n_fe_layers = 12)
    our_model.compile(
        optimizer=get_optimizer(), 
        loss='mae',
        metrics=[psnr],
    )

# resolve_and_tensorboard_plot(our_model, ['demo/0869x4-crop.png', 'demo/0829x4-crop.png', 'demo/0851x4-crop.png'], "Start")
resolve_and_tensorboard_plot(our_model, ['demo/0869x4-crop.png'], "Start")


tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

initial_epoch = our_model.optimizer.iterations.numpy() // STEPS_PER_EPOCH
print("Starting on initial epoch: {0}".format(initial_epoch))

our_model.fit(
    train_ds, 
    validation_data=valid_ds.take(10), 
    epochs = 100,
    steps_per_epoch=STEPS_PER_EPOCH, 
    initial_epoch=initial_epoch,
    callbacks=[cp_callback, tensorboard_callback],
    verbose=1,
)

resolve_and_tensorboard_plot(our_model, ['demo/0869x4-crop.png'], "End")
# resolve_and_tensorboard_plot(our_model, ['demo/0869x4-crop.png', 'demo/0829x4-crop.png', 'demo/0851x4-crop.png'], "End")

our_model.save(SAVED_MODEL_DIR)
