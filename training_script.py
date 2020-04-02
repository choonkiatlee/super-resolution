import os
import matplotlib.pyplot as plt

from data import DIV2K
import model.wdsr as wdsr
import train
# from train import WdsrTrainer

import tensorflow as tf
import datetime

# Number of residual blocks
depth = 32

# Super-resolution factor
scale = 4

# Downgrade operator
downgrade = 'bicubic'

# Location of model weights (needed for demo)
weights_dir = f'weights/wdsr-b-{depth}-x{scale}'
weights_file = os.path.join(weights_dir, 'weights.h5')

os.makedirs(weights_dir, exist_ok=True)

div2k_train = DIV2K(scale=scale, subset='train', downgrade=downgrade)
div2k_valid = DIV2K(scale=scale, subset='valid', downgrade=downgrade)

train_ds = div2k_train.dataset(batch_size=256, random_transform=True)
valid_ds = div2k_valid.dataset(batch_size=25, random_transform=False, repeat_count=1)

our_model = wdsr.wdsr_b(scale=scale, num_res_blocks=depth)


checkpoint_path = "training_1/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

# Create a callback that saves the model's weights
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)

# log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
# tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

STEPS_PER_EPOCH = 800//256

lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
  0.001,
  decay_steps=STEPS_PER_EPOCH*1000,
  decay_rate=1,
  staircase=False)

def get_optimizer():
  return tf.keras.optimizers.Adam(lr_schedule)

our_model.compile(
    optimizer=get_optimizer(), 
    loss='mae',
)

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

our_model.fit(
    train_ds, 
    validation_data=valid_ds, 
    epochs = 100,
    steps_per_epoch=STEPS_PER_EPOCH, 
    callbacks=[cp_callback],
    verbose=1
)