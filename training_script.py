import os
import matplotlib.pyplot as plt

from data import DIV2K
import model.wdsr as wdsr
import train
# from train import WdsrTrainer

import tensorflow as tf

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

train_ds = div2k_train.dataset(batch_size=512, random_transform=True)
valid_ds = div2k_valid.dataset(batch_size=64, random_transform=False, repeat_count=1)

our_model = wdsr.wdsr_b(scale=scale, num_res_blocks=depth)

our_model.compile(loss='mae')

checkpoint_path = "training_1/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

# Create a callback that saves the model's weights
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)

log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

our_model.fit(train_ds, epochs=3, validation_data=valid_ds, steps_per_epoch=5000, callbacks=[cp_callback, tensorboard_callback])