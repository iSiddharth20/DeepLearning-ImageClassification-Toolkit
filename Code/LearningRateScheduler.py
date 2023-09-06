'''
Source : https://github.com/iSiddharth20/DeepLearning-ImageClassification-Toolkit
'''


'''
Learning Rate Scheduler
  - Use whichever suits the Dataset, Batch Size, Requirements
  - Modify as necessary
'''


# Manual 
import tensorflow as tf
initial_learning_rate = 0.001
def scheduler(epoch, lr):
  if epoch < 10:
    return lr
  else:
    return lr * tf.math.exp(-0.1)
lr_scheduler = tf.keras.callbacks.LearningRateScheduler(scheduler)


# ReduceLROnPlateau
from tensorflow.keras.callbacks import ReduceLROnPlateau
initial_learning_rate = 0.01
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, verbose=1, mode='min', cooldown=1, min_lr=1e-4)


# Cosine Annealing with Warm Restarts (Best if Paired with SGD with Momentum)
import tensorflow as tf
initial_learning_rate = 0.01
lr_scheduler = tf.keras.experimental.CosineDecayRestarts(
    initial_learning_rate,
    first_decay_steps=10,
    t_mul=1.5,
    m_mul=0.9
)


# Cosine Annealing
import tensorflow as tf
import numpy as np
initial_learning_rate = 0.0001
def cosine_annealing(epoch, num_epochs=EPOCHS, initial_rate=initial_learning_rate, min_lr=1e-5):
    cosine_decay = 0.5 * (1 + np.cos(np.pi * epoch / num_epochs))
    lr = (initial_rate - min_lr) * cosine_decay + min_lr
    return lr
lr_scheduler = tf.keras.callbacks.LearningRateScheduler(lambda epoch: cosine_annealing(epoch))
