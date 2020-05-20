import tensorflow as tf
import keras.models as mod
import keras.layers as lyr
import keras.regularizers as reg
import keras.initializers as init
import keras.callbacks as call
import keras.optimizers as opt
import keras.metrics as met
import keras.losses as losses
from shared.custom_layers.local_response_normalization import lrn_parametric, lrn_shape
from shared.generators.augmentation_list import AugmentationList
import shared.definitions.paths as paths
import albumentations
import os
from os import path

"""
Model Definition
"""
version = 0
num_classes = 1000
num_batches = None

k, n, alpha, beta = 2, 5, 1, 0.75
lrn = lambda tensor: lrn_parametric(tensor, k, n, alpha, beta)

model = mod.Sequential(
 [# We input 224 x 224 pixel images with 3 channels (rgb)
  # We create 96 feature maps, by using an 11x11 pixel kernel with stride 4
  lyr.Conv2D(96, 11, strides=4, padding='same',
             input_shape=(224, 224, 3,), activation='relu',
             kernel_initializer=init.he_uniform(),  # original: RandomNormal(stdev=0.01)
             bias_initializer=init.Zeros(),
             kernel_regularizer=reg.l2(0.0005),
             bias_regularizer=reg.l2(0.0005)
             ),
  # The lambdas here are the local response normalization
  lyr.Lambda(lrn, output_shape=lrn_shape),
  lyr.MaxPool2D(3, 2),
  lyr.Conv2D(256, 5, strides=1, padding='same', activation='relu',
             kernel_initializer=init.he_uniform(),  # original: RandomNormal(stdev=0.01)
             bias_initializer=init.Zeros(), # original: init.Ones()
             kernel_regularizer=reg.l2(0.0005),
             bias_regularizer=reg.l2(0.0005)
             ),
  lyr.Lambda(lrn, output_shape=lrn_shape),
  lyr.MaxPool2D(3, 2),
  lyr.Conv2D(384, 3, strides=1, padding='same', activation='relu',
             kernel_initializer=init.he_uniform(),  # original: RandomNormal(stdev=0.01)
             bias_initializer=init.Zeros(),
             kernel_regularizer=reg.l2(0.0005),
             bias_regularizer=reg.l2(0.0005)
             ),
  lyr.Conv2D(384, 3, strides=1, padding='same', activation='relu',
             kernel_initializer=init.he_uniform(),  # original: RandomNormal(stdev=0.01)
             bias_initializer=init.Zeros(), # original: init.Ones()
             kernel_regularizer=reg.l2(0.0005),
             bias_regularizer=reg.l2(0.0005)
             ),
  lyr.Conv2D(256, 3, strides=1, padding='same', activation='relu',
             kernel_initializer=init.he_uniform(),  # original: RandomNormal(stdev=0.01)
             bias_initializer=init.Zeros(), # original: init.Ones()
             kernel_regularizer=reg.l2(0.0005),
             bias_regularizer=reg.l2(0.0005)
             ),
  # Note from the diagram there is one last pool op after the last conv
  lyr.MaxPool2D(3,2),
  lyr.Flatten(),
  lyr.Dense(4096, activation='relu',
            kernel_initializer=init.he_uniform(),  # original: RandomNormal(stdev=0.01)
            bias_initializer=init.Zeros(),  # original: init.Ones()
            kernel_regularizer=reg.l2(0.0005),
            bias_regularizer=reg.l2(0.0005)
            ),
  # The paper says the first two dense layers are regularized with dropout
  lyr.Dropout(0.5),
  lyr.Dense(4096, activation='relu',
            kernel_initializer=init.he_uniform(),  # original: RandomNormal(stdev=0.01)
            bias_initializer=init.Zeros(),  # original: init.Ones()
            kernel_regularizer=reg.l2(0.0005),
            bias_regularizer=reg.l2(0.0005)
            ),
  lyr.Dropout(0.5),
  lyr.Dense(num_classes,
            activation='softmax',
            kernel_initializer=init.he_uniform(),  # original: RandomNormal(stdev=0.01)
            bias_initializer=init.Zeros(),  # original: init.Ones()
            kernel_regularizer=reg.l2(0.0005),
            bias_regularizer=reg.l2(0.0005)
            ),
 ]
)

"""
Optimizer, Loss, & Metrics

Using zero built in decay and relying exclusively on the heuristic
used in the original paper.
"""
optimizer = opt.Adam(0.0001)
# This line below would match the paper exactly but was MUCH slower than using Adam on 2 class example
# optimizer = opt.SGD(learning_rate=0.01, momentum=0.9, decay=0.0) # Note decay refers to learning rate

top_1_acc = met.categorical_accuracy


def loss(y_true, y_pred):
    return losses.categorical_crossentropy(y_true, y_pred, from_logits=False, label_smoothing=0)


def top_5_acc(y_true, y_pred):
    return  met.top_k_categorical_accuracy(y_true, tf.cast(y_pred, dtype='float32'), k=5)


model.compile(optimizer=optimizer,
              loss=loss,
              metrics=[top_1_acc, top_5_acc],
              )

"""
Epochs & Batch Sizes
"""
num_epochs = 90
train_batch_size = 128
val_batch_size = 128
test_batch_size = 13

"""
Callback Params
"""
scheduler_params = {'factor': 0.1,
                    'monitor': 'val_categorical_accuracy',
                    'verbose': 1,
                    'mode': 'auto',
                    'patience': 5,
                    'min_lr': 10**(-8),
                    'min_delta': 0.0001}

scheduler = call.ReduceLROnPlateau(**scheduler_params)

log_dir = paths.models + 'alexnet/v{:02d}/logs'.format(version)
checkpoint_dir = paths.models + 'alexnet/v{:02d}/checkpoints'.format(version)

if not path.isdir(checkpoint_dir):
    os.mkdir(checkpoint_dir)

if not path.isdir(log_dir):
    os.mkdir(log_dir)

# Experiment directory format is {model}/{version}/{filetype}
tensorboard_params = {'log_dir': log_dir,
                      'batch_size': train_batch_size,
                      'write_grads': True,
                      'write_images': True}

checkpointer_params = {'filepath': checkpoint_dir + '/weights.{epoch:02d}-{val_loss:.2f}.hdf5',
                       'verbose': 1}

"""
Augmentation Parameters
"""

aug_list = AugmentationList(albumentations.HorizontalFlip())
shift_scale = 0.1


"""
Loading Params
"""

# If model_file is not None and checkpoint_dir is not None then loading happens, else not
loading_params = {'checkpoint_dir': None,
                  'model_file': None,
                  'epoch_start': None}


if __name__ == '__main__':
    print(model.summary())