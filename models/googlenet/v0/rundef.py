import keras.models as mod
import keras.layers as lyr
import keras.optimizers as opt
import keras.metrics as met
import tensorflow as tf
import keras.losses as losses
import keras.regularizers as reg
import keras.initializers as init
import shared.custom_layers.inception_module as custom
import shared.definitions.paths as paths
import albumentations
import os
from os import path

"""
Model Definition
"""
version = 0

# FIXME: Initialization and regularization not clearly specified in paper
init_reg = {
    'kernel_initializer': init.he_uniform(),
    'bias_initializer': init.Zeros(),
    'kernel_regularizer': reg.l2(0.0005),
    'bias_regularizer': reg.l2(0.0005)
}

input = lyr.Input((224, 224, 3,))
lyrout1 = lyr.Conv2D(64, 7, strides=2, padding='same', activation='relu', **init_reg)(input)
lyrout2 = lyr.MaxPool2D(3, 2, 'same')(lyrout1)
lyrout3 = lyr.Conv2D(192, 3, strides=1, padding='same', activation='relu', **init_reg)(lyrout2)
lyrout4 = lyr.MaxPool2D(3, 2, 'same')(lyrout3)
lyrout5 = custom.inception(lyrout4, 64, 96, 128, 16, 32, 32, **init_reg)
lyrout6 = custom.inception(lyrout5, 128, 128, 192, 32, 96, 64, **init_reg)
lyrout7 = lyr.MaxPool2D(3, 2, 'same')(lyrout6)
lyrout8 = custom.inception(lyrout7, 192, 96, 208, 16, 48, 64, **init_reg)
lyrout9 = custom.inception(lyrout8, 160, 112, 224, 24, 64, 64, **init_reg)
lyrout10 = custom.inception(lyrout9, 128, 128, 256, 24, 64, 64, **init_reg)
lyrout11 = custom.inception(lyrout10, 112, 144, 288, 32, 64, 64, **init_reg)
lyrout12 = custom.inception(lyrout11, 256, 160, 320, 32, 128, 128, **init_reg)
lyrout13 = lyr.MaxPool2D(3, 2, 'same')(lyrout12)
lyrout14 = custom.inception(lyrout13, 256, 160, 320, 32, 128, 128, **init_reg)
lyrout15 = custom.inception(lyrout14, 384, 192, 384, 48, 128, 128, **init_reg)
lyrout16 = lyr.AvgPool2D(7, 1, 'valid')(lyrout15)
lyrout17 = lyr.Flatten()(lyrout16)
lyrout18 = lyr.Dropout(0.4)(lyrout17)
out = lyr.Dense(1000, activation='softmax', **init_reg)(lyrout18)

model = mod.Model(inputs=input, outputs=out)



"""
Optimizer, Loss, & Metrics

Using zero built in decay and relying exclusively on the heuristic
used in the original paper.
"""
# FIXME: Is this a good starting optimizer?
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
# FIXME: Number of epochs for this model?
num_epochs = 90
train_batch_size = 128
val_batch_size = 128
test_batch_size = 13

"""
Callback Params
"""

# FIXME - start with defaults for this model?
scheduler_params = {'factor': 0.1,
                    'monitor': 'val_loss',
                    'verbose': 1,
                    'mode': 'auto',
                    'patience': 5,
                    'min_lr': 10**(-8),
                    'min_delta': 0.0001}

if not path.isdir('checkpoints'):
    os.mkdir('checkpoints')

if not path.isdir('logs'):
    os.mkdir('logs')

# Experiment directory format is {model}/{version}/{filetype}
tensorboard_params = {'log_dir': paths.models + 'googlenet/v{:02d}/logs'.format(version),
                      'batch_size': train_batch_size,
                      'write_grads': True,
                      'write_images': True}

checkpointer_params = {'filepath': paths.models + 'googlenet/v{:02d}/checkpoints'.format(version)
                                   + '/weights.{epoch:02d}-{val_loss:.2f}.hdf5',
                       'verbose': 1}

"""
Augmentation Parameters
"""

# FIXME: Which augmentations to start with for GoogLeNet?
aug_list = [albumentations.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=25),
            albumentations.HorizontalFlip()]

# FIXME: Do we want this augmentation for this model?
shift_scale = 1


"""
Loading Params
"""

# If model_file is not None and checkpoint_dir is not None then loading happens, else not
loading_params = {'checkpoint_dir': None,
                  'model_file': None,
                  'epoch_start': None}


if __name__ == '__main__':
    # FIXME: 6.994M parameters vs. 6.798 parameters in paper - why difference?
    print(model.summary())