import keras.models as mod
import keras.layers as lyr
import keras.optimizers as opt
import keras.metrics as met
import keras.callbacks as call
import tensorflow as tf
import keras.losses as losses
import keras.regularizers as reg
import keras.initializers as init
import shared.custom_layers.inception_module as custom
import shared.definitions.paths as paths
from shared.generators.augmentation_list import AugmentationList
import albumentations
import os
from os import path

"""
Model Definition
"""
version = 0
num_classes = 1000
num_datapoints = None
steps_per_epoch = None

# Can't find initialization or regularization of weights in paper. Using same as ResNet.
init_reg = {
    'kernel_initializer': init.he_uniform(),
    'bias_initializer': init.Zeros(),
    'kernel_regularizer': reg.l2(0.0001),
    'bias_regularizer': reg.l2(0.0001)
}

# Note we're not adding the auxiliary classifiers, just to keep it simple.
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
out = lyr.Dense(num_classes, activation='softmax', **init_reg)(lyrout18)

model = mod.Model(inputs=input, outputs=out)



"""
Optimizer, Loss, & Metrics

Using zero built in decay and relying exclusively on the heuristic
used in the original paper.
"""
optimizer = opt.Adam(0.001)
# This line below is what was used in the original ResNet paper. Adam is the recommended approach now though.
# We use this for GoogLeNet because its paper only specifies that they used momentum=0.9 but not what the initial
# learning rate was.
# optimizer = opt.SGD(learning_rate=0.1, momentum=0.9, decay=0.0) # Note decay refers to learning rate

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
# Can't find references to this in the paper. Using what we used for ResNet.
# No references for number of epochs to just leaving it at 90.
num_epochs = 90
train_batch_size = 256
val_batch_size = 256
test_batch_size = 26

"""
Callback Params
"""

scheduler_params = {'schedule': lambda epoch, lr: 0.96 * lr if (epoch + 1) % 4 == 0 else lr,
                    'verbose': 1}

scheduler = call.LearningRateScheduler(**scheduler_params)

log_dir = paths.models + 'googlenet/v{:02d}/logs'.format(version)
checkpoint_dir = paths.models + 'googlenet/v{:02d}/checkpoints'.format(version)

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

# These are the augmentations from the GoogLeNet paper, which come from: https://arxiv.org/pdf/1312.5402.pdf
# We do not know how the perturbation magnitude in paper maps to albumentations so just used library defaults

# Furthermore, the GoogLeNet paper recommends "sampling of various sized patches of the image whose size is distributed
# evenly between 8% and 100% of the image area with aspect ratio constrained to the interval [3/4, 4/3]."
# We upscale the image by 3.5 so that the 224x224 sampled patch has 8% area of original image. We scale
# uniformly though.
aug_list = AugmentationList(albumentations.RandomBrightnessContrast(p=1),
                            albumentations.RGBShift(p=1),
                            albumentations.RandomScale(scale_limit=(1, 3.5), p=1),
                            albumentations.HorizontalFlip(),
                            shuffle=True)

shift_scale = 0.1


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