import keras.models as mod
import keras.layers as lyr
import keras.optimizers as opt
import keras.metrics as met
import keras.callbacks as call
import tensorflow as tf
import keras.losses as losses
import keras.regularizers as reg
import keras.initializers as init
import shared.custom_layers.residual_unit as custom
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
num_batches = None

init_reg = {
    'kernel_initializer': init.he_uniform(),
    'bias_initializer': init.Zeros(),
    'kernel_regularizer': reg.l2(0.0001),
    'bias_regularizer': reg.l2(0.0001)
}

# Number of repetitions of each type of residual unit
convx_rep = [2, 2, 2, 2]
# Parameters for convolutions in residual unit, format (filter_size, feature_maps)
convx_spec = [(3, 64), (3,128), (3, 256), (3, 512)]

input = lyr.Input((224, 224, 3,))
conv1 = lyr.Conv2D(64, 7, strides=2, padding='same', **init_reg)(input)
maxout1 = lyr.MaxPool2D(3, 2, padding='same')(conv1)

current_intermediate = maxout1
initial = True

# Dynamically generate convN_x for N from 2 to 5. See: https://arxiv.org/pdf/1512.03385.pdf
# This allows us to switch between the smaller ResNet depths, 18 and 34
for reps, spec in zip(convx_rep, convx_spec):
    for rep in range(reps):

        if not initial and rep == 0:
            downsample = True
        else:
            downsample = False
            initial = False

        size = spec[0]
        filters = spec[1]
        current_intermediate = custom.residual_unit(current_intermediate, size, filters, downsample, **init_reg)

avg_pool = lyr.AvgPool2D(7, 1, padding='valid')(current_intermediate)
flatten = lyr.Flatten()(avg_pool)
out = lyr.Dense(num_classes, activation='relu', **init_reg)(flatten)

model = mod.Model(inputs=input, outputs=out)


"""
Optimizer, Loss, & Metrics

Using zero built in decay and relying exclusively on the heuristic
used in the original paper.
"""
optimizer = opt.Adam(0.001)
# This line below is what was used in the original ResNet paper. Adam is the recommended approach now though.
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
num_epochs = 120  # Maximum, equivalent to 60 x 10^4 iterations where iteration = step and minibatch size is 256
train_batch_size = 256
val_batch_size = 256
test_batch_size = 26

"""
Callback Params
"""

# Paper does not specify what the patience was when training the original model
scheduler_params = {'factor': 0.1,  # Reduce by factor of 10
                    'monitor': 'val_categorical_accuracy',  # They monitor the error not the loss
                    'verbose': 1,
                    'mode': 'auto',
                    'patience': 5,
                    'min_lr': 10**(-8),
                    'min_delta': 0.0001}

scheduler = call.ReduceLROnPlateau(**scheduler_params)

log_dir = paths.models + 'resnet/v{:02d}/logs'.format(version)
checkpoint_dir = paths.models + 'resnet/v{:02d}/checkpoints'.format(version)

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

# Referred to as the "standard color augmentation" in original ResNet paper
shift_scale = 0.1


"""
Loading Params
"""

# If model_file is not None and checkpoint_dir is not None then loading happens, else not
loading_params = {'checkpoint_dir': None,
                  'model_file': None,
                  'epoch_start': None}


if __name__ == '__main__':
    # FIXME: 13.075M parameters vs. 11.4 parameters in paper - why difference? Could just be the 1x1 convolutions...
    print(model.summary())