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

version = 27
num_classes = 1000
num_batches = None

# Changing k and alpha like this makes the output of LRN O(1) when inputs are standardized Gaussian
# Note that n bounds the number of terms in the sum which bounds the order of the denominator independent of # maps
k, n, alpha, beta = 0.2, 5, 0.1, 0.75   # FIXME CHANGE # 5: Increase alpha to 0.1 from 10**(-4), decrease k to 0.2
lrn = lambda tensor: lrn_parametric(tensor, k, n, alpha, beta)

model = mod.Sequential(
 [# We input 224 x 224 pixel images with 3 channels (rgb)
  # We create 96 feature maps, by using an 11x11 pixel kernel with stride 4
  lyr.Conv2D(96, 11, strides=4, padding='same',
             input_shape=(224, 224, 3,), activation='relu',
             kernel_initializer=init.he_uniform(),  # tried: init.TruncatedNormal(0.01), original: RandomNormal(stdev=0.01)
             bias_initializer=init.Zeros(),
             kernel_regularizer=reg.l2(0.0001), # FIXME CHANGE # 4: Reduce bias regularization to 0.0001 from 0.0005
             bias_regularizer=reg.l2(0.0001)  # FIXME CHANGE # 3: Reduce bias regularization to 0.0001 from 0.005
             ),
  # The lambdas here are the local response normalization
  lyr.Lambda(lrn, output_shape=lrn_shape),
  lyr.MaxPool2D(3, 2),
  lyr.Conv2D(256, 5, strides=1, padding='same', activation='relu',
             kernel_initializer=init.he_uniform(),  # tried: init.TruncatedNormal(0.01), original: RandomNormal(stdev=0.01)
             bias_initializer=init.Zeros(), # FIXME CHANGE #1 make into zeroes - changed
             kernel_regularizer=reg.l2(0.0001), # FIXME CHANGE # 4: Reduce bias regularization to 0.0001 from 0.0005
             bias_regularizer=reg.l2(0.0001)  # FIXME CHANGE # 3: Reduce bias regularization to 0.0001 from 0.005
             ),
  lyr.Lambda(lrn, output_shape=lrn_shape),
  lyr.MaxPool2D(3, 2),
  lyr.Conv2D(384, 3, strides=1, padding='same', activation='relu',
             kernel_initializer=init.he_uniform(),  # tried: init.TruncatedNormal(0.01), original: RandomNormal(stdev=0.01)
             bias_initializer=init.Zeros(),
             kernel_regularizer=reg.l2(0.0001), # FIXME CHANGE # 4: Reduce bias regularization to 0.0001 from 0.0005
             bias_regularizer=reg.l2(0.0001)  # FIXME CHANGE # 3: Reduce bias regularization to 0.0001 from 0.005
             ),
  #lyr.Lambda(lrn, output_shape=lrn_shape),   # FIXME CHANGE #9: add LRN (previously batch normalization)
  #lyr.Dropout(0.2),   # FIXME CHANGE # 7: extra Add dropout layers
  lyr.Conv2D(384, 3, strides=1, padding='same', activation='relu',
             kernel_initializer=init.he_uniform(),  # tried: init.TruncatedNormal(0.01), original: RandomNormal(stdev=0.01)
             bias_initializer=init.Zeros(), # FIXME CHANGE #1 make into zeroes - changed
             kernel_regularizer=reg.l2(0.0001), # FIXME CHANGE # 4: Reduce bias regularization to 0.0001 from 0.0005
             bias_regularizer=reg.l2(0.0001)  # FIXME CHANGE # 3: Reduce bias regularization to 0.0001 from 0.005
             ),
  #lyr.Lambda(lrn, output_shape=lrn_shape),   # FIXME CHANGE #9: add LRN (previously batch normalization)
  #lyr.Dropout(0.2),   # FIXME CHANGE # 7: extra Add dropout layers
  lyr.Conv2D(256, 3, strides=1, padding='same', activation='relu',
             kernel_initializer=init.he_uniform(),  # tried: init.TruncatedNormal(0.01), original: RandomNormal(stdev=0.01)
             bias_initializer=init.Zeros(), # FIXME CHANGE #1 make into zeroes - changed
             kernel_regularizer=reg.l2(0.0001), # FIXME CHANGE # 4: Reduce bias regularization to 0.0001 from 0.0005
             bias_regularizer=reg.l2(0.0001)  # FIXME CHANGE # 3: Reduce bias regularization to 0.0001 from 0.005
             ),
  # Note from the diagram there is one last pool op after the last conv
  lyr.MaxPool2D(3,2),
  #lyr.Lambda(lrn, output_shape=lrn_shape),   # FIXME CHANGE #9: add LRN (previously batch normalization)
  lyr.Flatten(),
  #lyr.Dropout(0.2),  # FIXME CHANGE # 7: extra Add dropout layers
  lyr.Dense(4096, activation='relu',
            kernel_initializer=init.he_uniform(), # tried: init.TruncatedNormal(0.01), original: RandomNormal(stdev=0.01)
            bias_initializer=init.Zeros(), # FIXME CHANGE #1 make into zeroes - changed
            kernel_regularizer=reg.l2(0.0001),  # FIXME CHANGE # 4: Reduce bias regularization to 0.0001 from 0.0005
            bias_regularizer=reg.l2(0.0001)  # FIXME CHANGE # 3: Reduce bias regularization to 0.0001 from 0.005
            ),
  # The paper says the first two dense layers are regularized with dropout
  #lyr.BatchNormalization(),   # FIXME CHANGE #8: add batch normalization
  lyr.Dropout(0.5),
  lyr.Dense(4096, activation='relu',
            kernel_initializer=init.he_uniform(), # tried: init.TruncatedNormal(0.01), original: RandomNormal(stdev=0.01)
            bias_initializer=init.Zeros(), # FIXME CHANGE #1 make into zeroes - changed
            kernel_regularizer=reg.l2(0.0001),  # FIXME CHANGE # 4: Reduce bias regularization to 0.0001 from 0.0005
            bias_regularizer=reg.l2(0.0001)  # FIXME CHANGE # 3: Reduce bias regularization to 0.0001 from 0.005
            ),
  #lyr.BatchNormalization(),  # FIXME CHANGE #8: add batch normalization
  lyr.Dropout(0.5),
  lyr.Dense(1000,
            activation='softmax',
            kernel_initializer=init.he_uniform(), # tried: init.TruncatedNormal(0.01), original: RandomNormal(stdev=0.01)
            bias_initializer=init.Zeros(), # FIXME CHANGE #1 make into zeroes - changed
            kernel_regularizer=reg.l2(0.0001),  # FIXME CHANGE # 4: Reduce bias regularization to 0.0001 from 0.0005
            bias_regularizer=reg.l2(0.0001)  # FIXME CHANGE # 3: Reduce bias regularization to 0.0001 from 0.005
            ),
 ]
)


"""
Using zero built in decay and relying exclusively on the heuristic
used in the original paper.
"""
optimizer = opt.Adam(0.0001) # FIXME CHANGE #2 - reduce learning rate to 0.0001 from 0.001
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
# FIXME CHANGE #11 - Divide learning rate by 2
scheduler_params = {'factor': 0.1,
                    'monitor': 'val_categorical_accuracy',  # FIXME - change back to monitoring loss?
                    'verbose': 1,
                    'mode': 'auto',
                    'patience': 5,
                    'min_lr': 10**(-8),
                    'min_delta': 0.0001}

scheduler = call.ReduceLROnPlateau(**scheduler_params)

# Experiment directory format is {model}/{version}/{filetype}
tensorboard_params = {'log_dir': paths.models + 'alexnet/v{:02d}/logs'.format(version),
                      'batch_size': train_batch_size,
                      'write_grads': True,
                      'write_images': True}

checkpointer_params = {'filepath': paths.models + 'alexnet/v{:02d}/checkpoints'.format(version)
                                   + '/weights.{epoch:02d}-{val_loss:.2f}.hdf5',
                       'verbose': 1}

"""
Augmentation Parameters
"""

# FIXME CHANGE #10 - add this augmentation
aug_list = AugmentationList(albumentations.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=25),
                            albumentations.HorizontalFlip())

# FIXME CHANGE #6: Increased to 1 from 0.1
shift_scale = 1

"""
Loading Params
"""

loading_params = {'checkpoint_dir': None,
                  'model_file': None,
                  'epoch_start': None}

if __name__ == '__main__':
    print(model.summary())