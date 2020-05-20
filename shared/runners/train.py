# Set memory restrictions to allow some room for our monitor to render :)
from keras import backend as K
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.95
session = tf.Session(config=config)
K.set_session(session=session)

import keras.callbacks as call
import gc
from shared.generators.generators import get_train_gen, get_val_gen
print("Assemble & Compile Model")
import sys
import importlib
rundef_path = sys.argv[1]
print(rundef_path)
rundef = importlib.import_module(rundef_path)  # Run definition

"""Model Loading (If Applicable)"""
checkpoint_dir = rundef.loading_params['checkpoint_dir']
model_file = rundef.loading_params['model_file']
epoch_start = rundef.loading_params['epoch_start']

if model_file is not None and checkpoint_dir is not None:
    print("Loading Model")
    rundef.model.load_weights(checkpoint_dir + model_file)
    epochs = rundef.num_epochs - epoch_start
else:
    epoch_start = 0
    epochs = rundef.num_epochs

"""Callbacks"""
scheduler = rundef.scheduler  # Scheduler class can vary
tensorboard = call.TensorBoard(**rundef.tensorboard_params)
checkpointer = call.ModelCheckpoint(**rundef.checkpointer_params)

# Added to address OOM Error - not sure if needed anymore
# See: https://github.com/keras-team/keras/issues/3675
garbage_collection = call.LambdaCallback(on_epoch_end=lambda epoch, logs: gc.collect())

print("Create Generators")
"""Generators"""
train_gen = get_train_gen(rundef.train_batch_size, rundef.num_classes, rundef.num_datapoints, rundef.shift_scale, rundef.aug_list)
val_gen = get_val_gen(rundef.val_batch_size, rundef.num_classes, rundef.num_datapoints)


""" Model train code """
print("Begin Training Model")
rundef.model.fit_generator(train_gen,
                           epochs=rundef.num_epochs,
                           steps_per_epoch=rundef.steps_per_epoch,
                           validation_data=val_gen,
                           verbose=1,  # 0 in notebook, verbose doesn't slow down training, we checked
                           callbacks=[tensorboard,
                                      scheduler,
                                      checkpointer,
                                      garbage_collection,
                                      ],
                           shuffle=True, # Note this only works because we removed step parameters
                           use_multiprocessing=False, # Actually significantly faster when this is false.
                           workers=4,  # Optimal: More workers doesn't increase images/second, makes it slightly slower
                           max_queue_size=4,  # Optimal: A larger queue seems to not make much of a difference
                           initial_epoch=epoch_start
                          )
