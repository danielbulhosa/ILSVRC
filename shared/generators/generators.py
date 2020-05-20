import cupy as cp
import os
import time
import re
import numpy as np
import pickle
from shared.generators.imagenet_sequence import ImagenetSequence
from shared.generators.augmentation_list import AugmentationList
import shared.definitions.paths as paths

"""Load pixel averages"""
results = None
with open(paths.constants + 'pixel_avg.pkl', 'rb') as file:
    results = pickle.load(file)

pixel_avg = results['pixel_avg']

"""Load pixel covariances"""
# The covariances roughly align with the precalculated values used here
# https://medium.com/@kushajreal/training-alexnet-with-tips-and-checks-on-how-to-train-cnns-practical-cnns-in-pytorch-1-61daa679c74a
cov_results = None
with open(paths.constants + 'cov.pkl', 'rb') as file:
    cov_results = pickle.load(file)

cov = cov_results['cov']
# Note that the eigenvectors returned are normalized
eigenvalues, eigenvectors = cp.linalg.eigh(cov/255**2)
stdev = cp.sqrt(cov.diagonal()/255**2)

"""Map Classes To Names & One-Hot Encodings"""
with open(paths.dataset + 'sysnet_words.txt', 'r') as file:
    words = file.readlines()

codes = [re.match('n[0-9]*', word).group() for word in words]
labels = [word.replace(code, '').strip() for word, code in zip(words, codes)]
code_label_map = dict(zip(codes, labels))
index_code_map = dict(enumerate(codes))
code_index_map = {code: index for index, code in index_code_map.items()}

# Get list of training and validation examples and labels
training_dir = paths.training
validation_dir = paths.validation


def get_paths_and_classes(root_dir, num_classes, code_index_map=code_index_map):
    """
    Get paths and classes for training data. This gets passed to custom
    generator to pull image files.

    :param root_dir: The directory from which to pull the images.
    :param nclasses: Number of classes to pull. Set to less than 1,000 for debugging.
    :param cim: The code index map used to map class codes to class indices.
    :return: Paths to image files and corresponding array of one-hot encoded classes.
    """

    classes = []
    example_paths = []

    assert 0 < num_classes <= 1000, "Number of classes must be between 1 and 1000."

    class_dirs = [class_dir for class_dir in os.listdir(root_dir)
                  if class_dir[-4:] != ".tar"]
    class_dirs = class_dirs if num_classes == 1000 else class_dirs[:num_classes]

    # Reindex classes to fit reduced number of classes when applicable
    code_index_map = code_index_map if num_classes == 1000 else dict(zip(class_dirs, range(num_classes)))

    for num, class_dir in enumerate(class_dirs):

        full_class_path = root_dir + class_dir + '/'
        class_images = [class_dir + '/' + image for image in os.listdir(full_class_path)
                        if image[-5:] == ".JPEG"]

        example_paths.extend(class_images)
        class_index_list = [code_index_map[class_dir]]
        classes.extend(len(class_images) * class_index_list)

        assert len(example_paths) == len(classes), "Number of labels must match number of examples"

    # https://stackoverflow.com/questions/38592324/one-hot-encoding-using-numpy
    classes_array = np.array([classes]).reshape(-1)
    one_hot_classes = np.eye(num_classes)[classes_array]

    return example_paths, one_hot_classes


def get_train_gen(batch_size, nclasses, num_datapoints, shift_scale, aug_list):
    x_paths_train, y_labels_train = get_paths_and_classes(training_dir, nclasses)
    train_indices = np.random.choice(np.arange(len(x_paths_train)), size=len(x_paths_train), replace=False)
    # Shuffle parameter in fit generator only shuffles minibatch order. Without shuffling ahead
    # of time mini-batches will tend to be extremely correlated (i.e. same class) which will make
    # our gradient estimate biased.
    x_paths_train, y_labels_train = np.array(x_paths_train)[train_indices].tolist(), y_labels_train[train_indices]

    # Filter number of minibatches processed for debugging
    return ImagenetSequence(x_paths_train if num_datapoints is None else x_paths_train[:num_datapoints],
                           y_labels_train if num_datapoints is None else y_labels_train[:num_datapoints],
                           batch_size,
                           paths.training,
                           np.array(eigenvectors.tolist()),
                           np.array(eigenvalues.tolist()),
                           np.array(pixel_avg.tolist()),
                           np.array(stdev.tolist()),
                           shift_scale,
                           aug_list,
                           'train')


def get_val_gen(batch_size, nclasses, num_datapoints):
    x_paths_val, y_labels_val = get_paths_and_classes(validation_dir, nclasses)
    val_indices = np.random.choice(np.arange(len(x_paths_val)), size=len(x_paths_val), replace=False)
    # Shuffle parameter in fit generator only shuffles minibatch order. Without shuffling ahead
    # of time mini-batches will tend to be extremely correlated (i.e. same class) which will make
    # our gradient estimate biased.
    x_paths_val, y_labels_val = np.array(x_paths_val)[val_indices].tolist(), y_labels_val[val_indices]

    # Filter number of minibatches processed for debugging
    return ImagenetSequence(x_paths_val if num_datapoints is None else x_paths_val[:num_datapoints],
                           y_labels_val if num_datapoints is None else y_labels_val[:num_datapoints],
                           batch_size,
                           paths.validation,
                           np.array(eigenvectors.tolist()),
                           np.array(eigenvalues.tolist()),
                           np.array(pixel_avg.tolist()),
                           np.array(stdev.tolist()),
                           0,
                           AugmentationList(*[]),
                           'validate')


def get_test_gen(batch_size, nclasses):
    x_paths_val, y_labels_val = get_paths_and_classes(validation_dir, nclasses)
    val_indices = np.random.choice(np.arange(len(x_paths_val)), size=len(x_paths_val), replace=False)
    # Shuffle parameter in fit generator only shuffles minibatch order. Without shuffling ahead
    # of time mini-batches will tend to be extremely correlated (i.e. same class) which will make
    # our gradient estimate biased.
    x_paths_val, y_labels_val = np.array(x_paths_val)[val_indices].tolist(), y_labels_val[val_indices]

    return ImagenetSequence(x_paths_val,
                           y_labels_val,
                           batch_size,  # Actual batch size is 10x because of TTAs
                           paths.validation,
                           np.array(eigenvectors.tolist()),
                           np.array(eigenvalues.tolist()),
                           np.array(pixel_avg.tolist()),
                           np.array(stdev.tolist()),
                           0,
                           AugmentationList(*[]),
                           'test')