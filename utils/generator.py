import random
import secrets
from typing import List

import numpy as np
import tensorflow as tf
from torch.utils.data import Dataset, DataLoader


def set_generator(base):
    if base.lower() == 'keras':
        base_class = tf.keras.utils.Sequence
    elif base.lower() == 'pytorch':
        base_class = Dataset
    else:
        raise ValueError('Only keras and pytorch are allowed base classes')

    class DataGenerator(base_class):
        def __init__(self, list_data: List[tuple],
                     batch_size=32, shuffle_all_dataset=True, num_iter=1000, shuffle_per_label=True,
                     fix_iterations=False, preprocessing_function=None):

            """
            Create a generator for a given deep learning library.
            :param list_data (list):  A list of lists or tuples where the first value is the label or mask address
                                    if segmentation or caption if captioning and the rest of the values are the image
                                    and whatever is required. For instance, we could have a list with
                                    (label, image_filename, bounding_box)
            :param batch_size (int): int with batch size
            :param shuffle (boolean): whether to shuffle the data or not
            :param num_iter (int): The number of iterations in one epoch, this is used when either when shuffle_folder
                                    or fix_iterations are set to True. Otherwise, it will use all the available data.
            :param shuffle_folders (boolean): It will select randomly a label and then randomly a data with that label,
                                    this is useful to balance out data, since this does not allow epochs, num_iter
                                    is used. Notice that this should not be used in the case of segmentation.
            :param fix_iterations (boolean):  When True the number of iterations is set to num_iter.
                                            This is not used when shuffle_folders = True, since it cannot use
                                            all the data, so it shuffle_foler=True implies fix_iterations = True
                                            regardless of the user input.
            :param preprocessing_function (fun): A function to perform on the image
            """

            super().__init__()
            'Initialization'
            self.batch_size = batch_size if batch_size > 0 else 1
            self.not_batch = True if batch_size < 0 else False
            self.list_data = list_data
            self.n_classes = None
            self.shuffle_all_dataset = shuffle_all_dataset

            self.num_iter = num_iter
            self.shuffle_per_label = shuffle_per_label
            self.fix_iterations = fix_iterations
            self.epoch_number = 0

            self.iterator = 0

            self.list_data = list_data

            # Storage for iteration. They are created in on_epoch_end
            self.indexes = list(range(len(list_data)))
            self.labels = []
            self.data = {}

            self.preprocessing_function = preprocessing_function
            self.current_files = []
            self.on_epoch_end()

        def __len__(self) -> int:
            """
            Denotes the number of batches per epoch
            :return: The number of batches
            """
            if self.shuffle_per_label or self.fix_iterations:
                return self.num_iter
            else:
                return max(min(1, len(self.indexes)), int(np.floor(len(self.indexes) / self.batch_size)))

        def __getitem__(self, index):
            """
            Return a batch of data. This is composed on a numpy array as a tensor batch_size images and another numpy
            array with the labels.
            :param index: The current batch number to create.
            :return: A tuple with images and labels
            """
            'Generate one batch of data'
            # Generate indexes of the batch
            """indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
            # Find list of IDs
            list_urls_temp = [self.list_urls[k] for k in indexes]"""
            #
            labels = []
            y = []
            i = 0
            self.current_files = []
            while i < self.batch_size:
                if not self.shuffle_per_label:
                    position = self.indexes[self.iterator]
                    label = self.list_data[position][0]
                    values = self.list_data[position][1:]
                else:
                    label = secrets.choice(list(self.data.keys()))
                    all_files = self.data[label]
                    pos = np.random.randint(0, len(all_files), 1)[0]
                    values = all_files[pos]  # list_files_temp.append(all_files[pos])
                    labels.append(label)

                im, label = self._data_generation_single(values, label)
                if im is not None and len(im.shape) > 2:
                    if isinstance(values[0], str):
                        self.current_files.append(values[0])
                    if i == 0:
                        X = np.empty((self.batch_size, *im.shape), dtype=np.float32)

                        for label_i in label:
                            yi = []
                            if hasattr(label_i, 'shape'):
                                yi = np.zeros((self.batch_size, *label_i.shape), dtype=np.float32)
                            y.append(yi)

                    X[i: i + 1, ...] = im

                    for ii, label_i in enumerate(label):
                        if hasattr(label_i, 'shape'):
                            y[ii][i: i+1, ...] = label_i
                        else:
                            y[ii].append(label_i)

                    i += 1
                self.iterator += 1
                if self.iterator >= len(self.indexes):
                    self.iterator = 0

            if base.lower() == 'pytorch':
                X = X.transpose([0, 3, 1, 2])

            if self.not_batch:
                X = X[0, ...]
                for ii, y_i in enumerate(y):
                    y[ii] = y_i[0]

            if len(y) == 1:
                y = y[0]
            elif not self.not_batch:
                for ii, y_i in enumerate(y):
                    y[ii] = np.array(y_i)

            X = X.astype(np.float32)

            return X, y

        def on_epoch_end(self) -> None:
            """
            Updates indexes after each epoch. It takes into consideration whether random shuffling of the whole dataset
            or shuffling of folders is going to be used.
            :return: None
            """
            self.iterator = 0
            # list_data has a list of lists or tuples where the first value is the label or mask address if segmentation
            # or caption if captioning and the rest of the values are the image and whatever is required. For instance,
            # we could have a list with (label, image_filename, bounding_box).
            if self.shuffle_per_label:
                if self.epoch_number == 0:
                    self.data = {}
                    for values in self.list_data:
                        self.data.setdefault(values[0], []).append(values[1:])
                    self.n_classes = len(self.data)
            else:
                if self.shuffle_all_dataset:
                    # Indexes are a list from 0 to the number of elements in self.data. Although, self.data could be
                    # shuffle it is much faster to randomise a list of numbers. The idea is that self.data could
                    # potentially have the image itself and not the address.
                    random.shuffle(self.indexes)

            self.epoch_number += 1

        def _data_generation_single(self, values, label) -> tuple:
            """
            Get one image and one label. This function makes use of the pre_processing function that the user must
            passed.
            :param values: A set of values organised in the way the user wants to use them
            :param label: The label of the data
            :return: Image as a numpy array and a label
            """
            try:
                output = self.preprocessing_function(label, *values)
                if output is None:
                    return None, None

                if isinstance(output, np.ndarray):
                    im = output
                    y = label
                else:
                    im, y = output

                if isinstance(y, (str, tuple, np.ndarray)):
                    y = [y]

            except Exception as e:
                print(f"Error at handling image: {values[0]}. The error was: {repr(e)}")
                return None, None

            if im is not None:
                return im, y

            return None, None

    return DataGenerator
