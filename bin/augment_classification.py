"""
This file as well as the other examples will store the augmented images in the vyn-augment/images/output_data folder.
On the other hand, the folder notebooks contains the same examples with the images being plotted in the same file
instead of being saved in memory. In addition, a better explanation about the augmentor options is provided.
"""
import os

import numpy as np
from skimage.io import imread, imsave

from utils.data_processing import set_generator_classifier
from src.vyn_augment.augmentor import Augmentor


def pre_processing_function(label, filename: str, augmentor: Augmentor = None):
    """
    Pre-processing function. This function is run within the generator, which calls it for each individual image
    regardless the batch size.
    :param label: Anything that can be used as a identification for the image type.
    :param filename: The complete path to the image file. This function will read the image into a numpy array.
    :param augmentor: An object of type Augmentor
    :return:
    """
    image = imread(filename)
    if augmentor is not None:
        image = np.round(augmentor.run(image)).astype(np.uint8)

    return image, label


def set_augmentor():
    """
    Set the augmentor.
    1. Select the operations and create the config dictionary
    2. Pass it to the Augmentor class with any other information that requires
    3. Return the instance of the class.
    :return:
    """
    config = {'blur': {'values': ('gaussian', 0.7, 1.0), 'prob': 0.3},
              'brightness': {'values': (0.6, 1.0), 'prob': 0.1},
              'brightness1': {'values': (1.0, 1.5), 'prob': 0.1},
              'flip': {'values': ('hor',), 'prob': 0.5},
              'grid_mask': {'values': (0, 0.2, 0, 0.2, 0.01, 0.1, 0.01, 0.1, 0.1, 0.2, 0.1, 0.2), 'prob': 0.4},
              'illumination': {'values': ('blob_negative', 0.1, 0.2, 100, 150), 'prob': 0.2},
              'noise': {'values': (2, 10), 'use_gray_noise': True, 'prob': 1},
              'rotate': {'values': (-45, 45), 'prob': 0.4},
              'translate': {'values': ('RANDOM', -0.2, 0.2), 'prob': 0.2, 'use_replication': True},
              'zoom': {'values': (0.5, 1.5), 'prob': 0.9, 'use_replication': True}}

    augmentor = Augmentor(config, no_repetition=True)

    return augmentor


def generate_n_augmented_images(data_dirname: str, root_dirname: str, n=20) -> None:
    """
    Generate n new augmented images, where n is an input parameter
    :param data_dirname: The directory where the initial set of images are.
    :param root_dirname: The directory where to save the augmented set of images.
    :param n: The number of augmented images
    :return: None
    """
    augmentor = set_augmentor()
    preprocessing_fun = lambda *args: pre_processing_function(*args, augmentor=augmentor)
    generator = set_generator_classifier(data_dirname, preprocessing_fun, batch_size=1, number_of_images=n)
    generator.not_batch = True
    # NOTICE: This generator can be used for keras and pytorch in case that instead of saving images one desires to
    # augment images on the fly. Use a number larger than 1 for the batch size when training directly a CNN.

    # Save the new generated images
    counter_labels = {}
    for image, label in generator:
        counter = counter_labels.get(label, 0)
        output_filename = label + '_' + str(counter_labels.get(label, 0)) + '.jpg'
        save_dirname = os.path.join(root_dirname, label)
        if not os.path.isdir(save_dirname):
            os.makedirs(save_dirname)

        filename = os.path.join(save_dirname, output_filename)
        imsave(filename, image.astype(np.uint8))
        counter_labels[label] = counter + 1

    print(f'Finished image generation. The output images were saved in {root_dirname}')


if __name__ == '__main__':
    root = os.path.dirname(os.path.dirname(__file__))
    data_dirname = os.path.join(root, 'images', 'classification')
    root_dirname = os.path.join(root, 'images', 'output_data', 'classification')

    generate_n_augmented_images(data_dirname, root_dirname)
