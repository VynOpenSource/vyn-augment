"""
This file as well as the other examples will store the augmented images in the vyn-augment/images/output_data folder.
On the other hand, the folder notebooks contains the same examples with the images being plotted in the same file
instead of being saved in memory. In addition, a better explanation about the augmentor options is provided.
"""
import os

import numpy as np
from skimage.io import imread, imsave

from utils.data_processing import set_generator_segmentor
from src.vyn_augment.augmentor import Augmentor


def pre_processing_function(mask_filename: str, filename: str, augmentor: Augmentor = None):
    """
    Pre-processing function. This function is run within the generator, which calls it for each individual image
    regardless the batch size.
    :param mask_filename: The filename where the mask is stored. This function will read the image into a numpy array.
    :param filename: The complete path to the image file. This function will read the image into a numpy array.
    :param augmentor: An object of type Augmentor
    :return:
    """
    image = imread(filename)
    mask = imread(mask_filename)
    if augmentor is not None:
        image, mask = augmentor.run([image, mask], mask_positions=[1])
        image = np.round(image).astype(np.uint8)
        mask = np.round(mask).astype(np.uint8)

    # Returning the name or label is not mandatory, it will just be used to give a name to the file when saving the
    # augmented image
    name = os.path.basename(filename)
    name = name[:name.rfind('.')]
    return image, [mask, name]


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
              'flip': {'values': ('random',), 'prob': 0.5},
              'occlusion': {'values': ('hide_and_seek', 0.2, 0.4, 0.2, 0.3), 'prob': 0.4},
              'posterisation': {'values': (32, 64), 'prob': 0.5},
              'rotate': {'values': (-45, 45), 'prob': 0.4},
              'translate': {'values': ('RANDOM', -0.2, 0.2), 'prob': 0.2, 'use_replication': False},
              'zoom': {'values': (0.5, 1.5), 'prob': 0.9, 'use_replication': False}}

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
    input_image_dirname = os.path.join(data_dirname, 'images')
    input_mask_dirname = os.path.join(data_dirname, 'masks')

    augmentor = set_augmentor()
    preprocessing_fun = lambda *args: pre_processing_function(*args, augmentor=augmentor)
    generator = set_generator_segmentor(input_image_dirname, input_mask_dirname, preprocessing_fun,
                                        batch_size=1, number_of_images=n)
    generator.not_batch = True
    # NOTICE: This generator can be used for keras and pytorch in case that instead of saving images one desires to
    # augment images on the fly. Use a number larger than 1 for the batch size when training directly a CNN.

    image_dirname = os.path.join(root_dirname, 'images')
    mask_dirname = os.path.join(root_dirname, 'masks')
    if not os.path.isdir(image_dirname):
        os.makedirs(image_dirname)
    if not os.path.isdir(mask_dirname):
        os.makedirs(mask_dirname)
    # Save the new generated images
    counter_labels = {}
    for image, info in generator:
        mask = info[0]
        name = info[1]
        counter = counter_labels.get(name, 0)
        image_filename = name + '_' + str(counter).zfill(4) + '.jpg'
        mask_filename = name + '_' + str(counter).zfill(4) + '.png'

        filename = os.path.join(image_dirname, image_filename)
        imsave(filename, image.astype(np.uint8))

        filename = os.path.join(mask_dirname, mask_filename)
        imsave(filename, mask.astype(np.uint8))

        counter_labels[name] = counter + 1


if __name__ == '__main__':
    data_dirname = '../images/segmentation'
    root_dirname = '../images/output_data/segmentation'

    generate_n_augmented_images(data_dirname, root_dirname)
