import json
import os

import numpy as np
from skimage.io import imread, imsave

from utils.data_processing import set_generator_object_detector
from utils.visualisers import plot_bb_image
from src.vyn_augment.augmentor import Augmentor


def pre_processing_function(bb_filename: str, filename: str, augmentor: Augmentor = None):
    """
    Pre-processing function. This function is run within the generator and it is performed to each individual image
    regardless the batch size.
    :param mask_filename: The filename where the mask is stored.
    :param filename: The complete path to the image file
    :param augmentor: An object of type Augmentor
    :return:
    """
    with open(bb_filename, 'r') as f:
        bbs = json.load(f)
    image = imread(filename)

    # Augmentor expects the bounding boxes to be 4 values only instead of 5 with the label, so we are going to
    # remove it and then add it again. In addition, the values must be integers.
    h, w = image.shape[:2]
    sizes = [w, h, w, h]
    classes = []
    bbs2 = []
    for bb_i in bbs:
        classes.append(bb_i[0])
        bbs2.append([int(round(size * coord)) for coord, size in zip(bb_i[1:], sizes)])

    if augmentor is not None:
        output = augmentor.run(image, use_colour=-1, bounding_boxes=[bbs2])
        bbs2 = output['bounding_boxes'][0]
        image = np.round(output['images']).astype(np.uint8)

    # We need to add the labels again
    bbs = []
    for bb_i, class_i in zip(bbs2, classes):
        bbs.append([class_i] + [coord/float(size) for coord, size in zip(bb_i, sizes)])

    label = os.path.basename(filename)
    label = label[:label.rfind('.')]

    return image, [label, bbs]


def set_augmentor():
    """
    Set the augmentor.
    1. Select the operations and create the config dictionary
    2. Pass it to the Augmentor class with any other information that requires
    3. Return the instance of the class.
    :return:
    """
    # Operations that must be only applied inside the bounding boxes have a _object after the operation
    config = {'blur': {'values': ('gaussian', 0.7, 1.0), 'prob': 0.3},
              'brightness': {'values': (0.6, 1.0), 'prob': 0.1},
              'brightness1': {'values': (1.0, 1.5), 'prob': 0.1},
              'flip': {'values': ('hor',), 'prob': 0.5},
              'flip_object': {'values': ('hor',), 'prob': 0.5},
              'grid_mask_object': {'values': (0, 0.2, 0, 0.2, 0.1, 0.15, 0.1, 0.15, 0.2, 0.3, 0.2, 0.3),
                                   'prob': 0.3},
              'occlusion': {'values': ('hide_and_seek', 0.2, 0.4, 0.2, 0.3), 'prob': 0.4},
              'posterisation': {'values': (32, 64), 'prob': 0.5},
              'rotate': {'values': (-45, 45), 'prob': 0.4},
              'translate': {'values': ('RANDOM', -0.2, 0.2), 'prob': 0.2, 'use_replication': False},
              'translate_object': {'values': ('RANDOM', -0.1, 0.1), 'prob': 0.25, 'use_replication': True},
              'zoom': {'values': (0.5, 1.5), 'prob': 0.9, 'use_replication': False},
              'zoom_object': {'values': (0.5, 1.5), 'prob': 0.9, 'use_replication': False}}

    augmentor = Augmentor(config)

    return augmentor


def generate_n_augmented_images(data_dirname: str, root_dirname: str, n=20, plot=False) -> None:
    """
    Generate n new augmented images, where n is an input parameter
    :param data_dirname: The directory where the initial set of images are.
    :param root_dirname: The directory where to save the augmented set of images.
    :param n: The number of augmented images
    :return: None
    """
    input_image_dirname = os.path.join(data_dirname, 'images')
    input_bb_dirname = os.path.join(data_dirname, 'bounding_boxes')

    augmentor = set_augmentor()
    preprocessing_fun = lambda *args: pre_processing_function(*args, augmentor=augmentor)
    generator = set_generator_object_detector(input_image_dirname, input_bb_dirname, preprocessing_fun,
                                        batch_size=1, number_of_images=n)
    generator.not_batch = True
    # NOTICE: This generator can be used for keras and pytorch in case that instead of saving images one desires to
    # augment images on the fly. Use a number larger than 1 for the batch size when training directly a CNN.

    image_dirname = os.path.join(root_dirname, 'images')
    bb_dirname = os.path.join(root_dirname, 'bounding_boxes')
    if not os.path.isdir(image_dirname):
        os.makedirs(image_dirname)
    if not os.path.isdir(bb_dirname):
        os.makedirs(bb_dirname)
    # Save the new generated images
    counter_labels = {}
    for image, info in generator:
        label = info[0]
        bbs = info[1]
        counter = counter_labels.get(label, 0)
        image_filename = label + '_' + str(counter).zfill(4) + '.jpg'
        mask_filename = label + '_' + str(counter).zfill(4) + '.txt'

        filename = os.path.join(image_dirname, image_filename)
        imsave(filename, image)

        filename = os.path.join(bb_dirname, mask_filename)
        with open(filename, 'w') as f:
            json.dump(bbs, f)

        if plot:
            plot_bb_image(image, bbs)

        counter_labels[label] = counter + 1


if __name__ == '__main__':
    data_dirname = '../images/object_detection'
    root_dirname = '../images/output_data/object_detection'

    generate_n_augmented_images(data_dirname, root_dirname, plot=False)
