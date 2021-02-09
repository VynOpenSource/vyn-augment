import os

from bin.generator import set_generator


def set_generator_classifier(root_dir, preprocessing_function=None, batch_size=16, number_of_images=100):
    """
    Set the generator with the data from path. The assumption is that root_dir contains a set of folders each of them
    with images from a different class.
    :param root_dir: The path where the main folder with the data is stored.
    :param preprocessing_function: A function to pre process the images. Generally, it requires a reader, the augmentor
                                and any other process that is useful. For instance, when using deep learning networks
                                normalisation per channel or total should be done.
    :param batch_size: The size of the batch that is returned by the generator. Use batch_size of one to store images
    :param number_of_images: The number of images to produce. This is the number of iterations per epoch when training
                            a deep neural network.
    :return: A generator object
    """
    DataGenerator = set_generator('keras')
    input_data = []
    for folder in os.listdir(root_dir):
        dirname = os.path.join(root_dir, folder)
        for filename in dirname:
            filename = os.path.join(dirname, filename)
            input_data.append((folder, filename))

    # The conditions shuffle_per_label = True and shuffle_all_dataset = False are used to balance the dataset.

    generator = DataGenerator(input_data, batch_size=batch_size, shuffle_per_label=True, shuffle_all_dataset=False,
                              preprocessing_function=preprocessing_function, fix_iterations=number_of_images)


    return generator


def set_generator_segmentor(image_dir, masks_dir, preprocessing_function=None, batch_size=16, number_of_images=100):
    """
    Set the generator with the data from the image and mask directories. The assumption is that the images and the masks
    have the same name.
    :param image_dir: The path where the raw images are stored.
    :param masks_dir: The path where the masks are stored.
    :param preprocessing_function: A function to pre process the images. Generally, it requires a reader, the augmentor
                                and any other process that is useful. For instance, when using deep learning networks
                                normalisation per channel or total should be done.
    :param batch_size: The size of the batch that is returned by the generator. Use batch_size of one to store images
    :param number_of_images: The number of images to produce. This is the number of iterations per epoch when training
                            a deep neural network.
    :return: A generator object
    """
    DataGenerator = set_generator('keras')
    input_data = []
    for filename in image_dir:
        image_filename = os.path.join(image_dir, filename)
        mask_filename = os.path.join(masks_dir, filename)
        input_data.append((mask_filename, image_filename))

    # The conditions shuffle_per_label = True and shuffle_all_dataset = False are used to balance the dataset.
    generator = DataGenerator(input_data, batch_size=batch_size, shuffle_per_label=True, shuffle_all_dataset=False,
                              preprocessing_function=preprocessing_function, fix_iterations=number_of_images)

    return generator


def set_generator_obejct_detector(image_dir, bb_dir, preprocessing_function=None, batch_size=16,
                                  number_of_images=100):
    """
    Set the generator with the data from the image and mask directories. The assumption is that the images and the
    bounding box files have the same name as the images, just with txt extension.
    :param image_dir: The path where the raw images are stored.
    :param bb_dir: The path where the bounding boxes are stored. Bounding boxes are text
    :param preprocessing_function: A function to pre process the images. Generally, it requires a reader, the augmentor
                                and any other process that is useful. For instance, when using deep learning networks
                                normalisation per channel or total should be done.
    :param batch_size: The size of the batch that is returned by the generator. Use batch_size of one to store images
    :param number_of_images: The number of images to produce. This is the number of iterations per epoch when training
                            a deep neural network.
    :return: A generator object
    """
    DataGenerator = set_generator('keras')
    input_data = []
    for filename in image_dir:
        image_filename = os.path.join(image_dir, filename)
        filename[filename.rfind('.'):] = '.txt'
        mask_filename = os.path.join(bb_dir, filename)
        input_data.append((mask_filename, image_filename))

    # The conditions shuffle_per_label = True and shuffle_all_dataset = False are used to balance the dataset.
    generator = DataGenerator(input_data, batch_size=batch_size, shuffle_per_label=True, shuffle_all_dataset=False,
                              preprocessing_function=preprocessing_function, fix_iterations=number_of_images)

    return generator