import math
import os
from typing import List, Union

import numpy as np
from PIL import Image, ImageOps, ImageEnhance
from skimage.io import imread
from skimage.filters import gaussian, median
from skimage.transform import resize as imresize

from src.vyn_augment.image import (skew,
                                   checker,
                                   swap_patches,
                                   check_range,
                                   create_circular_mask,
                                   convert_to_absolute,
                                   create_grid_masks)


class Augmentor(object):
    """
    Modify images according to some augmenting functions. The input is a dictionary with a set of options. The keys are
    the operations and the values the range or type of operations. Another option is to pass a dictionary as value where
    values is the key for the values. This allows to pass other dta such as probability that specifies the probability of
    applying the operations, otherwise is set to 1.

    Some methods are rewritten from
    https://github.com/mdbloice/Augmentor/blob/master/Augmentor/Operations.py
    """

    def __init__(self, operations, std_noise=50, no_repetition=False, seed=None):
        """
        Use seed if you want to apply the same transformation to different group of images. For instance, when images and
        masks need to be processed.
        :param operations: This a dictionary with all the operations. The key are the operations and the values the parameters.
                        Possible keys and the expected value
                         - blur: (min_value, max_value) The values of the kernel. In the case of Gaussian, this represents
                                    the range of the standard deviation.
                         - brightness: (min_value, max_value) The values must for brightness must be between 0.05 and 10
                         - colour_balance: (min_value, max_value) colour_balance must be between 0 and 10
                         - contrast: (min_value, max_value) contrast must be between 0 and 10
                         - flip: 'horizontal' or 'hor', 'vertical' or 'ver', both
                         - greyscale: []
                         - grid_mask: (min_x_pos, max_x_pos, min_y_pos, max_y_pos, min_width_square, max_width_square,
                                        min_height_square, max_heigth_square, min_x_distance_between_squares,
                                        max_x_distance_between_squares, min_y_distance_between_squares, max_y_distance_between_squares)
                                        Values must be between 0 to 1 sinc they are relative to the size of the image.
                                        Generally, the initial position should be similar to the distance between squares
                                        This type of augmentations can be used two o three times with different parameters, since it is
                                        good to have a lot of different grids without having too much of the image covered.
                         - illumination: (min_radius, max_radius, min_magnitude, max_magnitude)  -- standard (0.05, 0.1, 100, 200)
                         - noise: (min_sigma, max_sigma) -- gaussian noise wiht mean 0
                         - occlusion: (type, min_height, max_height, min_width, max_width)  - creates a box of noise to block the image.
                                    The types are hide_and_seek and cutout so far. As extra parameter accepts 'num_patches' which can be a number or a range
                                    By default is 1.
                         - posterisation: (min_number_levels, max_number_levels) Reduce the number of levels of the image. It is assumed a 255
                                        level (at least it is going to be returned in this way). However, this will perform a reduction to less levels than 255
                         - rgb swapping: True or False. This opertion swaps the RGB channels randomly
                         - rotation: (min angle, max angle) - in degrees
                         - shear: (type, magnitude_min, magnitude_max) types are "random", 'hor', 'ver'. The magnitude are the angles to shear in degrees
                         - skew: (type, magnitude_min, magnitude_max), where types are: "TILT", "TILT_LEFT_RIGHT", "TILT_TOP_BOTTOM", "CORNER", "RANDOM", "ALL"
                         - solarise: [] doing a solarisation (max(image) - image)
                         - translate: (min_x, max_x, min_y, max_y) values are relative to the size of the image (0, 0.1, 0, 0.1)
                         - whitening: (min_alpha, max_alpha)  -- new image is  alpha*white_image + (1-alpha) * image
                         - zoom: (min value, max value) - the values are relative to the current size. So, 1 is the real size image (standard 0.9, 1.1)

                         Apart from this, the values could be a dictionary where of the form {'values': [values], 'probability': 1, special_parameter: VALUE}
                         The probability is the ratio of using this operation and special_parameters are indicated in the above descriptions when they have

                         In addition, the same operations can be used in individual objects by usin ooperation_object. For intance, tramslate_object. The only
                         requirement is to pass the bounding boxes in the run operation as
                         bounding_boxes=[[[x00_tp, y00_tp, x00_bl, y00_bl],[x01_tp, y01_tp, x01_bl, y01_bl]],[[x10_tp, y10_tp, x10_bl, y10_bl]]],
                         where xij and yij are the x and y position of image i box j, and tp and bl are top right and bottom left

        :param no_repetition (boolean): This will not allowed operations of the same kind to be performed simultanously.
        :param seed:
        """
        self.perform_checker = True
        self.seed = seed
        self._operations = operations
        self._std_noise = std_noise

        self.no_repetition = no_repetition

        self._skew_types = ["TILT", "TILT_LEFT_RIGHT", "TILT_TOP_BOTTOM", "CORNER", "RANDOM", "ALL"]
        self._flip_types = ['VERTICAL', 'VER', 'HORIZONTAL', 'HOR', 'RANDOM', 'ALL']
        self._occlusion_types = ['hide_and_seek', 'cutout']
        self._illumination_types = ['blob_positive', 'blob_negative', 'blob', 'constant_positive', 'constant_negative',
                                   'constant', 'positive', 'negative', 'all', 'random']
        self._locations = ['no', 'intra', 'inter']

        self.initial_prob = {'flip': 0.5, 'solarise': 0.5, 'greyscale': 0.5, 'rgb_swapping': 0.5}

        # There are methods that are going to be run by PIL and others by numpy functions. In order to speed up the
        # all the methods that require a PIL image are going to be run first followed by those that uses numpy arrays.
        # This internal member variables contains the name of the methods/transformations using numpy arrays
        self._numpy_fun = ['blur', 'change', 'grid_mask', 'illumination', 'noise', 'occlusion', 'posterisation',
                          'rgb_swapping', 'sample_pairing', 'translate', 'zoom']

        # This variable will stored the name of the methods use last time. It can be used to check the transformations
        # that were performed on the augmented images.
        self.last_operations = []

    @property
    def operations(self):
        return self._operations

    @operations.setter
    def operations(self, operations):
        self._operations = operations
        self.perform_checker = True

    def rescale(self, im: np.ndarray, min_int=None, max_int=None) -> np.ndarray:
        """
        Rescale an image between 0 and 255
        """
        if np.max(im) == np.min(im):
            return (im * 0).astype(np.uint8)

        if not min_int:
            min_int = max(np.min(im), 0)
        if not max_int:
            max_int = min(np.max(im), 255) - min_int
        return (max_int * (im.astype(np.float) - np.min(im)) / (np.max(im) - np.min(im)) + min_int).astype(np.uint8)

    def run(self, images: List[np.ndarray], **kwargs):
        self.last_operations = []
        if len(images) == 0:
            return images

        past_operations = []

        bbs = kwargs.get('bounding_boxes', None)

        new_operations = {'numpy': {}, 'pil': {}}
        for operation, values in self.operations.items():
            # Remove digits from operation
            operation2 = ''.join([i for i in operation if not i.isdigit()])
            pos = operation2.find('_object')
            if pos >= 0:
                operation2 = operation2[:operation2.find('_object')]
            if operation2 in self._numpy_fun:
                new_operations['numpy'][operation] = values
            else:
                new_operations['pil'][operation] = values

        get_first_value = False
        if not isinstance(images, (set, tuple, list)):
            get_first_value = True
            images = [images]

        channels = []
        for image in images:
            channel = 0
            if len(image.shape) == 3:
                channel = image.shape[2]
            channels.append(channel)
            if not isinstance(image, np.ndarray):
                raise TypeError('Images must be of type ndarray')

        norm = lambda x: np.squeeze(np.uint8(x)) if np.max(x) > 1.0 else np.uint8(x * 255)
        output = [Image.fromarray(norm(image)) for image in images]

        # If some extra data gets modified, like the labels of the images, then it must be returned. The output would be
        # a dictionary with the images in the key images and the other parameters with the same name as in the input.
        output_extra = {}
        for type_data in ['pil', 'numpy']:
            operations = new_operations[type_data]
            if type_data == 'numpy':
                num_output = []
                for i, image in enumerate(output):
                    # The number of channels must be preserved, operations with PIL like greyscale or greyscale images
                    # would removed the dimension and leave a 2D image
                    channel = channels[i]
                    image = np.array(image)
                    if len(image.shape) == 2 and channel == 1:
                        image = image[..., None]
                    elif (len(image.shape) == 2 and channel == 3) or (
                            len(image.shape) == 3 and image.shape[2] == 1 and channel == 3):
                        image = np.dstack([image, image, image])
                    num_output.append(image)

                output = num_output

            for operation, values in operations.items():
                if not isinstance(values, dict):
                    input_data = {'values': values}
                else:
                    input_data = values

                probability = input_data.get('probability')
                probability = probability if probability else input_data.get('prob',
                                                                             self.initial_prob.get(operation, 1.0))

                extra_data = {key: value for key, value in input_data.items() if
                              key not in ['values', 'probability', 'prob', 'bounding_box']}
                for key, val in kwargs.items():
                    extra_data[key] = val

                if np.random.rand(1)[0] < probability:
                    # Remove digits
                    operation_real = operation
                    operation = ''.join([i for i in operation if not i.isdigit()])
                    wrapper_op = operation
                    if '_object' in operation:
                        extra_data['operation'] = operation[:operation.find('_object')]
                        wrapper_op = 'apply_on_objects'
                    op = getattr(self, wrapper_op, None)
                    if op is not None:
                        if not self.no_repetition or operation not in past_operations:
                            self.last_operations.append(operation_real)
                            past_operations.append(operation)
                            output = op(output, list(input_data.get('values', [])), **extra_data)

                            # If the operation returns a dictionary, means that a parameters from extra_data has been
                            # modified, when this happens, extra_data must be updated for future calls and the final
                            # output should include the modified values.
                            if isinstance(output, dict):
                                copy_dict = output
                                output = copy_dict['images']
                                for key, value in copy_dict.items():
                                    if key == 'bounding_boxes':
                                        kwargs[key] = value
                                    if key != 'images':
                                        output_extra[key] = value
                                        extra_data[key] = value
                    else:
                        print('The operation {} does not exist, Aborting'.format(operation))

        # output = [np.array(image) for image in pil_obj]

        if get_first_value:
            output = output[0]
        if output_extra or bbs:
            if 'bounding_boxes' not in output_extra:
                output_extra['bounding_boxes'] = bbs
            output_extra['images'] = output
            output = output_extra

        return output

    def check_images_equal_size(self, images: List[Union[np.ndarray, Image.Image]]) -> bool:
        """
        Check that all the images have the same size
        :param images: A list of images
        :return: True if all images have same size otherwise False
        """
        get_size = lambda x: x.size
        if isinstance(images[0], np.ndarray):
            get_size = lambda x: x.shape[:2]

        h, w = get_size(images[0])
        for image in images:
            h1, w1 = get_size(image)
            if h1 != h or w1 != w:
                return False

        return True

    def check_groups_images_equal_size(self, images: List[Union[np.ndarray, Image.Image]],
                                       images2: List[Union[np.ndarray, Image.Image]]) -> bool:
        """
        Check that all the images in images and images2 has the same size. It assumes that both images and images2
        have at least one image
        :param images: A list of images (numpy array or PIL)
        :param images2: A list of images (numpy or PIL)
        """
        output = True
        output = output and self.check_images_equal_size(images)
        output = output and self.check_images_equal_size(images2)

        return output and self.check_images_equal_size([np.array(images[0]), np.array(images2[0])])

    def _correct_bb(self, bb: List[int], max_size: List[int]) -> bool:
        """
        Correct the bb so that the values are clipped to 0 and max_size
        :param bb: The bb values [x0, y0, xend, yend]
        :param max_size:  The maximum size of the image (w, h
        :return: Clipped bb
        """
        for i, bb_i in enumerate(bb):
            bb_i = max(bb_i, 0)
            bb_i = min(bb_i, max_size[i % 2])
            bb[i] = int(round(bb_i))

        return bb

    def _is_wrong_bbs(self, bbs: List[List[List[int]]], num_images: int) -> bool:
        """
        Check whether the bounding boxes are correctly passed.
        :param bbs: This should a list of lists of lists for each image a list of bounding
                            boxes and each bounding box should be a list of 4 values [x0, y0, x1, y1]
        :param num_images: The number of images
        :return: True if the bbs is correct otherwise false.
        """
        if not hasattr(bbs, '__len__'):
            return True
        # An empty list is allowed
        if len(bbs) == 0:
            return False
        if not len(bbs) == num_images:
            return True

        for i in range(num_images):
            for bbi in bbs[i]:
                if len(bbi) != 4:
                    return True

        return False

    def apply_on_objects(self, images: List[np.ndarray], values: list, **kwargs) -> List[np.ndarray]:
        """
        Apply a transformation in a set of objects in a set of images
        :param images: A list of numpy arrays, each being an image
        :param values: The values are specific of the operation.
        :param kwargs: Specific for the operation. However, in all object detection, the following values
                                    can appear.
                                    -- operation (str): The name of the operation. This is passed automatically.
                                    -- bounding_boxes (mandatory): A list of lists of lists. A list  for
                                                    each element to be bounded to one image (so len(images) == len(bounding_boxes)
                                                    Then, for each image a list of bounding boxes, one for each object.
                                                    Lastly, each inner list contains 4 values [x0, y0, x1, y1].

                                                    IMPORTANT: The order of the set of bounding boxes per image should
                                                    follow the order of the images. For instance, if the images. For images
                                                    that are masks use an empty list []. This is to keep consistency
                                                    with the rest of operations
                                    --change_location (optional): Options are 'No', 'intra', 'inter', meaning no changes
                                                                    changes within the image and in another image respectively.
                                    -- mask_matching (optional): A dictionary with keys being the position of the real image
                                                                and the values a LIST with the positions of the masks
                                                                associated to it. Notice that it must be a list, it cannot
                                                                be a single value.
                                    -- prob_bb (optional): A value from 0 to 1 representing the probability of applying
                                                            operation to a given object. By default 1.
        :return: Processed images, same ordere as the entered.
        """
        op = kwargs['operation']
        bbs = kwargs.get('bounding_boxes', None)
        prob = kwargs.get('prob_bb', 1.0)
        mask_matching = kwargs.get('mask_matching', {})

        if not isinstance(mask_matching, dict):
            raise ValueError(f"""In {op}_object, Mask positions must be a dictionary with keys as images and values 
            a list of masks positions""")

        mix_images = kwargs.get('mix_images', [])
        mix_labels = kwargs.get('mix_labels', [])
        if op.lower() == 'sample_pairing':
            labels = np.array(kwargs.get('object_label', None))
            if np.any(labels == None):
                raise ValueError('In sample pairing between images, the key label must be passed.')
            if len(images) != len(labels):
                raise ValueError('In sample pairing between images, the number of labels must be the same as images.')
            for i, bbs_i in enumerate(bbs):
                if len(labels[i]) != len(bbs_i):
                    raise ValueError("""In sample pairing between images, the number of labels must be the same as the 
                    number of bounding boxes per image""")

            if not hasattr(values, '__len__') or len(values) != 2:
                raise ValueError('In sample pairing between images, there must be 2 values.')
            if len(mix_images) != len(mix_labels):
                raise ValueError("""In sample pairing between images, the number of labels must be the same as images 
                (mix_images and mix_labels).""")

        all_image_pos = list(mask_matching.keys())
        all_mask_pos = []
        for mask_pos in mask_matching.values():
            if isinstance(mask_pos, list):
                raise ValueError(f"""In {op}_object, Mask positions must be a dictionary with keys as images and 
                values a list of masks positions""")
            all_mask_pos.extend(mask_pos)

        if bbs == None:
            print(f'In {op}_object, bounding boxes not passed. Aborting!')
            return images

        if not hasattr(bbs, '__len__'):
            raise ValueError(f'In {op}_object, the bbs must be a list, numpy array or any element that can be iterated')

        if not len(images) == len(bbs):
            raise ValueError(
                f'In {op}_object, the bbs must be a list with as many elements as images without counting masks')

        if prob < 0 or prob > 1:
            raise ValueError(
                f'In {op}_object, the probability of flipping a bounding box must be greater than 0 and smaller than 1')

        positions = []
        mask_positions = []
        bbs_out = []
        for i in range(len(images)):
            if i in all_image_pos:
                positions.append(i)
                mask_positions.append(mask_matching[i])
                bbs_out.append(bbs[i])
            elif i not in all_mask_pos:
                positions.append(i)
                mask_positions.append([])
                bbs_out.append(bbs[i])

        bbs = bbs_out

        if op == 'change':
            return self.change(images, values, bbs, **kwargs)

        fun = getattr(self, op)
        imresize2 = lambda image, nh, nw: imresize(image, (nh, nw), preserve_range=True, mode='constant',
                                                   anti_aliasing=True)
        for i in positions:
            masks = mask_matching.get(i, [])
            for ii, bb in enumerate(bbs[i]):
                inputs = []
                if len(bb) != 4:
                    raise ValueError(f"""In {op}_object, bounding boxes found with less than 4 elements. 
                    They must be (x0, y0, x1, y1)""")
                if np.random.rand() < prob:
                    if op in self._numpy_fun:
                        inputs.append(images[i][bb[1]:bb[3], bb[0]:bb[2]])
                        if op.lower() == 'sample_pairing':
                            kwargs['labels'] = [labels[i][ii]]
                            p = np.random.randint(0, len(mix_labels))
                            kwargs['mix_images'] = [
                                imresize(mix_images[p], inputs[0].shape[:2], anti_aliasing=True, preserve_range=True)]
                            kwargs['mix_labels'] = [mix_labels[p]]

                        kwargs['mask_positions'] = []
                        for iii, p in enumerate(masks):
                            inputs.append(images[p][bb[1]:bb[3], bb[0]:bb[2]])
                            kwargs['mask_positions'].append(iii + 1)
                        shape = inputs[0].shape[1::-1]
                        im_shape = images[i].shape[:2]
                    else:
                        inputs.append(images[i].crop(bb))
                        kwargs['mask_positions'] = []
                        for iii, p in enumerate(masks):
                            inputs.append(images[p].crop(tuple(bb)))
                            kwargs['mask_positions'].append(iii + 1)

                        shape = inputs[0].size
                        im_shape = images[i].size[-1::-1]

                    # For some operations: zoom and translations, it is necessary to allow the image to
                    # be outside the boundaries, so the a flag is activated and then the b is changed
                    kwargs['object_detection'] = True
                    kwargs['bounding_boxes'] = []
                    inputs = fun(inputs, values, **kwargs)
                    if isinstance(inputs, dict):
                        bb_diff = inputs.get('bb_diff', [0, 0, 0, 0])
                        bb = [bb[i] + bb_diff[i] for i in range(4)]
                        shape = [shape[i] + max(abs(bb_diff[i]), abs(bb_diff[i + 2])) for i in range(2)]
                        inputs = inputs['images']
                        if op in self._numpy_fun:
                            new_shape = inputs[0].shape[1::-1]
                        else:
                            new_shape = inputs[0].size
                        if shape[0] != new_shape[0] or shape[1] != new_shape[1]:
                            inputs[0] = imresize2(inputs[0], shape[1], shape[0])

                        s = [0, 0, shape[0], shape[1]]
                        if bb[0] < s[0]:
                            s[0] = -bb[0]
                            bb[0] = 0
                        if bb[1] < s[1]:
                            s[1] = -bb[1]
                            bb[1] = 0
                        if bb[2] > im_shape[1]:
                            s[2] = shape[0] - (bb[2] - im_shape[1])
                            bb[2] = images[0].shape[1]
                        if bb[3] > im_shape[0]:
                            s[3] = shape[1] - (bb[3] - im_shape[0])
                            bb[3] = im_shape[0]
                    else:
                        s = [0, 0, shape[0], shape[1]]
                        if op in self._numpy_fun:
                            new_shape = inputs[0].shape[1::-1]
                        else:
                            new_shape = inputs[0].size
                        if shape[0] != new_shape[0] or shape[1] != new_shape[1]:
                            inputs[0] = imresize2(inputs[0], shape[1], shape[0])

                    if op in self._numpy_fun:
                        images[i][bb[1]:bb[3], bb[0]:bb[2]] = inputs[0][s[1]:s[3], s[0]:s[2]]
                        for iii, p in enumerate(masks):
                            images[p][bb[1]:bb[3], bb[0]:bb[2]] = inputs[ii][s[1]:s[3], s[0]:s[2]]
                    else:
                        images[i].paste(inputs[0].crop(tuple(s)), tuple(bb))
                        for iii, p in enumerate(masks):
                            images[p].paste(inputs[iii + 1].crop(tuple(s)), tuple(bb))

        return images

    def blur(self, images: List[np.ndarray], values: list, **kwargs) -> List[np.ndarray]:
        """
        Blur an image based on a Gaussian kernel. In future, new kernels should be added.
        :param images(list or array): A list or array of images, each being 3D
        :param values: 3 values. The kernel type: Gaussian, median or random.
                                The minimum and maximum of the variation for the blurriness
                                - Gaussian: It represents the standard deviation. The minimum value is 1e-6 and the
                                maximum is the size of smallest between height and width of the image size
                                - Median: It represents the size of the kernel. It will create a square kernel of 1's.
        :param kwargs: For this operation, we have
                        mask_positions: The positions in images that are masks.
                        filter_parameters (dict): This are extra parameters to pass to the function performing the blurriness
                                            In the case of Gaussian, check the parameters in
                                            https://scikit-image.org/docs/dev/api/skimage.filters.html#skimage.filters.gaussian
                                            The first parameter is the standard deviation that is being randomly selected.
                                            The rest of them can be passed using a dictionary.
        :return: A list with the images with some brightness change
        """

        no_mask_positions = np.ones(len(images)).astype(bool)
        for pos in kwargs.get('mask_positions', []): no_mask_positions[pos] = False
        filter_parameters = kwargs.get('filter_parameters', {})

        if not hasattr(values, '__len__') or len(values) != 3:
            raise ValueError("""In blur, the number of elements must be 3. The type of blurrines and the minimum and 
            maximum value of the parameter to change""")

        kernel_type = values[0]
        values = values[1:]

        output = []
        for i, image in enumerate(images):
            if no_mask_positions[i]:
                kernel = kernel_type.lower()
                if kernel == 'random':
                    kernel = np.random.choice(['gaussian', 'median'])
                if kernel == 'gaussian':
                    factor = int(np.round(checker('blur', 'standard deviation of Gaussian kernel', values, 2, 1e-6,
                                                  min(*image.shape[:2]))))
                    im = gaussian(image, factor, preserve_range=True, multichannel=True, **filter_parameters)
                    output.append(self.rescale(im))
                elif kernel == 'median':
                    factor = int(np.round(
                        checker('blur', ' Size of the median kernel', values, 2, 1, min(*image.shape[:2]) / 4)))
                    selem = np.ones((factor, factor))
                    im = median(image, selem[:, :, None], **filter_parameters)
                    output.append(self.rescale(im))
                else:
                    raise ValueError(
                        f'Kernel {kernel} not found. Currently, only Gaussian, median and random are available')
            else:
                output.append(image)

        return output

    def brightness(self, images: List[Image.Image], values: list, **kwargs) -> List[Image.Image]:
        """
        Change the birghtness of the image, factors smaller than 0 will make the image darker and a factor greater than 1
        will make it brighter. 0 means completely black and it should be avoided
        :param images: A list of PIL images
        :param values: 2 values. The minimum and maximum change in brightness, the values must be between 0.05 and 10
                                beyond those points the result are too dark or too bright respectively.
        :return: A list with the images with some brightness change
        """

        factor = checker('brightness', 'range of brightness', values, 2, 0.05, 10)
        no_mask_positions = np.ones(len(images)).astype(bool)
        for pos in kwargs.get('mask_positions', []): no_mask_positions[pos] = False

        output = []
        for i, image in enumerate(images):
            if no_mask_positions[i]:
                image_enhancers = ImageEnhance.Brightness(image)
                output.append(image_enhancers.enhance(factor))
            else:
                output.append(image)

        return output

    def change(self, images: List[np.ndarray], values: list, bbs: list, **kwargs) -> List[np.ndarray]:
        """
        Copy the transform object into a random location of the image or of a new image
        :param image: List of images
        :param values: The input of the transformation. There are 3 values
                                    - String with the type of transformation ('No', 'Inter', Intra', where Inter means
                                        a different image is used and intra the same image.
                                    - Boolean to use the same location when copying or a random location
                                    - Dictionary with the possible transformations for the object. Use equal for no
                                        transformation.
        :param bbs: The size of the object in the part
        :param kwargs: Extra parameters.
                         - mix_images (mandatory?): It is only mandatory when the type_change is 'Inter'
                         - number_copies (optional): 1 by default. This is the number of copies of the object in a new
                                                        or the same image. It is only useful when the random_location
                                                        is True

        :return: The list of images with the transform objects
        """

        if not hasattr(values, '__len__') or len(values) != 3:
            raise ValueError('In change_object, The dimension of values must be ')

        type_change = values[0]
        random_location = values[1]

        if not isinstance(type_change, str):
            raise TypeError(
                ('The first input of values must be a string with values' + '{}, ' * len(self._locations)).format(
                    *self._locations)[:-2])
        if type_change.lower() not in self._locations:
            raise ValueError(
                (f'In change_object, the value {type_change} does not exists it should be one of ' + '{}, ' * len(
                    self._locations)).format(*self._locations))[:-2]
        if type_change.lower() == 'no':
            return images
        if not isinstance(random_location, bool):
            raise TypeError('The second input of values must be a boolean')
        if not isinstance(values[2], dict):
            raise TypeError('The third input of values must be a dictionary for a new augmentor only for boxes')

        d = {}
        for key, value in values[2].items():
            if '_object' in key:
                print('Operations with _object are not allowed for change_object. This operation will not be used')
            else:
                d[key] = value

        mix_images = kwargs.get('mix_images', [])
        if not isinstance(mix_images, list) or (type_change == 'inter' and len(mix_images) == 0):
            raise ValueError(
                'In change_objects, the mix_image parameter is mandatory and it must be a list with images')

        number_copies = kwargs.get('number_copies', 1)
        if not isinstance(mix_images, (int, list, tuple)) or len(number_copies) != 2:
            raise ValueError("""In change_objects, the number_copies parameter must be an integer, 
            list or tuple with 2 values at most""")

        if not random_location:
            number_copies = 1

        if hasattr(number_copies, '__len__'):
            if min(number_copies) < 0:
                raise ValueError('In change_objects, the number_copies parameter must be larger or equal than 0')

            number_copies = np.random.randint(number_copies[0], number_copies[1] + 1, 1)[0]

        if number_copies < 0:
            raise ValueError('In change_objects, the number_copies parameter must be larger or equal than 0')

        obj_aug = Augmentor(d)

        num_images = len(mix_images)
        bbs2 = []
        for i, image in enumerate(images):
            bbs2.append([])
            if type_change.lower() == 'inter':
                pos = np.random.randint(0, num_images)
                # Create copy of images so that the background does not get modified
                mix_image = mix_images[pos].copy()
            else:
                mix_image = image.copy()

            h, w = mix_image.shape[:2]

            for j, bb in enumerate(bbs[i]):
                if len(bb) != 4:
                    raise ValueError(f"""In change_object, bounding boxes found with less than 4 elements. 
                    They must be (x0, y0, x1, y1)""")

                for _ in range(number_copies):
                    inputs = [image[bb[1]:bb[3], bb[0]:bb[2]]]
                    kwargs['object_detection'] = False
                    kwargs['bounding_boxes'] = []
                    inputs = obj_aug.run(inputs, **kwargs)
                    part = inputs[0]

                    if part.shape[0] > h or part.shape[1] > w:
                        hp, wp = part.shape[:2]
                        scale = min(wp / w, hp / h)
                        nw, nh = int(scale * w), int(scale * h)
                        part = imresize(image, (nh, nw), preserve_range=True, mode='constant', anti_aliasing=True)

                    hp, wp = part.shape[:2]
                    if random_location:
                        start = [np.random.randint(0, w - wp, 1)[0], np.random.randint(0, h - hp, 1)[0]]
                        bb2 = start + [si + se for si, se in zip(start, [wp, hp])]
                    else:
                        bb2 = list(bb)
                        if bb[3] - bb[1] != hp or bb[2] - bb[0] != wp:
                            bb2[3] = bb[1] + hp
                            bb2[2] = bb[0] + wp
                            if bb2[3] > mix_image.shape[0]:
                                extra = mix_image.shape[0] - bb2[3]
                                bb2[3] += extra
                                bb2[1] += extra
                            if bb2[2] > mix_image.shape[1]:
                                extra = mix_image.shape[1] - bb2[2]
                                bb2[2] += extra
                                bb2[0] += extra
                        # part = imresize(part, (bb[3] - bb[1], bb[2] - bb[0]), anti_aliasing=True, preserve_range=True)

                    bbs2[i].append(bb2)
                    mix_image[bb2[1]:bb2[3], bb2[0]:bb2[2]] = part

                images[i] = mix_image

        return {'images': images, 'bounding_boxes': bbs2}

    def colour_balance(self, images: List[Image.Image], values: list, **kwargs) -> List[Image.Image]:
        """
        Change the saturation of the image, factors smaller than 1 will make the image darker and a factor greater than 1
        will make it brighter. 0 means completely black and it should be avoided
        :param images: A list of numpy arrays, each being an image
        :param values: A list with 2 values. Minimum value for colour balance (greater than 0)
                                            Maximum value for colour balance (smaller than 10)
        :param kwargs: For this operation, the only extra parameter is the whether an image is a mask.
                         mask_positions: The positions in images that are masks.
        :return:
        """
        factor = checker('brightness', 'range of colour_balance', values, 2, 0, 10)
        no_mask_positions = np.ones(len(images)).astype(bool)
        for pos in kwargs.get('mask_positions', []): no_mask_positions[pos] = False

        output = []
        for i, image in enumerate(images):
            if no_mask_positions[i]:
                image_enhancers = ImageEnhance.Color(image)
                output.append(image_enhancers.enhance(factor))
            else:
                output.append(image)

        return output

    def contrast(self, images: List[Image.Image], values: list, **kwargs) -> List[Image.Image]:
        """
        Change the contrast of the image, factors smaller than 1 will make the image to have a solid colour and a factor greater than 1
        will make it brighter.
        :param images: A list of numpy arrays, each being an image
        :param values: A list with 2 values. Minimum value for contrast balance (greater than 0)
                                           Maximum value for contrast balance (smaller than 10)
        :param kwargs: For this operation, the only extra parameter is the whether an image is a mask.
                        mask_positions: The positions in images that are masks.
        :return:
        """

        factor = checker('contrast', 'range of contrast', values, 2, 0, 10)
        no_mask_positions = np.ones(len(images)).astype(bool)
        for pos in kwargs.get('mask_positions', []): no_mask_positions[pos] = False

        output = []
        for i, image in enumerate(images):
            if no_mask_positions[i]:
                image_enhancers = ImageEnhance.Contrast(image)
                output.append(image_enhancers.enhance(factor))
            else:
                output.append(image)

        return output

    def crop(self, images: List[np.ndarray], values: list, **kwargs) -> List[np.ndarray]:
        """
        Perform a crop of the images. The main difference with zoom is that zoom creates a crop from the center of the
        image and respects the dimension of the images, in addition it may increase the size of the image.
        This operation selects a random position in the image and extract a patch with a random height and width. Both
        the height and the width can be restricted to a given range.
        :param images: A list of numpy arrays, each being an image
        :param values: 4 values: Minimum height of the crop (or both height and width if there are only two values)
                                Maximum height of the crop (or both height and width if there are only 2 values).
                                Minimum width of the crop (optional)
                                Maximum width of the crop (optional)

                                All the values must be between 0 and 1, meaning a relatie crop with respect to
                                the size of the image.
        :param kwargs: For this operation, the only extra parameter is the whether an image is a mask.
                         - mask_positions: The positions in images that are masks.
                         - bounding_boxes (list): A list of lists of lists. A list  for
                                                    each element to be bounded to one image
                                                    (so len(images) == len(bounding_boxes)
                                                    Then, for each image a list of bounding boxes, one for each object.
                                                    Lastly, each inner list contains 4 values [x0, y0, x1, y1].

                                                    IMPORTANT: The number of elements in the outer list is the number
                                                    of images. This is different to where the elements are with the
                                                    object part. In that case the outer lit does not include masks
                        - keep_size (optional): Boolean Keep the same size as the cropping withot resizing it to the
                        						original size
        :return: The cropped images
        """

        no_mask_positions = np.ones(len(images)).astype(bool)
        for pos in kwargs.get('mask_positions', []): no_mask_positions[pos] = False

        keep_size = kwargs.get('keep_size', False)

        output = []
        name = 'crop'
        if not self.check_images_equal_size(images):
            print('For {}, the size of the images must be the same. Aborting'.format(name))
            return images

        shape = images[0].size
        bb = [shape[0], shape[1], 0, 0]
        # Get the smallest box covering all bounding boxes
        bbs = kwargs.get('bounding_boxes', [])
        if self._is_wrong_bbs(bbs, len(images)):
            raise ValueError('In crop, The bbs were not correctly passed.')

        for bbs_i in bbs:
            for bbi in bbs_i:
                bb[0] = min(bbi[0], bb[0])
                bb[1] = min(bbi[1], bb[1])
                bb[2] = max(bbi[2], bb[2])
                bb[3] = max(bbi[3], bb[3])

        name_params = ['height', 'width']
        for i in range(len(values) // 2):
            check_range(name, name_params[i], values[i * 2:(i + 1) * 2], 0, 1)

        if len(values) != 2 and len(values) != 4:
            raise ValueError('The length of values in crop must be 2 or 4')

        if len(values) == 4:
            cropped_height = np.random.uniform(values[0], values[1])
            cropped_width = np.random.uniform(values[2], values[3])
        else:
            if values[1] > 1:
                raise ValueError('When only two elements are use, the values of the crop must be relative 0 to 1')

            cropped_height = np.random.uniform(values[0], values[1])
            cropped_width = cropped_height

        cropped_height = cropped_height if cropped_height >= 1 else int(cropped_height * shape[0])
        cropped_width = cropped_width if cropped_width >= 1 else int(cropped_width * shape[1])

        center_w = \
            np.random.randint(int(np.ceil(cropped_width / 2.0)), int(np.ceil(shape[1] - cropped_width / 2.0)), 1)[0]
        center_h = \
            np.random.randint(int(np.ceil(cropped_height / 2.0)), int(np.ceil(shape[0] - cropped_height / 2.0)), 1)[0]

        width = int(np.ceil(cropped_width / 2.0))
        height = int(np.ceil(cropped_height / 2.0))

        crop = [center_h - height, center_w - width, center_h + height, center_w + width]
        for i, crop_i in enumerate(crop):
            if i < 2:
                if crop_i > bb[i]:
                    crop[i] = bb[i]
                for ii, bbs_i in enumerate(bbs):
                    for iii, bbi in enumerate(bbs_i):
                        bbs[ii][iii][i] -= crop[i]
                        bbs[ii][iii][2 + i] -= crop[i]
            else:
                if crop_i < bb[i]:
                    crop[i] = bb[i]

        for i, image in enumerate(images):
            # if no_mask_positions[i]:
            image = image.crop(crop)
            if not keep_size:
                image = image.resize((shape[0], shape[1]))
            if bbs:
                w, h = [shape[0], shape[1]]
                center = [w // 2, h // 2]
                factor = w / (crop[2] - crop[0]), h / (crop[3] - crop[1])
                for ii, bbi in enumerate(bbs[i]):
                    bbs[i][ii] = self._correct_bb([bbi[iii] * factor[iii % 2] for iii in range(len(bbi))], [w, h])
            # image = Image.fromarray(self.rescale(image))
            output.append(image)

        if bbs:
            output = {'images': output, 'bounding_boxes': bbs}

        return output

    def equal(self, images: List[np.ndarray], values: list, **kwargs) -> List[np.ndarray]:
        """
        Leave the images as they are. This operations is used in the case other operations are pile up. For instance,
        using equal_object with the option of change_location will make the original objects to be place in a different
        part of the image or in another image.
        :return: The original images
        """
        return images

    def flip(self, images: List[np.ndarray], values: list, **kwargs) -> List[np.ndarray]:
        """
        Flip the image, vertically, horizontally or both
        :param images: A list of numpy arrays, each being an image
        :param values: 1 value, the type ('horizontal', 'vertical', 'all', 'random')
        :param kwargs: bounding_boxes (list): A list of lists of lists. A list  for
                                                each element to be bounded to one image
                                                (so len(images) == len(bounding_boxes)
                                                Then, for each image a list of bounding boxes, one for each object.
                                                Lastly, each inner list contains 4 values [x0, y0, x1, y1].

                                                IMPORTANT: The number of elements in the outer list is the number
                                                of images. This is different to where the elements are with the
                                                object part. In that case the outer lit does not include masks
        :return: The flipped images
        """
        if isinstance(values, (tuple, list)):
            values = values[0]

        bbs = kwargs.get('bounding_boxes', [])
        if self._is_wrong_bbs(bbs, len(images)):
            raise ValueError('In flip, The bbs were not correctly passed.')

        if values.upper() not in self._flip_types:
            raise ValueError('''The name {} does not exist for the flip operation. 
            Possible values are: {}'''.format(values, self._flip_types))
        if values.lower() == 'random':
            values = np.random.choice(['horizontal', 'vertical', 'all'], 1)[0]

        bbs_out = []
        if values.lower() == 'horizontal' or values.lower() == 'hor' or values.lower() == 'both' or \
                values.lower() == 'all':
            output = [image.transpose(Image.FLIP_LEFT_RIGHT) for image in images]
            if bbs:
                for i, image in enumerate(images):
                    bbs_out.append([[image.size[0] - bbi[2], bbi[1], image.size[0] - bbi[0], bbi[3]] for ii, bbi in
                                    enumerate(bbs[i])])
                bbs = bbs_out

        bbs_out = []
        if values.lower() == 'vertical' or values.lower() == 'ver' or values.lower() == 'both' or \
                values.lower() == 'all':
            output = [image.transpose(Image.FLIP_TOP_BOTTOM) for image in images]
            if bbs:
                for i, image in enumerate(images):
                    bbs_out.append([[bbi[0], image.size[1] - bbi[3], bbi[2], image.size[1] - bbi[1]] for ii, bbi in
                                    enumerate(bbs[i])])
                bbs = bbs_out

        if bbs:
            output = {'images': output, 'bounding_boxes': bbs}

        return output

    def greyscale(self, images: List[Image.Image], values: list, **kwargs) -> List[Image.Image]:
        """
        Convert to greyscale with probability one
        :param images: A list of numpy arrays, each being an image
        :param values: None
        :param kwargs: For this operation, the only extra parameter is the whether an image is a mask.
                         mask_positions: The positions in images that are masks.
        :return: A list with the images converted into greyscale
        """
        no_mask_positions = np.ones(len(images)).astype(bool)
        for pos in kwargs.get('mask_positions', []):
            no_mask_positions[pos] = False

        output = []
        for i, image in enumerate(images):
            if no_mask_positions[i]:
                output.append(ImageOps.grayscale(image))
            else:
                output.append(image)

        return output

    def grid_mask(self, images: List[np.ndarray], values: list, **kwargs) -> List[np.ndarray]:
        """
        Add a grid mask to the images following https://arxiv.org/pdf/2001.04086.pdf
         :param images: A list of numpy arrays, each being an image
         :param values: 8 values: Minimum and maximum value for the initial x position (top left corner of the top
                                    left square)
                                    Minimum and maximum value for the initial y position (top left corner of the top
                                    left square)
                                    Minimum and maximum value (range) for the width of the square
                                    Minimum and maximum value (range) for the height of the square
                                    Minimum and maximum value (range) for the x distance between square
                                    Minimum and maximum value (range) for the y distance between square

                                    All the values must be between 0 and 1 since they are relative to the image size.
         :param kwargs: For this operation, the only extra parameter is the whether an image is a mask.
                         mask_positions: The positions in images that are masks.
                        use_colour: The colour to use. If the colour is not passed or it is a negative value or greater
                        than 255, gaussian noise will be used instead.
        :return: List of images with occlusions by a grid of masks
        """
        use_colour = kwargs.get('use_colour', -1)
        no_mask_positions = np.ones(len(images)).astype(bool)
        for pos in kwargs.get('mask_positions', []): no_mask_positions[pos] = False

        if not hasattr(values, '__len__') or len(values) != 12:
            raise ValueError(
                'The number of values for the grid_mask operation must be a list or tuple with 12 values. The range of the initial point, square size and distance between square in x and y for the three of them')

        if not self.check_images_equal_size(images):
            print('For grid masks, the size of the images must be the same. Aborting')
            return images

        h, w = images[0].shape[:2]
        params = []
        name = 'grid_mask'
        name_params = ['initial x position', 'initial y position', 'width square', 'height square',
                       'x distance between squares', 'y distance between squares']
        for i in range(len(values) // 2):
            param = checker(name, name_params[i], values[i * 2:(i + 1) * 2], 2, 0, 1)
            if i % 2 == 0:
                param = int(np.ceil(param * w))
            else:
                param = int(np.ceil(param * h))
            params.append(param)

        images_to_use = []
        for ii in range(len(images)):
            if no_mask_positions[ii]:
                if use_colour < 0 or use_colour > 255:
                    im = self._std_noise * np.random.randn(*(images[ii].shape)) + 127.5
                    im[im < 0] = 0
                    im[im > 255] = 255
                else:
                    im = use_colour * np.ones(tuple(images[ii].shape))
            else:
                im = np.zeros(tuple(images[ii].shape))
            images_to_use.append(im)

        return create_grid_masks(images, params[:2], params[2:4], params[4:], images_to_use, no_mask_positions.tolist())

    def illumination(self, images: List[np.ndarray], values: list, **kwargs) -> List[np.ndarray]:
        """
         Add illumination circles to the image following paper: https://arxiv.org/pdf/1910.08470.pdf
         :param images: A list of numpy arrays, each being an image
         :param values: 4 values: Minimum and maximum radius (float). The values must be between 0 and 1
                                 Minimum and maximum intensity to add (int). These values cannot be larger than 255 and lower than 0.
         :param kwargs: For this operation, the only extra parameter is the whether an image is a mask.
                         mask_positions: The positions in images that are masks.
         :return:
        """
        name = 'illumination'  # inspect.currentframe().f_code.co_name
        if not self.check_images_equal_size(images):
            print('For {}, the size of the images must be the same. Aborting'.format(name))
            return images

        param_values = [('radius', 1), ('intensity', 255)]
        if not hasattr(values, '__len__') or len(values) != 5:
            raise ValueError(
                'The number of values for the illumination operation must be a list or tuple with 5 values'.format(
                    name))

        if values[0].lower() not in self._illumination_types:
            raise ValueError(
                'The name {} does not exist for the flip operation. Possible values are: {}'.format(values[0],
                                                                                                    self._illumination_types))

        no_mask_positions = np.ones(len(images)).astype(bool)
        for pos in kwargs.get('mask_positions', []): no_mask_positions[pos] = False

        shape = images[0].shape
        for i in range(len(values) // 2):
            check_range(name, param_values[i][0], values[i * 2 + 1:(i + 1) * 2 + 1], 0, param_values[i][1])

        aux = convert_to_absolute(values[1:3], shape)
        values[1] = aux[0]
        values[2] = aux[1]

        type_illumination = values[0].lower()
        if type_illumination == 'random':
            type_illumination = np.random.choice(self._illumination_types[:-1], 1)

        blob = np.zeros(shape[:2])
        if 'constant' not in type_illumination:
            radius = np.random.uniform(values[1], values[2])
            intensity = np.random.uniform(values[3], values[4])

            yc = np.random.randint(0, shape[0], 1)
            xc = np.random.randint(0, shape[1], 1)

            _, blob = create_circular_mask(shape[0], shape[1], (xc, yc), radius, get_distance_map=True)
            min_val = np.min(blob[blob > 0])
            blob = (blob - min_val) / (1 - min_val)
            blob[blob < 0] = 0

            if 'positive' in type_illumination:
                sign = 1
            elif 'negative' in type_illumination:
                sign = -1
            else:
                sign = int(np.random.rand(1) < 0.5)
            blob = sign * (intensity * blob)

        if 'blob' not in type_illumination:
            intensity = np.random.uniform(values[3], values[4])
            if 'positive' in type_illumination:
                sign = 1
            elif 'negative' in type_illumination:
                sign = -1
            else:
                sign = int(np.random.rand(1) < 0.5)
            blob += sign * intensity

        if len(shape) == 3:
            blob = blob[:, :, np.newaxis]

        output = []
        for i, image in enumerate(images):
            if no_mask_positions[i]:
                image = image.astype(np.float)
                image += blob
                image = self.rescale(image)
            output.append(image)

        return output

    def noise(self, images: List[np.ndarray], values: list, **kwargs) -> List[np.ndarray]:
        """
        Add noise to the images
        :param images: A list of numpy arrays, each being an image
        :param values: 2 values:
                        int: Minimum number for the std of the noise. Values are between 0 and 255. Recommendation: not higher than 30
                        int: Maximum number for the std of the noise. Values are between 0 and 255. Recommendation: not higher than 30
                        Selection is done by a uniform distribution between minimum and maximum values.
        :param kwargs: For this operation, the only extra parameter is the whether an image is a mask.
                        mask_positions: The positions in images that are masks.
        :return:
        """
        std = checker('noise', 'standard deviation of the noise', values, 2, 0, 100)

        no_mask_positions = np.ones(len(images)).astype(bool)
        for pos in kwargs.get('mask_positions', []): no_mask_positions[pos] = False

        use_gray_noise = kwargs.get('use_gray_noise', False)

        output = []
        for i, image in enumerate(images):
            if no_mask_positions[i]:
                gauss = std * np.clip(np.random.randn(*image.shape), -3, 3)
                if len(image.shape) == 3 and use_gray_noise:
                    gauss = np.tile(gauss[:, :, 0][:, :, np.newaxis], [1, 1, 3])
                noisy = image + gauss
                output.append(self.rescale(noisy))  # Image.fromarray(self.rescale(noisy)))
            else:
                output.append(image)

        return output

    def occlusion(self, images: List[np.ndarray], values: list, **kwargs) -> List[np.ndarray]:
        """
        Perform hide and seek and cutout occlusions on some images
        :param images: A list of numpy arrays, each being an image
        :param values: 5 values:
                        str: type of occlusion: at this moment - 'hide_and_seek', 'cutout'
                        int: Minimum number of boxes columns directions
                        int: Maximum number of boxes columns directions
                        int: Minimum number of boxes rows directions
                        int: Maximum number of boxes rows directions
                        Selection is done by a uniform distribution between minimum and maximum values.
        :param kwargs: For this operation, there are two extra parameters:
                        mask_positions: The positions in images that are masks.
                        use_colour: The colour to use. If the colour is not passed or it is a negative value or greater
                        than 255, gaussian noise will be used instead.
        :return: List of images with occlusion
        """
        use_colour = kwargs.get('use_colour', -1)
        if not self.check_images_equal_size(images):
            print('For occlusions, the size of the images must be the same. Aborting')
            return images

        if not hasattr(values, '__len__') or len(values) != 5:
            raise ValueError('The number of values for the occlusion operation must be a list or tuple with 5 values')
        if values[0] not in self._occlusion_types:
            raise ValueError(
                'The name {} does not exist for the skew operation. Possible values are: {}'.format(values[0],
                                                                                                    self._skew_types))

        no_mask_positions = np.ones(len(images)).astype(bool)
        for pos in kwargs.get('mask_positions', []): no_mask_positions[pos] = False

        swapped_images = []
        for ii in range(len(images)):
            if no_mask_positions[ii]:
                if use_colour < 0 or use_colour > 255:
                    im = self._std_noise * np.random.randn(*(images[ii].shape)) + 127.5
                    im[im < 0] = 0
                    im[im > 255] = 255
                else:
                    im = use_colour * np.ones(tuple(images[ii].shape))
            else:
                im = np.zeros(tuple(images[ii].shape))
            swapped_images.append(im)

        new_images = swap_patches(images, values, 'occlusion', swapped_images, **kwargs)

        return [self.rescale(image) for image in new_images]

    def posterisation(self, images: List[np.ndarray], values: list, **kwargs) -> List[np.ndarray]:
        """
        Reduce the number of levels of the image. It is assumed a 255 level (at least it is going to be returned in this way).
        However, this will perform a reduction to less levels than 255
        :param images: A list of numpy arrays, each being an image
        :param values: Two values, representing the minimum value and maximum value of levels to apply the posterisation.
                        An uniform distribution will be used to select the value
        :param kwargs: For this operation, the only extra parameter is the whether an image is a mask.
                        mask_positions: The positions in images that are masks.
        :return:
        """

        if not hasattr(values, '__len__') or len(values) != 2:
            raise ValueError(
                'The number of values for the posterisation operation must be a list or tuple with 2 values')

        levels = checker('Posterisation', 'levels', values, 2, 1, 256)
        no_mask_positions = np.ones(len(images)).astype(bool)
        for pos in kwargs.get('mask_positions', []): no_mask_positions[pos] = False

        # the idea is to reduce the number of levels in the image, so if we need to get 128 levels, means that we need
        # to get the pixels to be between 0 and 128 and then multiply them by 2. So we need to first divide them between
        # 256 /128 = 2
        levels = 256. / levels

        outputs = []
        for i, image in enumerate(images):
            if no_mask_positions[i]:
                min_int = np.min(image)
                max_int = np.max(image)
                image = np.round(self.rescale(image) / levels).astype(np.uint8)
                image = self.rescale(image, min_int, max_int)

            outputs.append(image)

        return outputs

    def rgb_swapping(self, images: List[np.ndarray], values: list, **kwargs) -> List[np.ndarray]:
        """
        Swap to rgb components in the images
        :param images: A list of numpy arrays, each being an image
        :param values: Not used
        :param kwargs: mask_positions, in case one or more of images are masks, then this transformation is not applied
        :return: The same as images
        """
        no_mask_positions = np.ones(len(images)).astype(bool)
        for pos in kwargs.get('mask_positions', []): no_mask_positions[pos] = False

        outputs = []
        for i, image in enumerate(images):
            if no_mask_positions[i]:
                c = np.random.choice(np.arange(3), 2, replace=False)
                image = np.copy(image)
                image[:, :, c[0]] = image[:, :, c[1]]
                image[:, :, c[1]] = image[:, :, c[0]]

            outputs.append(image)

        return outputs

    def rotate(self, images: List[Image.Image], values: list, **kwargs) -> List[Image.Image]:
        """
        Rotate an image by selecting an angle from a range
        :param images: A list of PIL arrays, each being an image
        :param values: 2 values the minimum and maximum angle. Values between -360 and 360
        :param kwargs: bounding_boxes (list): A list of lists of lists. A list  for
                                                each element to be bounded to one image (so len(images) == len(bounding_boxes)
                                                Then, for each image a list of bounding boxes, one for each object.
                                                Lastly, each inner list contains 4 values [x0, y0, x1, y1].

                                                IMPORTANT: The number of elements in the outer list is the number
                                                of images. This is different to where the elements are with the
                                                object part. In that case the outer lit does not include masks
        :return: The same as images
        """
        if not hasattr(values, '__len__') or len(values) != 2:
            raise ValueError('The number of values for the rotate operation must be a list or tuple with 2 values')
        if min(values) < -360 or max(values) > 360:
            raise ValueError("The range of the angles must be between {} and {}.".format(-360, 360))

        bbs = kwargs.get('bounding_boxes', [])
        if self._is_wrong_bbs(bbs, len(images)):
            raise ValueError('In rotate, The bbs were not correctly passed.')

        angle = checker('Rotate', 'range of rotation', values, 2, -360, 360)
        use_colour = kwargs.get('use_colour', None)
        use_replication = kwargs.get('use_replication', False)
        no_mask_positions = np.ones(len(images)).astype(bool)
        for pos in kwargs.get('mask_positions', []): no_mask_positions[pos] = False

        angle_pi = angle * np.pi / 180
        rot_mat = np.array([[np.cos(angle_pi), -np.sin(angle_pi)], [np.sin(angle_pi), np.cos(angle_pi)]])

        output = []
        for i, image in enumerate(images):
            # Get size before we rotate
            w = image.size[0]
            h = image.size[1]

            if no_mask_positions[i]:
                colour = use_colour
                if isinstance(use_colour, (int, float)) and (image.mode == 'RGB' or image.mode == 'RGBA'):
                    colour = (use_colour, use_colour, use_colour)
                if isinstance(use_colour, (list, tuple)) and not (image.mode == 'RGB' or image.mode == 'RGBA'):
                    colour = use_colour[0]
                if isinstance(use_colour, list):
                    colour = tuple(use_colour)
            else:
                colour = (0, 0, 0)

            # Rotate, while expanding the canvas size
            image = image.rotate(angle, expand=True, resample=Image.BICUBIC, fillcolor=colour)

            if bbs:
                # Rotate the bounding box, it requires the four points and then take the minimum and
                # the maximum points as the new bounding box. Notice that this will always result in
                # a bounding box larger or equal to the desired one. The maximum difference will be
                # for an object touching the middle size of each edge of the bounding box and rotating 45 degrees.
                # This would imply a sqrt(2) increase with respect to the correct bounding box.
                w1, h1 = image.size
                factor = w / w1, h / h1
                center = np.array([w // 2, h // 2])
                rotate_point = lambda point: np.round(np.dot(rot_mat.transpose(), (point - center)) + center).astype(
                    int)

                # Get the transformation of the image corners. This is for debugging
                min_val = np.zeros(2)
                im_corners = [[0, 0], [w, 0], [0, h], [w, h]]
                rotate_corners = []
                for p in im_corners:
                    rotate_corners.append(rotate_point(np.array(p)))
                    min_val = np.min(np.vstack([rotate_corners[-1], min_val]), axis=0)

                for ii, bbi in enumerate(bbs[i]):
                    ys = [bbi[:2], bbi[2:], bbi[0::3], [bbi[2], bbi[1]]]
                    xs = [rotate_point(np.array(ys_i)) for ys_i in ys]
                    get = lambda xs, pos: [x[pos] for x in xs]

                    min_x, min_y = min_val.tolist()
                    for iii in range(len(xs)):
                        xs[iii][0] -= min_x
                        xs[iii][1] -= min_y

                    bbs[i][ii] = [min(get(xs, 0)), min(get(xs, 1)), max(get(xs, 0)), max(get(xs, 1))]
                    bbs[i][ii] = self._correct_bb([bbs[i][ii][iii] * factor[iii % 2] for iii in range(len(bbi))],
                                                  [w, h])

            # Return the image, re-sized to the size of the image passed originally
            output.append(image.resize((w, h), resample=Image.BICUBIC))

        if bbs:
            output = {'images': output, 'bbs': bbs}

        return output

    def sample_pairing(self, images: List[np.ndarray], values: list, **kwargs) -> List[np.ndarray]:
        """
        This augmentation performs a weighted average of the image being augmented an another one that must be included
        in the values. The paper recommends to use this augmentation in three steps for better convergence:
        1. SamplePairing is completely disabled for the first epochs.
        2. SamplePairing is enabled for a few epochs and then disabled again. This alternating process should be used
            for the majority of the epochs.
        3. At the end of training SamplePairing is completely disabled again. This is named fine-tuning in the paper.

        :param images: A list of numpy arrays, each being an image
        :param values:  A list or tuple with 2 values. The minimum and maximum weights
        :param kwargs: It will check whether a mask exists, if it does the process will not continue
                        - mask_positions: The positions in images that are masks.
                        It requires the labels of the images as integers, not one hot encoding. This is to avoid masks
                        tp be passed, only vectors will be allowed.
                        - labels (mandatory) (list of lists): Labels of the images as 1 hot encoding.
                        - mix_images (mandatory): Images to mix with the input images
                        - mix_labels (mandatory): Labels of the images to mix with other images
        :return: A list of numpy arrays with t

        """
        labels = np.array(kwargs.get('labels', None))
        if labels is not None and len(labels) == 0:
            raise ValueError('''For the operation sample_pairing the labels of the images must be passed 
            in the run function with the key labels''')
        if not hasattr(labels[0], '__len__'):
            raise ValueError('''In the operation sample_pairing the labels must be a 2D array or list of lists. 
            Only one hot encoding vectors''')
        if not hasattr(values, '__len__') or len(values) != 2:
            raise ValueError('''The number of values for the sample_pairing operation must be a list or tuple with 
            4 values. The minimum and maximum weights, a list with images to mix and a their respective labels''')

        if kwargs.get('mask_positions', None):
            print('The operation sample_pairing does not allow masks to be passed. Aborting')
            return images

        mix_images = kwargs.get('mix_images', [])
        mix_labels = kwargs.get('mix_labels', [])

        if hasattr(mix_images, '__len__') and hasattr(mix_labels, '__len__') and len(mix_images) != len(mix_labels):
            raise ValueError('''In the operation sample_pairing, the number of images and labels used for 
            mixing must be the same, since they correspondent''')
        if len(mix_labels) == 0:
            raise ValueError('''In the operation sample_pairing, at least one image and one labels must be passed to
             mix with the original images''')
        if not hasattr(mix_labels[0], '__len__'):
            raise ValueError('''In the operation sample_pairing the labels for mixing must be a 2D array or list of 
            lists. Only one hot encoding vectors''')

        output = {'images': [], 'labels': []}
        image_labels = list(zip(mix_images, mix_labels))
        num_images = len(image_labels)

        name = 'sample_pairing'
        if not self.check_groups_images_equal_size(images, mix_images):
            print('For {}, the size of all the images, including the ones for mixing must be the same. Aborting'.format(
                name))
            return images

        weight = checker(name, 'weights for averaging', values[:2], 2, 0, 1)

        for image, label in zip(images, labels):
            pos = np.random.randint(0, num_images)
            image_mixing, label_mixing = image_labels[pos]
            new_image = self.rescale(weight * image + (1 - weight) * image_mixing)
            output['images'].append(new_image)
            output['labels'].append(weight * label + (1 - weight) * np.array(label_mixing))

        return output

    def sharpness(self, images: List[Image.Image], values: list, **kwargs) -> List[Image.Image]:
        """
        Blurred or sharp an image
        :param images: A list of numpy arrays, each being an image
        :param values: 2 values: minimum value for the sharpness. It cannot be smaller than -5
                                 maximum value for the sharpness. It cannot be greater than 5.

                                The standard sharpness value is between 0 and 2, whereas 0 means blurred images,
                                1 means original image and 2 sharp image. However, negative values can be used to get
                                very blurry images and values greater than 2. The restrictions are -5 to 5 since
                                beyond those boundaries the fourier coefficients fail. It is recommended to use values
                                from -1 to 3.
        :param kwargs: For this operation, the only extra parameter is the whether an image is a mask.
                         mask_positions: The positions in images that are masks.
        :return: A list of image with changed contrast
        """

        factor = checker('sharpness', 'range of sharpness', values, 2, -5, 5)
        no_mask_positions = np.ones(len(images)).astype(bool)
        for pos in kwargs.get('mask_positions', []): no_mask_positions[pos] = False

        output = []
        for i, image in enumerate(images):
            if no_mask_positions[i]:
                image_enhancers = ImageEnhance.Sharpness(image)
                output.append(image_enhancers.enhance(factor))
            else:
                output.append(image)

        return output

    def shear(self, images: List[Image.Image], values: list, **kwargs) -> List[Image.Image]:
        """
        Shear transformation of an image from https://github.com/mdbloice/Augmentor/blob/master/Augmentor/Operations.py
        :param images: A list of numpy arrays, each being an image
        :param values: 3 values: type (random, both, all, horizontal, vertical) - both and all produces the same result
                                minimum value for shear,
                                maximum value.
        :param kwags: Not used
        :return: A list with the images after a shear transformation
        """

        width, height = images[0].size
        if not self.check_images_equal_size(images):
            print('The shear operation can only be performed when the images have the same dimensions. Aborting')
            return images

        if not isinstance(values, (tuple, list)) or len(values) != 3:
            raise ValueError('The number of values for the shear operation must be a list or tuple with 3 values')
        if values[0] not in self._flip_types:
            raise ValueError(
                'The name {} does not exist for the shear operation. Possible values are: {}'.format(values[0],
                                                                                                     self._flip_types))
        if values[1] > values[2]:
            values = [values[2], values[1]]
        if values[1] < 0 or values[2] > 360:
            raise ValueError("The magnitude range of the shear operation must be greater than 0 and less than 360.")

        direction = values[0]
        if values[0].lower() == 'random' or values[0].lower() == 'both' or values[0].lower() == 'all':
            direction = np.random.choice(['hor', 'ver'], 1)[0]

        angle_to_shear = int(np.random.uniform((abs(values[1]) * -1) - 1, values[2] + 1))
        if angle_to_shear != -1: angle_to_shear += 1

        # We use the angle phi in radians later
        phi = math.tan(math.radians(angle_to_shear))

        outputs = []
        if direction.lower() == "hor" or direction.lower() == 'horizontal':
            # Here we need the unknown b, where a is
            # the height of the image and phi is the
            # angle we want to shear (our knowns):
            # b = tan(phi) * a
            shift_in_pixels = phi * height

            if shift_in_pixels > 0:
                shift_in_pixels = math.ceil(shift_in_pixels)
            else:
                shift_in_pixels = math.floor(shift_in_pixels)

            # For negative tilts, we reverse phi and set offset to 0
            # Also matrix offset differs from pixel shift for neg
            # but not for pos so we will copy this value in case
            # we need to change it
            matrix_offset = shift_in_pixels
            if angle_to_shear <= 0:
                shift_in_pixels = abs(shift_in_pixels)
                matrix_offset = 0
                phi = abs(phi) * -1

            # Note: PIL expects the inverse scale, so 1/scale_factor for example.
            transform_matrix = (1, phi, -matrix_offset,
                                0, 1, 0)

            for image in images:
                image = image.transform((int(round(width + shift_in_pixels)), height),
                                        Image.AFFINE,
                                        transform_matrix,
                                        Image.BICUBIC)

                image = image.crop((abs(shift_in_pixels), 0, width, height))

                outputs.append(image.resize((width, height), resample=Image.BICUBIC))

        elif direction.lower() == "ver" or direction.lower() == 'vertical':
            shift_in_pixels = phi * width

            matrix_offset = shift_in_pixels
            if angle_to_shear <= 0:
                shift_in_pixels = abs(shift_in_pixels)
                matrix_offset = 0
                phi = abs(phi) * -1

            transform_matrix = (1, 0, 0,
                                phi, 1, -matrix_offset)

            for image in images:
                image = image.transform((width, int(round(height + shift_in_pixels))),
                                        Image.AFFINE,
                                        transform_matrix,
                                        Image.BICUBIC)

                image = image.crop((0, abs(shift_in_pixels), width, height))

                outputs.append(image.resize((width, height), resample=Image.BICUBIC))

        return outputs

    def skew(self, images: List[Image.Image], values: list, **kwargs) -> List[Image.Image]:
        """
        Skew images
        :param images(list or array): A list or array of images, each being 3D. This method requires all the images
                                    to be of the same size
        :param values: First value the skew type: TILT, TILT_TOP_BOTTOM, TILT_LEFT_RIGHT, CORNER or RANDOM.
                        The other two are the minimum and maximum skew (0 to 1 values).
                     - ``TILT`` will randomly skew either left, right, up, or down.
                       Left or right means it skews on the x-axis while up and down
                       means that it skews on the y-axis.
                     - ``TILT_TOP_BOTTOM`` will randomly skew up or down, or in other
                       words skew along the y-axis.
                     - ``TILT_LEFT_RIGHT`` will randomly skew left or right, or in other
                       words skew along the x-axis.
                     - ``CORNER`` will randomly skew one **corner** of the image either
                       along the x-axis or y-axis. This means in one of 8 different
                       directions, randomly.
        :param kwags: Extra parameters. They are not used in this method but it is required for consistency
        :return: A list with the skew images
        """
        if not self.check_images_equal_size(images):
            print('The skew operation can only be performed when the images have the same dimensions. Aborting')
            return images

        if not isinstance(values, (tuple, list)) or len(values) != 3:
            raise ValueError('The number of values for the skew operation must be a list or tuple with 3 values')
        if values[0] not in self._skew_types:
            raise ValueError(
                'The name {} does not exist for the skew operation. Possible values are: {}'.format(values[0],
                                                                                                    self._skew_types))

        magnitude = checker('Skew', 'range of skewness', values[1:], 2, 0, 1)

        return skew(images, values[0], magnitude)

    def solarise(self, images: List[Image.Image], values: list, **kwargs) -> List[Image.Image]:
        """
        Perform the solarisation of the image. This operation is computed by creating an inverse of the image, where
        high intensity pixels are changed to low and viceversa (255 - image normally).
        :param images: A list of numpy arrays, each being an image
        :param values: None
        :param kwargs: The parameter mask_positions can be used to avoid using this operation over masks.
        :return: A list wit the images after the solarisation
        """
        no_mask_positions = np.ones(len(images)).astype(bool)
        for pos in kwargs.get('mask_positions', []): no_mask_positions[pos] = False

        output = []
        for i, image in enumerate(images):
            if no_mask_positions[i]:
                output.append(ImageOps.invert(image))
            else:
                output.append(image)

        return output

    def translate(self, images: List[np.ndarray], values: list, **kwargs) -> List[np.ndarray]:
        """
        Translate an image along x, y or both. The way to move is given as in flipping
        :param images: A list of numpy arrays, each being an image
        :param values: A set of 3 or 5 values.
                        1. Type of translation: 'VERTICAL', 'VER', 'HORIZONTAL', 'HOR', 'RANDOM', 'ALL'
                        2. Minimum translation at x position (or both if only 3 values).
                        3. Maximum translation at x position (or both if only 3 values).
                        4. Minimum translation at y position (optional).
                        5. Maximum translation at y position (optional).
                        Values 2 - 5 are relative to the size of the image, so values are between -1 and 1.
        :param kwargs: For this operation, there is only one extra parameters:
                        use_colour: The colour to use. If the colour is not passed or it is a negative value or greater
                        than 255, gaussian noise will be used instead.
                        bounding_boxes (list): A list of lists of lists. A list  for
                                                each element to be bounded to one image (so len(images) == len(bounding_boxes)
                                                Then, for each image a list of bounding boxes, one for each object.
                                                Lastly, each inner list contains 4 values [x0, y0, x1, y1].

                                                IMPORTANT: The number of elements in the outer list is the number
                                                of images. This is different to where the elements are with the
                                                object part. In that case the outer lit does not include masks
        :return:
        """
        use_colour = kwargs.get('use_colour', -1)
        use_replication = kwargs.get('use_replication', False)

        no_mask_positions = np.ones(len(images)).astype(bool)
        for pos in kwargs.get('mask_positions', []): no_mask_positions[pos] = False

        bbs = kwargs.get('bounding_boxes', [])
        if self._is_wrong_bbs(bbs, len(images)):
            raise ValueError('In translate, The bbs were not correctly passed.')

        if not hasattr(use_colour, '__len__'):
            use_colour = [use_colour]

        if not len(use_colour) == 1 and not len(use_colour) == 3:
            raise ValueError(
                'In translate: use_colour must be a single number or a list, tuple or array with 1 or 3 values')

        if not self.check_images_equal_size(images):
            print('The translate operation can only be performed when the images have the same dimensions. Aborting')
            return images

        values = list(values)
        if not isinstance(values, (tuple, list)) or (len(values) != 3 and len(values) != 5):
            raise ValueError(
                'The number of values for the translation operation must be a list or tuple with 3 or 5 values')
        if values[0].upper() not in self._flip_types:
            raise ValueError(
                'The name {} does not exist for the translate operation. Possible values are: {}'.format(values,
                                                                                                         self._flip_types))
        for i, v in enumerate(values[1:]):
            if len(values) == 5:
                j = 1 - i // 2  # There are four values (two ranges) and every two we use the same values of the shape. The inversion is because PIL uses x,y instead of height, width
            else:
                j = 1 - i

            if isinstance(v, float) and (v > 1.0 or v < -1.0):
                raise ValueError('When float is used, the values must be between -1 and 1 inclusive.')
            if isinstance(v, float):
                values[i + 1] = int(images[0].shape[j] * v)
            elif isinstance(v, int):
                if v > images[0].shape[j] or v < -images[0].shape[j]:
                    raise ValueError(
                        'When integers are used, the values for translation must be within the size of the image.')
            else:
                raise TypeError('Only float and integers are allowed for translate.')

        if values[1] > values[2]:
            values = [values[2], values[1]]

        if len(values) == 3:
            tx = int(np.random.uniform(values[1], values[2]))
            ty = int(np.random.uniform(values[1], values[2]))
        else:
            tx = int(np.random.uniform(values[1], values[2]))
            ty = int(np.random.uniform(values[3], values[4]))

        if values[0].lower() == 'random':
            values[0] = np.random.choice(['horizontal', 'vertical', 'all'], 1)[0]

        if values[0].lower() == 'horizontal' or values[0].lower() == 'hor':
            ty = 0

        if values[0].lower() == 'vertical' or values[0].lower() == 'ver':
            tx = 0

        h, w = images[0].shape[:2]

        for i, bbs_i in enumerate(bbs):
            for ii, bbi in enumerate(bbs_i):
                bbs[i][ii] = self._correct_bb([bbi[0] + tx, bbi[1] + ty, bbi[2] + tx, bbi[3] + ty], [w, h])

        bb_diff = [0, 0, 0, 0]
        if kwargs.get('object_detection', False):
            h += abs(ty)
            w += abs(tx)
            bb_diff = [min(tx, 0), min(ty, 0), max(tx, 0), max(ty, 0)]
        # ty = max(ty, 0)
        # tx = max(tx, 0)

        output = []
        for i, image in enumerate(images):
            if len(image.shape) == 2:
                image = image[:, :, np.newaxis]
            c = image.shape[2]

            colour = use_colour
            if c == 1 and len(use_colour) == 3:
                colour = use_colour[0:1]
            elif c == 3 and len(use_colour) == 1:
                colour = [use_colour[0], use_colour[0], use_colour[0]]

            if not no_mask_positions[i]:
                colour = [0 for _ in colour]

            if colour[0] < 0 or colour[0] > 255:
                im = self._std_noise * np.abs(np.random.randn(h, w, c)) + 127.5
                im[im < 0] = 0
                im[im > 255] = 255
            else:
                im = np.array(colour)[None, None, :] * np.ones((h, w, c))

            image2 = image.copy()
            if not kwargs.get('object_detection', False):
                image2 = image[max(-ty, 0): min(h - ty, h), max(-tx, 0): min(w - tx, w), ...]

            im[max(ty, 0): min(h + ty, h), max(tx, 0): min(w + tx, w), ...] = image2

            if use_replication and no_mask_positions[i]:
                init_y = max(ty, 0)
                end_y = min(h + ty, h)
                init_x = max(tx, 0)
                end_x = min(w + tx, w)
                init_x_r = max(-tx, 0)
                init_y_r = max(-ty, 0)

                if ty > 0:
                    im_aux = np.vstack([image[init_y::-1, :, ...], image])
                else:
                    im_aux = np.vstack([image, image[-1:end_y - h - 1:-1, :, ...]])

                if tx > 0:
                    im_aux = np.hstack([im_aux[:, init_x::-1, ...], im_aux])
                else:
                    im_aux = np.hstack([im_aux, im_aux[:, -1:end_x - w - 1:-1, ...]])

                im = im_aux[init_y_r:h + init_y_r, init_x_r:w + init_x_r, ...]

            output.append(self.rescale(np.squeeze(im)))

        if kwargs.get('object_detection', False):
            output = {'images': output, 'bb_diff': bb_diff}
        if bbs:
            if not isinstance(output, dict):
                output = {'images': output}
            output['bounding_boxes'] = bbs

        return output

    def zoom(self, images: List[np.ndarray], values: list, **kwargs) -> List[np.ndarray]:
        """
        Zoom an image. This means to resize the image and then cropping it if the new size is larger or adding noise
        padding if it is smaller.
        :param images: A list of images
        :param values: Tuple with the range of values of the zoom factor. Values must be between 0.1 and 10
        :param kwargs: There are two values:
                        - use_colour: When the zoom is smaller than 1, outside of the image will be padded with a single
                                    colour, use a value outside of the range [0, 255] for noise.
                        - use_replication: When True the regions outside of the original imagea are padded with the image
                                            replicated from the closest pixels.
                        - bounding_boxes (list): A list of lists of lists. A list  for
                                                each element to be bounded to one image (so len(images) == len(bounding_boxes)
                                                Then, for each image a list of bounding boxes, one for each object.
                                                Lastly, each inner list contains 4 values [x0, y0, x1, y1].

                                                IMPORTANT: The number of elements in the outer list is the number
                                                of images. This is different to where the elements are with the
                                                object part. In that case the outer lit does not include masks
                        - keep_size (optional): Return the images after zooming obtaining a different size after this.
        :return: A list with the zoomed images
        """
        w, h = images[0].shape[:2]
        c = 1
        if len(images[0].shape) > 2:
            c = images[0].shape[2]
        # h, w = images[0].size
        # c = len(images[0].getbands())
        if not self.check_images_equal_size(images):
            print('The zoom operation can only be performed when the images have the same dimensions. Aborting')
            return images

        keep_size = kwargs.get('keep_size', False)

        bbs = kwargs.get('bounding_boxes', [])
        if self._is_wrong_bbs(bbs, len(images)):
            raise ValueError('In zoom, The bbs were not correctly passed.')

        use_replication = kwargs.get('use_replication', False)
        use_colour = kwargs.get('use_colour', -1)

        no_mask_positions = np.ones(len(images)).astype(bool)
        for pos in kwargs.get('mask_positions', []): no_mask_positions[pos] = False

        if not hasattr(use_colour, '__len__'):
            use_colour = [use_colour]

        if not len(use_colour) == 1 and not len(use_colour) == 3:
            raise ValueError(
                'In zoom: use_colour must be a single number or a list, tuple or array with 1 or 3 values')

        factor = checker('zoom', 'the range of zoom', values, 2, 0.1, 10)
        h_new, w_new = int(factor * h), int(factor * w)

        dim = [w, h, c]
        # if c == 1:
        #	dim = [w, h]

        output = []
        for i, image in enumerate(images):
            # image = image.resize((h_new, w_new))
            im_aux = image.copy()
            image = imresize(image, (w_new, h_new), anti_aliasing=True, preserve_range=True)
            if not keep_size:
                dif_h = int(np.round(np.abs(h_new - h) / 2))
                dif_w = int(np.round(np.abs(w_new - w) / 2))
                if dif_h == 0 or dif_w == 0:
                    output.append(im_aux)
                    continue
                if factor < 1:
                    # image = np.array(image)
                    if use_replication and no_mask_positions[i]:
                        im = np.zeros(tuple(dim))
                        im[dif_w: w_new + dif_w, dif_h:h_new + dif_h, ...] = image

                        if dif_w > w_new:
                            rep = 2 * np.ceil(dif_w / w_new).astype(int) // 2 + 1
                            if len(image.shape) == 2:
                                image = np.tile(image, [rep, 1])
                            else:
                                image = np.tile(image, [rep, 1, 1])
                            for i in range(1, rep, 2):
                                image[i * w_new + 1:(i + 1) * w_new, ...] = image[(i + 1) * w_new - 1:i * w_new:-1, ...]
                        im[:dif_w, dif_h:h_new + dif_h, ...] = image[dif_w - 1::-1, :, ...]
                        w2 = image.shape[0] - 1
                        im[w_new + dif_w:, dif_h:h_new + dif_h, ...] = image[w2:w2 - (im.shape[0] - w_new - dif_w):-1,
                                                                       :, ...]

                        im_aux = im[:, dif_h:h_new + dif_h, ...]
                        if dif_h > h_new:
                            rep = 2 * np.ceil(dif_h / h_new).astype(int) // 2 + 1
                            if len(image.shape) == 2:
                                im_aux = np.tile(im_aux, [1, rep])
                            else:
                                im_aux = np.tile(im_aux, [1, rep, 1])
                            for i in range(1, rep, 2):
                                image[:, i * h_new + 1:(i + 1) * h_new, ...] = image[:,
                                                                               (i + 1) * h_new - 1:i * h_new:-1, ...]
                        h2 = im_aux.shape[1] - 1
                        im[:, :dif_h:, ...] = im_aux[:, dif_h - 1::-1, ...]
                        im[:, h_new + dif_h:, ...] = im_aux[:, h2:h2 - (im.shape[1] - h_new - dif_h):-1, ...]
                    else:
                        colour = use_colour
                        if c == 1 and len(use_colour) == 3:
                            colour = use_colour[0:1]
                        elif c == 3 and len(use_colour) == 1:
                            colour = [use_colour[0], use_colour[0], use_colour[0]]

                        if not no_mask_positions[i]:
                            colour = [0 for _ in colour]

                        if colour[0] < 0 or colour[0] > 255:
                            im = self._std_noise * np.abs(np.random.randn(w, h, c)) + 127.5
                            im[im < 0] = 0
                            im[im > 255] = 255
                        else:
                            im = np.array(colour)[None, None, :] * np.ones((w, h, c))

                        im[dif_w: w_new + dif_w, dif_h:h_new + dif_h, ...] = image
                    diff = [dif_h, dif_w]
                    image = self.rescale(im)
                if factor > 1:
                    diff = [-dif_h, -dif_w]
                    if not kwargs.get('object_detection', False):
                        image = image[dif_w: w + dif_w, dif_h:h + dif_h, ...]
                    # image = image.crop((dif_h, dif_w, h + dif_h, w + dif_w))
            output.append(image)

        # if kwargs.get('object_detection', False) and factor > 1:
        #	output = {'images': output, 'bb_diff': [-dif_h, -dif_w, dif_h, dif_w]}
        if bbs:
            if dif_h != 0 or dif_w != 0:
                for i, bbs_i in enumerate(bbs):
                    for ii, bbi in enumerate(bbs_i):
                        bbs[i][ii] = self._correct_bb([bbi[iii] * factor + diff[iii % 2] for iii in range(len(bbi))],
                                                      [h, w])

            if not isinstance(output, dict):
                output = {'images': output}
            output['bounding_boxes'] = bbs

        return output
