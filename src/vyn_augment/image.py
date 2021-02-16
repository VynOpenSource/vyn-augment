from typing import Union, List
from PIL import Image
import numpy as np

from skimage.color import rgb2hsv, hsv2rgb


def image_extensions():
    return ['.jpg', '.jpeg', '.png', '.tiff', '.bmp', '.pgm']


def is_image(file_name: str):
    """
    Return true if the image is an image file
    :param file_name: The name of the file to check
    :return: True if the file is an image
    """
    extensions = image_extensions()
    if any([True if ext in file_name.lower() else False for ext in extensions]):
        return True

    return False


def enforce_3_channel_image(im):
    if np.ndim(im) < 3:
        return np.expand_dims(im, 2)
    if im.shape[2] < 3:
        return np.repeat(np.expand_dims(im[:, :, 0], 2), 3, axis=2)
    else:
        return im


def _rand_scale(scale):
    scale = np.random.uniform(1, scale)
    return scale if (np.random.randint(2) == 0) else 1. / scale


def random_distort_image(image, hue=18, saturation=1.5, exposure=1.5):
    # determine scale factors
    dhue = np.random.uniform(-hue, hue)
    dsat = _rand_scale(saturation)
    dexp = _rand_scale(exposure)

    # convert RGB space to HSV space
    image = rgb2hsv(image).astype('float')

    # change satuation and exposure
    image[:, :, 1] *= dsat
    image[:, :, 2] *= dexp

    # change hue
    image[:, :, 0] += dhue
    image[:, :, 0] -= (image[:, :, 0] > 180) * 180
    image[:, :, 0] += (image[:, :, 0] < 0) * 180

    # convert back to RGB from HSV
    return hsv2rgb(image).astype('float')  # cv2.cvtColor(image.astype('uint8'), cv2.COLOR_HSV2RGB)


def skew(images: List[Image.Image], skew_type: str, magnitude: float):
    """
        As well as the required :attr:`probability` parameter, the type of
        skew that is performed is controlled using a :attr:`skew_type` and a
        :attr:`magnitude` parameter. The :attr:`skew_type` controls the
        direction of the skew, while :attr:`magnitude` controls the degree
        to which the skew is performed.
        To see examples of the various skews, see :ref:`perspectiveskewing`.
        Images are skewed **in place** and an image of the same size is
        returned by this function. That is to say, that after a skew
        has been performed, the largest possible area of the same aspect ratio
        of the original image is cropped from the skewed image, and this is
        then resized to match the original image size. The
        :ref:`perspectiveskewing` section describes this in detail.
        :param skew_type: Must be one of ``TILT``, ``TILT_TOP_BOTTOM``,
         ``TILT_LEFT_RIGHT``, or ``CORNER``.
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
         To see examples of the various skews, see :ref:`perspectiveskewing`.
        :param magnitude: The degree to which the image is skewed.
        :type skew_type: String
        :type magnitude: Float 0 to 1.

        :param images: The image(s) to skew.
        :type images: List containing PIL.Image object(s).
        :return: The transformed image(s) as a list of object(s) of type
         PIL.Image.
        """

    # Width and height taken from first image in list.
    # This requires that all ground truth images in the list
    # have identical dimensions!
    w, h = images[0].size

    x1 = 0
    x2 = h
    y1 = 0
    y2 = w

    original_plane = [(y1, x1), (y2, x1), (y2, x2), (y1, x2)]

    max_skew_amount = max(w, h)
    max_skew_amount = int(np.ceil(max_skew_amount * magnitude))
    if max_skew_amount <= 1:
        skew_amount = 1
    else:
        skew_amount = np.random.randint(1, max_skew_amount)

    # Old implementation, remove.
    # if not self.magnitude:
    #    skew_amount = random.randint(1, max_skew_amount)
    # elif self.magnitude:
    #    max_skew_amount /= self.magnitude
    #    skew_amount = max_skew_amount

    if skew_type == "RANDOM":
        skew = np.random.choice(["TILT", "TILT_LEFT_RIGHT", "TILT_TOP_BOTTOM", "CORNER"])
    else:
        skew = skew_type

    # We have two choices now: we tilt in one of four directions
    # or we skew a corner.

    if skew == "TILT" or skew == "TILT_LEFT_RIGHT" or skew == "TILT_TOP_BOTTOM":

        if skew == "TILT":
            skew_direction = np.random.randint(0, 3)
        elif skew == "TILT_LEFT_RIGHT":
            skew_direction = np.random.randint(0, 1)
        elif skew == "TILT_TOP_BOTTOM":
            skew_direction = np.random.randint(2, 3)

        if skew_direction == 0:
            # Left Tilt
            new_plane = [(y1, x1 - skew_amount),  # Top Left
                         (y2, x1),  # Top Right
                         (y2, x2),  # Bottom Right
                         (y1, x2 + skew_amount)]  # Bottom Left
        elif skew_direction == 1:
            # Right Tilt
            new_plane = [(y1, x1),  # Top Left
                         (y2, x1 - skew_amount),  # Top Right
                         (y2, x2 + skew_amount),  # Bottom Right
                         (y1, x2)]  # Bottom Left
        elif skew_direction == 2:
            # Forward Tilt
            new_plane = [(y1 - skew_amount, x1),  # Top Left
                         (y2 + skew_amount, x1),  # Top Right
                         (y2, x2),  # Bottom Right
                         (y1, x2)]  # Bottom Left
        elif skew_direction == 3:
            # Backward Tilt
            new_plane = [(y1, x1),  # Top Left
                         (y2, x1),  # Top Right
                         (y2 + skew_amount, x2),  # Bottom Right
                         (y1 - skew_amount, x2)]  # Bottom Left

    if skew == "CORNER":

        skew_direction = np.random.randint(0, 7)

        if skew_direction == 0:
            # Skew possibility 0
            new_plane = [(y1 - skew_amount, x1), (y2, x1), (y2, x2), (y1, x2)]
        elif skew_direction == 1:
            # Skew possibility 1
            new_plane = [(y1, x1 - skew_amount), (y2, x1), (y2, x2), (y1, x2)]
        elif skew_direction == 2:
            # Skew possibility 2
            new_plane = [(y1, x1), (y2 + skew_amount, x1), (y2, x2), (y1, x2)]
        elif skew_direction == 3:
            # Skew possibility 3
            new_plane = [(y1, x1), (y2, x1 - skew_amount), (y2, x2), (y1, x2)]
        elif skew_direction == 4:
            # Skew possibility 4
            new_plane = [(y1, x1), (y2, x1), (y2 + skew_amount, x2), (y1, x2)]
        elif skew_direction == 5:
            # Skew possibility 5
            new_plane = [(y1, x1), (y2, x1), (y2, x2 + skew_amount), (y1, x2)]
        elif skew_direction == 6:
            # Skew possibility 6
            new_plane = [(y1, x1), (y2, x1), (y2, x2), (y1 - skew_amount, x2)]
        elif skew_direction == 7:
            # Skew possibility 7
            new_plane = [(y1, x1), (y2, x1), (y2, x2), (y1, x2 + skew_amount)]

    if skew_type == "ALL":
        # Not currently in use, as it makes little sense to skew by the same amount
        # in every direction if we have set magnitude manually.
        # It may make sense to keep this, if we ensure the skew_amount below is randomised
        # and cannot be manually set by the user.
        corners = dict()
        corners["top_left"] = (y1 - np.random.randint(1, skew_amount), x1 - np.random.randint(1, skew_amount))
        corners["top_right"] = (y2 + np.random.randint(1, skew_amount), x1 - np.random.randint(1, skew_amount))
        corners["bottom_right"] = (y2 + np.random.randint(1, skew_amount), x2 + np.random.randint(1, skew_amount))
        corners["bottom_left"] = (y1 - np.random.randint(1, skew_amount), x2 + np.random.randint(1, skew_amount))

        new_plane = [corners["top_left"], corners["top_right"], corners["bottom_right"], corners["bottom_left"]]

    # To calculate the coefficients required by PIL for the perspective skew,
    # see the following Stack Overflow discussion: https://goo.gl/sSgJdj
    matrix = []

    for p1, p2 in zip(new_plane, original_plane):
        matrix.append([p1[0], p1[1], 1, 0, 0, 0, -p2[0] * p1[0], -p2[0] * p1[1]])
        matrix.append([0, 0, 0, p1[0], p1[1], 1, -p2[1] * p1[0], -p2[1] * p1[1]])

    A = np.array(matrix, dtype=np.float)
    B = np.array(original_plane).reshape(8)

    perspective_skew_coefficients_matrix = np.linalg.pinv(A) @ B

    def do(image):
        return image.transform(image.size,
                               Image.PERSPECTIVE,
                               perspective_skew_coefficients_matrix,
                               resample=Image.BICUBIC)

    augmented_images = []

    for image in images:
        augmented_images.append(do(image))

    return augmented_images


def create_circular_mask(h: int, w: int, center: List[int] = None, radius: int = None,
                         get_distance_map: bool = False):
    """
    Create a circular mask given the size of an image, center and radius of the circle.
    :param h: Height of the image
    :param w: width of the image
    :param center: Position of the image
    :param radius: Radius of the circle
    :param get_distance_map: Whether to return a distance map as well as second parameter. This map will have
                                        values from 0 to 1, where 1 is the center of the disc and 0 is the furthest
                                        pixel in the image.
    :return: The mask and the distance map as optional.
    """

    if center is None:  # use the middle of the image
        center = (int(w / 2), int(h / 2))
    if radius is None:  # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w - center[0], h - center[1])

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2)

    mask = dist_from_center <= radius

    if get_distance_map:
        return mask, mask * (1 - dist_from_center / np.max(dist_from_center))
    return mask


def convert_to_absolute(values: List[float], shapes: List[int]):
    """
    Convert a value from relative to absolute if it is not
    :param values: a List with two values for height and width
    :param shapes: The shape of the image
    :return: Absolute values
    """
    return [int(value * shape) for value, shape in zip(values, shapes) if 1 > value > 0]


def checker(name: str, name_parameters: str, values: Union[list, np.ndarray], len_val: int, min_val: Union[float, int],
            max_val: Union[float, int]):
    """
    Check that set of values are within two other values and have a certain number of values.
    If not return an error with the name of the application or function and the name of the parameters with values
    :param name: The name of the function
    :param name_parameters: The name of the parameters with values
    :param values: Values to check that are within a range
    :param len_val: The number of parameters that values should have
    :param min_val: The minimum value of the range where values should be.
    :param max_val: The maximum value of the range where values should be.
    :return: ordered values
    """

    if not hasattr(values, '__len__') or len(values) != len_val:
        raise ValueError('The number of values for the {} operation must be a list or tuple with 2 values'.format(name))
    check_range(name, name_parameters, values, min_val, max_val)

    return np.random.uniform(values[0], values[1])


def check_range(name: str, name_parameters: str, values: Union[list, np.ndarray], min_val: Union[float, int],
                max_val: Union[float, int]):
    """
    Check that a range (two values) are correct
    :param name: The name of the function
    :param name_parameters: The name of the parameters with values
    :param values: Values to check that are within a range
    :param min_val: The minimum value of the range where values should be.
    :param max_val: The maximum value of the range where values should be.
    :return: None
    """
    if values[0] > values[1]:
        values = values[::-1]
    if values[0] < min_val or values[1] > max_val:
        raise ValueError(
            "The {} from operation {} must be between {} and {}.".format(name_parameters, name, min_val, max_val))


def swap_patches(images: List[np.ndarray], values: List[Union[str, int]], name_op: str,
                 swapped_images: List[np.ndarray], **kwargs):
    """
    Remove some patches from a set of images and changed them from patches in the same position of another image. To
    use it for occlusion, the swapped images can be noise or black
    :param images: A list of numpy arrays, each being an image
    :param values: 5 values:
                    str: type of occlusion
                    int: Minimum number of boxes columns directions
                    int: Maximum number of boxes columns directions
                    int: Minimum number of boxes rows directions
                    int: Maximum number of boxes rows directions
                    Selection is done by a uniform distribution between minimum and maximum values.
    :param name_op: The name of the operation for printing proper errors.
    :param swapped_images: A list with the images to get patches for the swapping. Passed noise if not images are
                            required.
    :param kwargs: For this operation, the only extra parameter is the whether an image is a mask.
                    mask_positions: The positions in images that are masks.
    :return:
    """
    h, w = images[0].shape[:2]

    for image, swapped_image in zip(images, swapped_images):
        if (image.shape != swapped_image.shape):
            raise ValueError('In {}, images and swapped images must have the same size'.format(name_op))

    values = list(values)
    for i, value in enumerate(values[1:]):
        max_v = h
        if i // 2 > 0:
            max_v = w
        if isinstance(value, float):
            if values[0].lower == 'hide_and_seek':
                raise TypeError('For Hide and seek mode only integers are allowed')
            if value > 1.0:
                raise ValueError('When float the number must be between 0 and 1 for occlusion size.')
            values[i + 1] = max_v * value

        elif isinstance(value, int):
            if (value <= 0 or value > max_v):
                if values[0].lower == 'hide_and_seek':
                    raise ValueError(
                        'The size of the grid for hide and seek {} patch cannot smaller or equal than 0 or larger than the size of the image'.format(
                            name_op))
                else:
                    raise ValueError(
                        'The size of the {} patch cannot be larger than the size of the image'.format(name_op))
        else:
            raise TypeError('In {} the type must be integers or float'.format(name_op))

    ver = int(np.random.uniform(values[1], values[2]))
    hor = int(np.random.uniform(values[3], values[4]))

    num_patches = kwargs.get('number_patches', 1)
    if not isinstance(num_patches, (int, float)) and not hasattr(num_patches, '__len__') or isinstance(num_patches,
                                                                                                       str):
        raise TypeError('Type {} is not an acceptable type for specifying the number of patches to occlude.',
                        type(num_patches))

    if not hasattr(num_patches, '__len__'):
        num_patches = num_patches if num_patches > 1 else kwargs.get('num_patches', 1)

    if hasattr(num_patches, '__len__'):
        num_patches = np.round(
            checker('occlusion', 'range of number of patches', num_patches, 2, 0, ver * hor - 1))

    new_images = [image.astype(float) for image in images]

    if values[0].lower() == 'hide_and_seek':
        ver = np.round(h / float(ver)).astype(int)
        hor = np.round(w / float(hor)).astype(int)
        num_divisions = ver * hor
        selected_patches = np.random.choice(np.arange(num_divisions), num_patches, replace=False)

        for patch_pos in selected_patches:
            i = patch_pos // ver
            j = patch_pos - i * ver

            size_w = w // hor
            size_v = h // ver

            for ii, image in enumerate(new_images):
                ch = image.shape[2] if len(image.shape) > 2 else 1
                patch = swapped_images[ii][j * size_v:(j + 1) * size_v, i * size_w:(i + 1) * size_w, ...]
                new_images[ii][j * size_v:(j + 1) * size_v, i * size_w:(i + 1) * size_w, ...] = patch
    else:
        for i in range(num_patches):
            ver = int(np.random.uniform(values[1], values[2]))
            hor = int(np.random.uniform(values[3], values[4]))

            center_x = int(np.random.uniform(hor // 2, w - hor // 2))
            center_y = int(np.random.uniform(ver // 2, h - ver // 2))

            for ii, image in enumerate(new_images):
                ch = image.shape[2] if len(image.shape) > 2 else 1

                a = ver % 2
                b = hor % 2
                patch = swapped_images[ii][center_y - ver // 2:center_y + ver // 2 + a,
                        center_x - hor // 2:center_x + hor // 2 + b, ...]
                new_images[ii][center_y - ver // 2:center_y + ver // 2 + a, center_x - hor // 2:center_x + hor // 2 + b,
                ...] = patch

    return new_images


def create_grid_masks(images: List[np.ndarray], init_position: List[int], size_square: List[int],
                      dist_between_squares: List[int], image_to_use: List[np.ndarray], is_not_mask: List[bool]):
    """
    Create a grid of masks with intensities given by image_to_use. The dimensions of the grid of masks is given
    by the size of the squares, the distance between squares and the most top left corner.

    :param images: A list with the images to add the grid of masks
    :param init_position: x and y of the most top left corner
    :param size_square: The width and height of the squares
    :param dist_between_squares: The separation between squares in x and y
    :param image_to_use: The image to use to extract the squares. It is assumed that image_to_use has the same channels
                        as images.
    :param is_not_mask: A list of boolean specifying whether the images are masks or not.
    """
    if not any(is_not_mask):
        return images

    im_index = is_not_mask.index(True)
    xo, yo = init_position
    sx, sy = size_square
    dx, dy = dist_between_squares
    shape = images[im_index].shape

    # To create the mask, we are going to create one block containing the separation and one square, then it is going
    # to be repeated until the size of the image, then it is going to be padded and cropped to the size of the image.
    mask_base = np.zeros((sy + dy, sx + dx))
    mask_base[:sy, :sx] = 1

    ny = int(np.ceil(float(shape[0] - yo) / (sy + dy)))
    nx = int(np.ceil(float(shape[1] - xo) / (sx + dx)))

    mask_base = np.tile(mask_base, [ny, nx])
    mask_base = np.pad(mask_base, pad_width=((yo, 0), (xo, 0)), constant_values=((0, 0), (0, 0)), mode='constant')
    mask_base = mask_base[:shape[0], :shape[1]]

    output = []
    for i, image in enumerate(images):
        mask = mask_base
        if len(shape) == 3:
            mask = np.tile(mask_base[..., None], [1, 1, 3])

        image[mask == 1] = image_to_use[i][mask == 1]
        output.append(image)

    return output
