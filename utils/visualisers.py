from typing import List

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np


def plot_bb_image(image: np.ndarray, bbs: List[List[float]] = [], relative_size_bb=True, ax: plt.Axes = None):
    """
    Plot an image and a set of bounding boxes.
    :param image: The image as a numpy array
    :param bbs:  A list of lists, which inner list having five values. The first one being the class and then
                        [xmin, xmax, ymin, ymax]
    :param relative_size_bb: When True the bounding boxes are assumed to be relative to the size of the image. So
                            values between 0 and 1.
    :param ax: A plt.Axes object where to plot the image and bounding boxes when required. If not passed, it will create
                one.
    :return: None.
    """
    pass_ax = True
    if ax is None:
        pass_ax = False
        # Create figure and axes
        fig, ax = plt.subplots(1)

    # Display the image
    ax.imshow(image.astype(np.uint8))

    h, w = image.shape[:2]
    sizes = [w, h, w, h]

    colors = ['r', 'g', 'b', 'k']
    for bb in bbs:
        label = bb[0]
        color = colors[label]

        if relative_size_bb:
            #Convert bounding boxes from relative to absolute values
            bb = [int(round(size * coord)) for coord, size in zip(bb[1:], sizes)]

        # Create a Rectangle patch
        rect = patches.Rectangle((bb[0], bb[1]), bb[2] - bb[0], bb[3] - bb[1], linewidth=2, edgecolor=color,
                                 facecolor='none')

        # Add the patch to the Axes
        ax.add_patch(rect)

    if not pass_ax:
        plt.axis('off')
        plt.show()


def plot_images(images: List[np.ndarray], masks=None, bbs=None, figsize=(15, 15)):
    """
    Plot a group of images with bounding boxes if required in the same plot. The structure is 4 images per row. When
    masks are passed, it will plot image, mask, image, mask.
    :param images: A list of numpy arrays containing the images
    :param masks:A list of numpy arrays with the masks. None by default
    :param bbs: A list with a list of bounding boxes.
    :param figsize: A tuple with the size of the image
    :return: None
    """

    num_plots = np.ceil(len(images) / 4) if masks is None else np.ceil(len(images) / 2)
    fig, axs = plt.subplots(int(num_plots), 4, figsize=figsize)

    if bbs is None:
        bbs = [[]] * len(images)
    n = 0
    for i, image in enumerate(images):
        r = n // 4
        c = n % 4
        plot_bb_image(image, bbs=bbs[i], ax=axs[r][c])
        axs[r][c].set_axis_off()

        if masks is not None:
            n += 1
            plot_bb_image(masks[i], ax=axs[r][c+1])
            axs[r][c+1].set_axis_off()
        n += 1

    plt.axis('off')
    plt.show()
