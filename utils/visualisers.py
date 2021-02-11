from typing import List

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np


def plot_bb_image(image: np.ndarray, bbs: List[List[float]], relative_size_bb=True):
    """
    Plot an image and a set of bounding boxes.
    :param image: The image as a numpy array
    :param bbs:  A list of lists, which inner list having five values. The first one being the class and then
                        [xmin, xmax, ymin, ymax]
    :param relative_size_bb: When True the bounding boxes are assumed to be relative to the size of the image. So
    :var                    values between 0 and 1.
    :return: None
    """

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

    plt.axis('off')
    plt.show()
