# imaging.py

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm


def imagify(input, pred, truth, n=3, randomize=True, null_code=0):
    """Process input, prediction and mask data ready for display

    Args:
        inputs: Input tensor from a batch
        predictions: Predictions tensor from a batch
        truth: Truth tensor from a batch
        n: The number of images to extract from the batch (default: 3)
        randomize: Choose random images from the batch if True, choose the first n otherwise (default: True)
        null_code: The null mask code (default: 0)

    Returns:
        A tuple (if n == 1) or zip of the processed images ready for display.
    """
    # Select the images to process
    choices = np.random.choice(np.array(range(inputs.shape[0])), size=n) if randomize else np.array(range(n))
    input_imgs = input[choices,0,...]
    truth_imgs = truth[choices,...]

    input_imgs = input_imgs.detach().cpu()
    truth_imgs = truth_imgs.detach().cpu()
    pred_imgs = pred[choices,...].detach().cpu()

    # Remove non-hit regions
    mask = truth_imgs == null_code
    pred_imgs = np.argmax(pred_imgs, axis=1)
    pred_imgs = np.ma.array(pred_imgs, mask = mask).filled(0)

    return zip(input_imgs, truth_imgs, pred_imgs) if n > 1 else (input_imgs, truth_imgs, pred_imgs)


def show_batch(epoch, batch, input, pred, truth, null_code=0, n=3, randomize=True):
    """Display the images for a given epoch and batch. Each row is a triplet of input, prediction and mask.

    Args:
        epoch: The current training epoch
        batch: The current training batch
        input: Input tensor from a batch
        pred: Predictions tensor from a batch
        truth: Truth tensor from a batch
        n: The number of images to extract from the batch (default: 3)
        randomize: Choose random images from the batch if True, choose the first n otherwise (default: True)
        null_code: The null mask code (default: 0).
    """
    ax = None
    rows, cols, size = 1, 2, 9
    cmap = ListedColormap(['white', 'red', 'cornflowerblue'])
    norm = BoundaryNorm([0., 0.5, 1.5, 2.5], cmap.N)
    xtr = dict(cmap=cmap, norm=norm, alpha=0.7)

    images = imagify(input, pred, truth, n, randomize, null_code)

    for i, imgs in enumerate(images):
        raw, cls, net = imgs
        pair = (cls, net)
        fig, axs = plt.subplots(1, cols, figsize=(cols * size, size))
        for img, ax in zip(pair, axs):
            ax.imshow(img, **xtr)
            ax.axis('off')
        plt.tight_layout()
        save_figure(plt, "output_{}_{}_{}".format(epoch, batch, i))
        plt.close(fig)


def save_figure(fig, name):
    """Output a matplotlib figure PNG, PDF and EPS formats.

    Args:
        fig (Figure): The matplotlib figure to save.
        name (str): The output filename excluding extension.
    """
    fig.savefig(name + ".png")
    #fig.savefig(name + ".pdf")
    #fig.savefig(name + ".eps")


def get_supported_formats():
    """Retrieve the supported image formats.

    Returns:
        A dictionary containing strings of file format descriptions keyed by extension.
    """
    return plt.gcf().canvas.get_supported_filetypes()