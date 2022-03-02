"""Dataset loaders for MNIST, fashion MNIST, and Genz time series"""
import os

import numpy as np
import torch
import torchvision
from torchvision import transforms as tf


def load_mnist(
    fashion=False,
    num_train=60000,
    num_test=10000,
    num_val=None,
    downscale=False,
    downscale_shape=None,
    num_bins=None,
    continuous=True,
    dataset_dir=None,
    device=None,
):
    """
    Function for loading unlabeled MNIST and FashionMNIST datasets

    Supports downscaling, discretizing images via binning, and splitting into
    extra validation set (pulled from training data).
    """
    assert 0 <= num_train <= 60000
    assert 0 <= num_test <= 10000
    if num_val is not None and num_val + num_train > 60000:
        num_train = 60000 - num_val

    if dataset_dir is None:
        dataset_dir = "./datasets/"

    if fashion:
        dataset = torchvision.datasets.FashionMNIST
        dset_dir = dataset_dir + "fashion_mnist/"
    else:
        dataset = torchvision.datasets.MNIST
        dset_dir = dataset_dir + "mnist/"

    # Get the desired number of images, separate into splits
    train = dataset(root=dset_dir, train=True, download=True, transform=tf.ToTensor())
    test = dataset(root=dset_dir, train=False, download=True, transform=tf.ToTensor())
    train_data = train.data[:num_train]
    test_data = test.data[:num_test]
    if num_val is None:
        sizes = (num_train, num_test)
        out = (train_data, test_data)
    else:
        val_data = train.data[-num_val:]
        sizes = (num_train, num_val, num_test)
        out = (train_data, val_data, test_data)

    # Resize images, flatten, and rescale values
    transform = tf.Resize(downscale_shape) if downscale else lambda x: x
    out = tuple(transform(ds).reshape(ns, -1) / 255 for ns, ds in zip(sizes, out))

    # Move datasets to the appropriate device
    if device is not None:
        out = tuple(ds.to(device) for ds in out)

    # Finally, discretize images and put in pytorch Dataset
    if num_bins is not None and not continuous:
        out = tuple(bin_data(ds, num_bins) for ds in out)
    return tuple(torch.utils.data.TensorDataset(ds) for ds in out)


@torch.no_grad()
def bin_data(input, num_bins=None):
    """
    Discretize greyscale values into a finite number of bins
    """
    if num_bins is None:
        return input
    assert num_bins > 0

    # Set each of the corresponding bin indices
    out_data = torch.full_like(input, -1)
    for i in range(num_bins):
        bin_inds = (i / num_bins <= input) * (input <= (i + 1) / num_bins)
        out_data[bin_inds] = i
    assert out_data.max() >= 0

    return out_data.long()


def load_genz(genz_num: int):
    """
    Load a dataset of time series with dynamics set by various Genz functions

    Separate train, validation, and test datasets are returned, containing
    8000, 1000, and 1000 time series. Each time series has length 100.

    Args:
        genz_num: Integer between 1 and 6 setting choice of Genz function

    Returns:
        train, val, test: Three arrays with respective shape (8000, 100, 1),
            (1000, 100, 1), and (1000, 100, 1).
    """
    assert 1 <= genz_num <= 6
    # Return saved dataset if we have already generated this previously
    save_file = f"datasets/genz/genz_{genz_num}.npz"
    if os.path.isfile(save_file):
        out = np.load(save_file)
        train, val, test = out["train"], out["val"], out["test"]
        assert val.shape == test.shape == (1000, 100, 1)
        assert train.shape == (8000, 100, 1)
        return train, val, test

    # Definitions of each of the Genz functions which drive the time series
    gfun = genz_funs[genz_num]

    # Initialize random starting values and update using Genz update function
    rng = np.random.default_rng(genz_num)
    x = rng.permutation(np.linspace(0.0, 1.0, num=10000))[:, None]
    all_series = np.empty((10000, 100, 1))
    for i in range(100):
        x = gfun(x)
        all_series[:, i] = x

    # Normalize the time series values to lie in range [0, 1]
    min_val, max_val = all_series.min(), all_series.max()
    all_series = (all_series - min_val) / (max_val - min_val)

    # Split into train, validation, and test sets, save to disk
    train = all_series[:8000]
    val = all_series[8000:9000]
    test = all_series[9000:]
    np.savez_compressed(save_file, train=train, val=val, test=test)

    return train, val, test

w = 0.5
c = 1.0  # I'm using the fact that c=1.0 to set c**2 = c**-2 = c
genz_funs = [
    None,  # Placeholder to give 1-based indexing
    lambda x: np.cos(2 * np.pi * w + c * x),
    lambda x: (c + (x + w)) ** -1,
    lambda x: (1 + c * x) ** -2,
    lambda x: np.exp(-c * np.pi * (x - w) ** 2),
    lambda x: np.exp(-c * np.pi * np.abs(x - w)),
    lambda x: np.where(x > w, 0, np.exp(c * x)),
]
