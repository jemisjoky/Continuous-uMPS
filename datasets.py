"""Dataset loaders for MNIST and fashion MNIST"""
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
    if num_bins is not None:
        out = tuple(bin_data(ds, num_bins).long() for ds in out)
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

    return out_data
