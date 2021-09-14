"""Dataset loaders for MNIST and fashion MNIST"""
import numpy as np
import torchvision


def _load_mnist(
    fashion=False,
    num_train=60000,
    num_test=10000,
    num_val=None,
    downscale=False,
    downscale_shape=None,
    num_bins=None,
    dataset_dir=None,
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

    # Build the desired transform for MNIST images
    tf = torchvision.transforms
    transform = tf.Compose([tf.ToTensor()])
    if downscale:
        transform.insert(0, tf.Resize(downscale_shape))

    # Get the desired number of flattened images
    train = dataset(root=dset_dir, train=True, download=True, transform=transform)
    test = dataset(root=dset_dir, train=False, download=True, transform=transform)
    train_data = train.data[:num_train].reshape(num_train, -1)
    test_data = test.data[:num_test].reshape(num_test, -1)
    if num_val is None:
        out = (train_data, test_data)
    else:
        val_data = train.data[-num_val:].reshape(num_val, -1)
        out = (train_data, val_data, test_data)

    # Finally, discretize images
    out = tuple(bin_data(ds, num_bins) for ds in out)
    return out


def bin_data(input, num_bins=None):
    """
    Discretize greyscale values into a finite number of bins
    """
    if num_bins is None:
        return input
    assert num_bins > 0

    # Set each of the corresponding bin indices
    out_data = torch.full_like(input, -1)
    for i in num_bins:
        bin_inds = i / num_bins <= input <= (i + 1) / num_bins
        out_data[bin_inds] = i
    assert out_data.max() >= 0
    
    return out_data