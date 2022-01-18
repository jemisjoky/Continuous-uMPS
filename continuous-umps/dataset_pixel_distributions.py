import torch
import matplotlib.pyplot as plt

from .datasets import load_mnist


if __name__ == "__main__":
    for fashion in [False, True]:
        name = ("Fashion" if fashion else "") + "MNIST"
        train, _ = load_mnist(
            fashion=fashion,
            num_train=60000,
            downscale=False,
        )
        train = train.tensors[0].reshape(-1).numpy()
        plt.hist(train, bins=100, density=True)
        plt.title(f"{name} Pixel Distribution")
        plt.show()