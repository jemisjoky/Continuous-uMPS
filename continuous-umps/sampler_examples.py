from math import sqrt

import torch
import matplotlib.pyplot as plt

from .sampler import sample, test_sampler


def print_pretrained_samples():
    num_bins = 2
    num_samps = 5
    bond_dim = 30
    # model_name = f"bd100_nb{num_bins}_nt10k_gs.model"
    # model_name = f"leg_bd10_nb{num_bins}_nt10k_gs.model"
    # model_name = f"trig_bd10_nb{num_bins}_nt10k_gs.model"
    model_name = f"bd{bond_dim}_gs_adam_E3.0.model"
    # model_name = f"bd8_gs.model"
    mps = torch.load(f"models/{model_name}")

    samples = sample(
        mps.core_tensors, mps.edge_vecs, num_samples=num_samps, embed_obj=mps.embedding
    )

    # Reshape and rescale sampled values
    width = round(sqrt(mps.core_tensors.shape[0]))
    samples = samples.reshape(num_samps, width, width)
    assert torch.all(samples >= 0)
    if torch.any(samples >= 2):
        samples = samples.float() / samples.max()

    # Plot everything with new sampler
    unseen = True
    for image in samples:
        if unseen:
            unseen = False
            # print(image)
        plt.imshow(image, cmap="gray_r", vmin=0, vmax=1)
        plt.show()

    if model_name[:5] != "trig_" and "_nb2_" in model_name:
        test_sampler(model_name)


def print_canned_samples():
    num_bins = 10
    num_samps = 3
    # model_name = f"bd10_nb{num_bins}_nt10k_gs.model"
    model_name = f"leg_bd10_nb{num_bins}_nt10k_gs.model"
    # model_name = f"trig_bd10_nb{num_bins}_nt10k_gs.model"
    mps = torch.load(f"models/{model_name}")
    embed_obj = mps.embedding

    # ALL BLACK
    if model_name[:4] != "leg_":
        core_tensors = torch.cat(
            [torch.ones(14 ** 2, 1, 1, 1), torch.zeros(14 ** 2, num_bins - 1, 1, 1)],
            dim=1,
        )
    else:
        core_tensors = torch.ones(14 ** 2, num_bins, 1, 1)
        core_tensors[:, 1::2] = -1
    edge_vecs = torch.ones(2, 1)

    # # ALL WHITE
    # if model_name[:4] != "leg_":
    #     core_tensors = torch.cat([torch.zeros(14 ** 2, num_bins - 1, 1, 1), torch.ones(14 ** 2, 1, 1, 1)], dim=1)
    # else:
    #     core_tensors = torch.ones(14 ** 2, num_bins, 1, 1)
    # edge_vecs = torch.ones(2, 1)

    # # VERTICAL BARS PATTERN
    # even_tensor = torch.cat(
    #     [torch.zeros(1, num_bins - 1, 1, 1), torch.ones(1, 1, 1, 1)], dim=1
    # )
    # odd_tensor = torch.cat(
    #     [torch.ones(1, 1, 1, 1), torch.zeros(1, num_bins - 1, 1, 1)], dim=1
    # )
    # core_tensors = torch.cat([even_tensor, odd_tensor] * (14 * 7))
    # edge_vecs = torch.ones(2, 1)

    # # SINGLE WHITE PIXEL
    # core_tensors = torch.tensor([[[0, 1], [0, 0]], [[1, 0], [0, 1]]])[None].expand(14 ** 2, 2, -1, -1).float()
    # assert core_tensors.shape == (14 ** 2, 2, 2, 2)
    # edge_vecs = torch.tensor([[1, 0], [0, 1]]).float()

    samples = sample(
        core_tensors, edge_vecs, num_samples=num_samps, embed_obj=embed_obj
    )

    # Reshape and rescale sampled values
    samples = samples.reshape(num_samps, 14, 14)
    assert torch.all(samples >= 0)
    if torch.any(samples >= 2):
        samples = samples.float() / samples.max()

    # Plot everything with new sampler
    unseen = True
    for image in samples:
        if unseen:
            unseen = False
            print(image)
        print(image.mean())
        plt.imshow(image, cmap="gray", vmin=0, vmax=1)
        plt.show()


if __name__ == "__main__":
    print_pretrained_samples()
    # print_canned_samples()
