import torch
import matplotlib.pyplot as plt
from opt_einsum import contract as einsum


@torch.no_grad()
def sample(cores, edge_vecs, num_samples=1, embed_obj=None, generator=None):
    """
    Produce continuous or discrete samples from an MPS Born machine

    Args:
        cores: Collection of n core tensors, for n the number of pixels in the
            input to the MPS.
        edge_vecs: Pair of vectors giving the left and right boundary
            conditions of the MPS.
        num_samples: Number of samples to generate in parallel.
        embed_obj: Embedding object of type `torchmps.FixedEmbedding` which
            gives all needed info about (continuous or discrete) embedding.
        generator: Pytorch Generator object used to set randomness in sampling.

    Returns:
        samples: Tensor of shape (num_samples, n) containing the sample values.
            These will be nonnegative integers, or floats if the parent MPS is
            using an embedding of a continuous domain.
    """
    input_dim = cores[0].shape[0]
    assert len(edge_vecs) == 2
    left_vec, right_vec = edge_vecs

    # Get right PSD matrices resulting from tracing over right cores of MPS
    right_mats = right_trace_mats(cores, right_vec)
    assert len(right_mats) == len(cores)

    # Precompute cumulative embedding mats for non-trivial embedding
    if embed_obj is not None:
        domain = embed_obj.domain
        continuous = domain.continuous
        embed_fun = embed_obj.embed

        if continuous:
            num_points = embed_obj.num_points
            points = torch.linspace(
                domain.min_val, domain.max_val, steps=embed_obj.num_points
            )
            dx = (domain.max_val - domain.min_val) / (embed_obj.num_points - 1)
            emb_vecs = embed_fun(points)
            assert emb_vecs.shape[1] == input_dim

            # Get rank-1 matrices for each point, then numerically integrate
            emb_mats = einsum("bi,bj->bij", emb_vecs, emb_vecs.conj())
            int_mats = torch.cumsum(emb_mats, dim=0) * dx

        else:
            num_points = domain.max_val
            points = torch.arange(num_points).long()
            emb_vecs = embed_fun(points)
            assert emb_vecs.shape[1] == input_dim

            # Get rank-1 matrices for each point, then sum together
            emb_mats = einsum("bi,bj->bij", emb_vecs, emb_vecs.conj())
            int_mats = torch.cumsum(emb_mats, dim=0)
    else:
        continuous = False
        num_points = input_dim
        int_mats = None

    # Initialize conditional left PSD matrix and generate samples sequentially
    l_vecs = left_vec[None].expand(num_samples, -1)
    samples = []
    for core, r_mat in zip(cores, right_mats):
        samps, l_vecs = _sample_step(
            core, l_vecs, r_mat, embed_obj, int_mats, num_samples, generator
        )
        samples.append(samps)
    samples = torch.stack(samples, dim=1)

    # If needed, convert integer sample outcomes into continuous values
    if continuous:
        samples = points[samples]

    return samples


def _sample_step(core, l_vecs, r_mat, embed_obj, int_mats, num_samples, generator):
    """
    Function for generating single batch of samples
    """
    # Get unnormalized probabilities and normalize
    if embed_obj is not None:
        probs = einsum(
            "bl,bm,ilr,uij,jms,rs->bu",
            l_vecs,
            l_vecs.conj(),
            core,
            int_mats,
            core.conj(),
            r_mat,
        )
    else:
        probs = einsum(
            "bl,bm,ilr,ims,rs->bi", l_vecs, l_vecs.conj(), core, core.conj(), r_mat
        )
    if probs.is_complex():
        probs = probs.real
    probs /= probs.sum(dim=1, keepdim=True)
    int_probs = torch.cumsum(probs, axis=1)
    try:
        assert torch.all(probs > -1e-5)     # Tolerance for small negative values
    except AssertionError:
        breakpoint()
    assert torch.allclose(int_probs[:, -1], torch.ones(1))

    # Sample from int_probs (argmax finds first int_p with int_p > rand_val)
    rand_vals = torch.rand((num_samples, 1), generator=generator)
    samp_ints = torch.argmax((int_probs > rand_vals).long(), dim=1)

    # Conditionally update new left boundary vectors
    if embed_obj is not None:
        emb_vecs = embed_obj.embed(samp_ints)
        l_vecs = einsum("bl,ilr,bi->br", l_vecs, core, emb_vecs)
    else:
        samp_mats = core[samp_ints]
        l_vecs = einsum("bl,blr->br", l_vecs, samp_mats)
    # Rescale all vectors to have unit 2-norm
    l_vecs /= torch.norm(l_vecs, dim=1, keepdim=True)

    return samp_ints, l_vecs


def right_trace_mats(tensor_cores, right_vec):
    """
    Generate virtual PSD matrices from tracing over right cores of MPS

    Note that resultant PSD matrices are rescaled to avoid exponentially
    growing or shrinking trace.

    Args:
        tensor_cores: Collection of n core tensors, for n the number of pixels
            in the input to the MPS.
        right_vec: The vector giving the right boundary condition of the MPS.

    Returns:
        right_mats: Collection of n PSD matrices, ordered from left to right.
    """
    uniform_input = hasattr(tensor_cores, "shape")
    assert not uniform_input or tensor_cores.ndim == 4

    # Build up right matrices iteratively, from right to left
    r_mat = right_vec[:, None] @ right_vec[None].conj()
    right_mats = [r_mat]
    for core in tensor_cores.flip(0)[:-1]:  # Iterate backwards up to first core
        r_mat = einsum("ilr,ims,rs->lm", core, core.conj(), r_mat)
        # Stabilize norm
        r_mat /= torch.trace(r_mat)
        right_mats.append(r_mat)

    if uniform_input:
        right_mats = torch.stack(right_mats)

    return right_mats


if __name__ == "__main__":
    num_samps = 5
    # model_name = "trig_bd10_nb10_nt10k_gs.model"
    # model_name = "bd10_nb10_nt10k_gs.model"
    model_name = "bd10_nb2_nt10k_gs.model"
    mps = torch.load(f"models/{model_name}")

    samples = sample(
        mps.core_tensors, mps.edge_vecs, num_samples=num_samps, embed_obj=mps.embedding
    )
    
    # Reshape and rescale sampled values
    samples = samples.reshape(num_samps, 14, 14)
    assert torch.all(samples >= 0)
    if torch.any(samples >= 2):
        samples = samples.float() / samples.max()

    # Plot everything
    unseen = True
    for image in samples:
        if unseen:
            unseen = False
            print(image)
        # plt.imshow(image, cmap="gray")
        plt.imshow(image, cmap="gray_r")
        plt.show()
