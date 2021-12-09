from math import sqrt

import torch
import numpy as np
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
        embed_fun = embed_obj.__call__

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
        points = None

    # Initialize conditional left PSD matrix and generate samples sequentially
    l_vecs = left_vec[None].expand(num_samples, -1)
    samples = []
    for core, r_mat in zip(cores, right_mats):
        samps, l_vecs = _sample_step(
            core, l_vecs, r_mat, embed_obj, int_mats, num_samples, points, generator
        )
        samples.append(samps)
    samples = torch.stack(samples, dim=1)

    # If needed, convert integer sample outcomes into continuous values
    if continuous:
        samples = points[samples]
    elif input_dim != 2:
        samples = samples.float() / (input_dim - 1)

    return samples


def _sample_step(
    core, l_vecs, r_mat, embed_obj, int_mats, num_samples, points, generator
):
    """
    Function for generating single batch of samples
    """
    # Get unnormalized probabilities and normalize
    if embed_obj is not None:
        int_probs = einsum(
            "bl,bm,ilr,uij,jms,rs->bu",
            l_vecs,
            l_vecs.conj(),
            core,
            int_mats,
            core.conj(),
            r_mat,
        )
        int_probs /= int_probs[:, -1][:, None]
    else:
        probs = einsum(
            "bl,bm,ilr,ims,rs->bi", l_vecs, l_vecs.conj(), core, core.conj(), r_mat
        )
        probs /= probs.sum(dim=1, keepdim=True)
        int_probs = torch.cumsum(probs, axis=1)

    if int_probs.is_complex():
        int_probs = int_probs.real
    try:
        assert torch.all(int_probs >= 0)  # Tolerance for small negative values
    except AssertionError:
        print(int_probs)
        print("WARNING: Some negative probabilities found")
    assert torch.allclose(int_probs[:, -1], torch.ones(1))

    # Sample from int_probs (argmax finds first int_p with int_p > rand_val)
    rand_vals = torch.rand((num_samples, 1), generator=generator)
    samp_ints = torch.argmax((int_probs > rand_vals).long(), dim=1)

    # Conditionally update new left boundary vectors
    if embed_obj is not None:
        samp_points = points[samp_ints]
        emb_vecs = embed_obj(samp_points)
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
    for core in tensor_cores.flip(dims=[0])[:-1]:  # Iterate backwards up to first core
        r_mat = einsum("ilr,ims,rs->lm", core, core.conj(), r_mat)
        # Stabilize norm
        r_mat /= torch.trace(r_mat)
        right_mats.append(r_mat)

    if uniform_input:
        right_mats = torch.stack(right_mats[::-1])

    return right_mats


### OLD SAMPLER CODE ###

# performs a contraction of a tensor train with given input values
def contractTrain(cores, values):

    res = cores[0]
    val = values[0]
    res = np.einsum("ijk,i->jk", res, val)

    for i in range(1, len(cores)):

        res = np.einsum("ik,jkm -> jim", res, cores[i])
        res = np.einsum("ijk,i->jk", res, values[i])

    return res


# return the canonical form resulting from the QR transform of the tensor train
def get_QR_transform(cores):

    newCores = []
    R = np.identity(cores[0].shape[1])

    for i in range(0, len(cores) - 1):

        A = cores[i]
        newA = np.einsum("ij,kjl->kil", R, A)
        temp = newA.reshape(-1, newA.shape[-1])
        Q, R = np.linalg.qr(temp)
        Q = Q.reshape(newA.shape[0], newA.shape[1], R.shape[0])
        newCores.append(Q)

    newA = newA = np.einsum("ij,kjl->kil", R, cores[-1])

    newCores.append(newA)
    return newCores


# efficient computation of the square norm assuming a left canonical form of
# the tensor train
def square_norm_leftQR(cores):
    A = cores[-1]
    res = np.einsum("ijk,ijm->km", A, A.conj())
    return res


# returns the unormalised probabilities of the input sequence given by values
def get_quasi_prob(cores, values):
    use_cores = cores[-len(values) :]
    temp = contractTrain(use_cores, values)
    temp = np.einsum("ij,ik->jk", temp, temp.conj())
    return temp


# compute the marginalised conditional probabilities associated to the possible
# next values in the chain given_val are the values currently known, this
# function computes the marginalisation for the next possible value
def margin(cores, given_val):

    dim = cores[0].shape[0]
    probs = np.zeros(dim)

    possible_values = []

    for i in range(dim):
        curr_val = np.zeros(dim)
        curr_val[i] = 1
        vals = [curr_val]
        vals += given_val
        probs[i] = get_quasi_prob(cores, vals)
        possible_values.append(curr_val)

    return probs, possible_values


# sampling recursion generating a new sample from the distribution learned in
# the tensor train
def old_sample(cores, item):

    given_vals = []

    if item == 1:
        Z = square_norm_leftQR(cores).item()
        probs, vals = margin(cores, given_vals)
        res = roll(probs / Z)
        return probs[res], [vals[res]]

    else:

        p_prec, vals_prec = old_sample(cores, item - 1)
        probs, vals = margin(cores, vals_prec)
        res = roll(probs / p_prec)
        given_vals = [vals[res]]
        given_vals += vals_prec
        return probs[res], given_vals


def seq_to_array(seq, w, h):
    seq = np.array(seq)
    seq = seq[:, 0]
    return seq.reshape(w, h)


# code testing the sampling code by generating and showing 10 images.
@torch.no_grad()
def test_sampler(model_name, save_dir="./models/"):
    assert model_name[-6:] == ".model"
    try:
        my_mps = torch.load(save_dir + model_name)
    except FileNotFoundError:
        raise FileNotFoundError(f"No model file '{model_name}' in {save_dir}")

    seq_len, input_dim, bond_dim = my_mps.core_tensors.shape[:3]
    cores = list(my_mps.core_tensors.numpy())
    edges = my_mps.edge_vecs.numpy()
    # bond_dim = 1
    # cores = list(torch.cat([torch.ones(14 ** 2, 1, 1, 1), torch.zeros(14 ** 2, 1, 1, 1)], dim=1))
    # edges = torch.ones(2, 1)
    assert edges.shape[1] == bond_dim

    firstCore = np.einsum("ijk,j->ik", cores[0], edges[0])
    cores[0] = firstCore.reshape(input_dim, 1, bond_dim)

    endCore = np.einsum("ijk,k->ij", cores[-1], edges[1])
    cores[-1] = endCore.reshape(input_dim, bond_dim, 1)

    trans_cores = get_QR_transform(cores)
    Z = square_norm_leftQR(trans_cores).item()

    # Infer size of images, which assumes square images
    im_len = round(sqrt(seq_len))
    assert seq_len == im_len ** 2

    for i in range(5):
        p, vals = old_sample(trans_cores, seq_len)
        print(p / Z)
        im = seq_to_array(vals, im_len, im_len)

        plt.imshow(im, cmap="gray", vmin=0, vmax=1)
        plt.show()


# utilitary function for the sampling
def roll(bias):

    guess = np.random.uniform()
    S = 0
    for i, val in enumerate(bias):
        if guess < S + val:
            return i
        else:
            S += val
