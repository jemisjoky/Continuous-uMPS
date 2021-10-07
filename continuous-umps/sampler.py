from math import sqrt

import torch
import numpy as np
import matplotlib.pyplot as plt


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
def sample(cores, item):

    given_vals = []

    if item == 1:
        Z = square_norm_leftQR(cores).item()
        probs, vals = margin(cores, given_vals)
        res = roll(probs / Z)
        return probs[res], [vals[res]]

    else:

        p_prec, vals_prec = sample(cores, item - 1)
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
        p, vals = sample(trans_cores, seq_len)
        print(p / Z)
        im = seq_to_array(vals, im_len, im_len)

        plt.imshow(im)
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


test_sampler("bd10_nb2_nt10k_gs.model")
