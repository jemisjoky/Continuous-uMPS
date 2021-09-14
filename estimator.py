"""Probabilistic MPS written as a sklearn estimator"""
from time import time

import numpy as np

# import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, DensityMixin

if exp_params["comet_log"]:
    from comet_ml import Experiment
import torch
import torchvision
import torch.nn as nn

from torchmps import ProbMPS
from torchmps.embeddings import DataDomain

# Boilerplate config for the experiment
dataset_dir = "./datasets/"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# torch.set_default_tensor_type('torch.cuda.FloatTensor') #appropriate for parallel gpu running

# Parameters for the experiment
# TODO: Move these into the corresponding functions
exp_params = {
    "fashion_mnist": False,
    "learn_rate": 1e-3,
    "comet_log": False,
    "downscale_image": True,
    "downscale_shape": (14, 14),
}


def load_dataset(
    fashion=False,
    num_train=60000,
    num_test=10000,
    num_val=None,
    downscale=False,
    downscale_shape=None,
    num_bins=None,
):
    """
    Function for loading unlabeled MNIST and FashionMNIST datasets

    Supports downscaling, discretizing images via binning, and splitting into
    extra validation set (pulled from training data)
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
    test_data = train.data[:num_test].reshape(num_test, -1)
    if num_val is None:
        out = (train_data, test_data)
    else:
        val_data = train.data[-num_val:].reshape(num_val, -1)
        out = (train_data, val_data, test_data)

    # Finally, discretize images
    if num_bins is not None:
        out = tuple(bin_data(ds, num_bins) for ds in out)
    return out


def bin_data(data, num_bins=None):
    """
    Discretize greyscale values into a finite number of bins
    """
    if num_bins is None:
        return data

    raise NotImplementedError


def discreteEmbedding(input):
    input = input.cpu()
    d1 = torch.tensor(np.ones(input.shape) * (input.numpy() > 0.2))
    d2 = torch.tensor(np.ones(input.shape) * (input.numpy() < 0.2))
    ret = torch.stack([d1, d2], dim=-1)
    return ret.float()


discreteDomain = DataDomain(False, 2)


def sincosEmbedding(input):
    input = input.cpu()
    d1 = np.cos(input * np.pi / 2)
    d2 = np.sin(input * np.pi / 2)
    ret = torch.stack([d1, d2], dim=-1)
    return ret.float()


sincosDomain = DataDomain(True, 1, 0)


def embeddingFunc(vect, s, d):
    emb = np.cos(vect * np.pi / 2) ** (d - s) * np.sin(vect * np.pi / 2) ** (s - 1)
    # print(emb.shape, vect.shape)
    return emb


def embedding(data, d):

    newEmbed = np.zeros([len(data), len(data[0]), d])
    for s in range(d):
        # print(newEmbed[:,:, s].shape)
        newEmbed[:, :, s] = embeddingFunc(data.cpu(), s + 1, d)

    return newEmbed


# utilitary function for the sampling
def roll(bias):

    guess = np.random.uniform()
    S = 0
    for i, val in enumerate(bias):
        if guess < S + val:
            return i
        else:
            S += val


# create a toy data set of size 4x4 where only one 2x2 corner has random values
def toyData(N):

    data = torch.zeros([N, 4, 4])

    for i in range(N):

        quarter = np.random.random([2, 2]) > 0.5
        bias = [0.333, 0.667]
        l = roll(bias)
        h = roll(bias)
        data[i, l * 2 : (l + 1) * 2, h * 2 : (h + 1) * 2] = torch.tensor(quarter)

    return data.long()


# main function handling the learning and the logging of the errors
def compute(
    trainX,
    testX,
    epochs,
    seq_len,
    bond_dim,
    batch_size,
    embedding,
    embDomain,
    lr=0.001,
    hist=False,
):

    # bond_dim = 10
    input_dim = 2
    # batch_size = 100
    sequence_len = seq_len
    complex_params = False

    # epochs=1
    test_loss_hist = []
    train_loss_hist = []

    my_mps = ProbMPS(
        sequence_len,
        input_dim,
        bond_dim,
        complex_params,
        embed_fun=embedding,
        domain=embDomain,
    )
    my_mps.to(device)
    optimizer = torch.optim.SGD(my_mps.parameters(), lr=lr)

    data = trainX

    totalB = int(len(data) / batch_size)
    print(totalB)
    # data=data.transpose(0, 1)

    test_data = testX

    def testLoop(dataTest):

        totalBT = int(len(dataTest) / batch_size)
        testLoss = 0
        for j in range(totalBT):

            # print("test       ", j)
            batchTest = dataTest[
                j * batch_size : min((j + 1) * batch_size, len(dataTest))
            ]
            toLearn = batchTest.transpose(1, 0)
            testLoss += my_mps.loss(toLearn).detach().item() * len(batchTest)

        testLoss = testLoss / len(dataTest)

        return testLoss

    prevLoss = -np.log(len(data))

    for e in range(epochs):

        order = np.array(range(len(trainX)))
        np.random.shuffle(order)
        trainX[np.array(range(len(trainX)))] = trainX[order]

        if hist:
            testl = testLoop(test_data)
            trainl = testLoop(data)
            test_loss_hist.append(testl)
            train_loss_hist.append(trainl)

        #             with experiment.train():
        #                 experiment.log_metric("logLikelihood", trainl, step=e)
        #             with experiment.test():
        #                 experiment.log_metric("logLikelihood", testl, step=e)

        print(e)

        # adapatation of the learning rate
        # if e>5:
        #     optimizer = torch.optim.Adam(my_mps.parameters(), lr=lr/10)

        if e % 5 == 0:  # progressive reduction of the learning rate
            reduction = 1.61 ** (e / 5)
            optimizer = torch.optim.SGD(my_mps.parameters(), lr=lr / reduction)

        for j in range(totalB):

            batchData = data[j * batch_size : min((j + 1) * batch_size, len(data))]
            batchData = batchData.transpose(1, 0)

            loss = my_mps.loss(batchData)  # <- Negative log likelihood loss

            loss.backward()
            optimizer.step()

    return my_mps, test_loss_hist, train_loss_hist


# performs a contraction of a tensor train with given input values
def contractTrain(cores, values):

    res = cores[0]
    val = values[0]
    res = np.einsum("ijk, i->jk", res, val)

    for i in range(1, len(cores)):

        res = np.einsum("ik, jkm -> jim", res, cores[i])
        res = np.einsum("ijk, i->jk", res, values[i])

    return res


# constract the tensor train resulting from taking the square norm of the train, assuming that the input maps to a frame
def contractSquareNorm(cores):

    res = np.einsum("ijk, imn ->jmkn", cores[0], cores[0])
    for i in range(1, len(cores)):

        temp = np.einsum("ijk, imn->jmkn", cores[i], cores[i])
        res = np.einsum("klij, ijmn->klmn", res, temp)

    return res


# return the canonical form resulting from the QR transform of the tensor train
def get_QR_transform(cores):

    newCores = []
    R = np.identity(cores[0].shape[1])

    for i in range(0, len(cores) - 1):

        A = cores[i]
        newA = np.einsum("ij, kjl->kil", R, A)
        temp = newA.reshape(-1, newA.shape[-1])
        Q, R = np.linalg.qr(temp)
        Q = Q.reshape(newA.shape[0], newA.shape[1], R.shape[0])
        newCores.append(Q)

    newA = newA = np.einsum("ij, kjl->kil", R, cores[-1])

    newCores.append(newA)
    return newCores


# efficient computation of the square norm assuming a left canonical form of the tensor train
def square_norm_leftQR(cores):
    A = cores[-1]
    res = np.einsum("ijk,ijm->km", A, A)
    return res


# returns the unormalised probabilities of the input sequence given by values
def get_quasi_prob(cores, values):
    use_cores = cores[-len(values) :]
    temp = contractTrain(use_cores, values)
    temp = np.einsum("ij, ik->jk", temp, temp)
    return temp


# compute the marginalised conditional probabilities associated to the possible next values in the chain
# given_val are the values currently known, this function computes the marginalisation for the next possible
# value
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


# sampling recursion generating a new sample from the distribution learned in the tensor train
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
def test(trainX, testX, epochs, bond_dim, seq_len, sizes):

    my_mps, t1, t2 = compute(trainX, testX, epochs, bond_dim, 10, seq_len)
    cores = []

    for i in range(len(my_mps.core_tensors)):
        cores.append(my_mps.core_tensors[i].detach().numpy())

    edge = my_mps.edge_vecs.detach().numpy()

    firstCore = np.einsum("ijk, j->ik", cores[0], edge[0])
    firstCore = firstCore.reshape(2, 1, bond_dim)
    cores[0] = firstCore

    endCore = np.einsum("ijk, j->ik", cores[-1], edge[1])
    endCore = endCore.reshape(2, bond_dim, 1)
    cores[-1] = endCore

    trans_cores = get_QR_transform(cores)
    Z = square_norm_leftQR(trans_cores).item()

    for i in range(10):
        p, vals = sample(trans_cores, seq_len)
        print(p / Z)
        im = seq_to_array(vals, sizes, sizes)

        plt.imshow(im)
        plt.show()


# code computing the training and test error and saving the resulting graph
def plot(
    trainX, testX, epochs, seq_len, bond_dim, batch_size, embedding, embDomain, lr
):
    # print(trainX.shape)

    my_mps, test_hist, train_hist = compute(
        trainX,
        testX,
        epochs,
        seq_len,
        bond_dim,
        batch_size,
        embedding,
        embDomain,
        lr=lr,
        hist=True,
    )
    x = np.arange(epochs)
    plt.plot(x, test_hist, "m")
    plt.plot(x, train_hist, "b")
    plt.savefig("graph1")


trainX, testX = load_dataset()


def getTrainTest(trainX, testX, pool=False, discrete=True, d=2):
    trainX = processMnist(trainX, pool=pool, discrete=discrete)
    testX = processMnist(testX, pool=pool, discrete=discrete)
    return trainX, testX


trainX, testX = getTrainTest(trainX, testX, pool=True, discrete=False)

# Hyper parameter dictionary
hyper_params = {
    "bond_dim": 10,
    "sequence_length": len(trainX[0]),
    "input_dim": 2,
    "batch_size": 1,
    "epochs": 10,
    "lr_init": 0.001,
}


plot(
    trainX[:10],
    testX[:10],
    hyper_params["epochs"],
    hyper_params["sequence_length"],
    hyper_params["bond_dim"],
    hyper_params["batch_size"],
    sincosEmbedding,
    sincosDomain,
    lr=hyper_params["lr_init"],
)
