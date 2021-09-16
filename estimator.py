"""Probabilistic MPS written as a sklearn estimator"""
from time import time
from math import ceil, log

import numpy as np
from sklearn.base import BaseEstimator, DensityMixin


class ProbMPS_Estimator(BaseEstimator, DensityMixin):
    """
    Wraps ProbMPS model to give scikit-learn API, simplifying model training

    Args:
        input_dim: Dimension of the MPS input indices
        num_inputs: Number of discrete inputs to the model
        bond_dim: Bond dimension of the model
    """

    def __init__(
        self,
        input_dim=4,
        seq_len=3,
        bond_dim=2,
        complex_params=False,
        use_bias=False,
        embed_spec=None,
        domain_spec=None,
        dataset="mnist",
        dataset_dir="./datasets/",
        apply_downscale=True,
        downscale_shape=(14, 14),
        comet_log=False,
        comet_args={},
        logging_dir="./logs/",
        core_init_spec="normal",
        optimizer="Adam",
        weight_decay=0.0001,
        momentum=0.0,
        constant_lr=False,
        learning_rate_init=0.001,
        learning_rate_final=1e-6,
        early_stopping=False,
        factor=0.1,
        patience=6,
        cooldown=2,
        max_epochs=100,
        batch_size=128,
        num_train=None,
        num_test=None,
        num_val=None,
        shuffle=True,
        verbose=True,
        seed=0,
    ):
        self.input_dim = input_dim
        self.seq_len = seq_len
        self.bond_dim = bond_dim
        self.complex_params = complex_params
        self.use_bias = use_bias
        self.embed_spec = embed_spec
        self.domain_spec = domain_spec
        self.dataset = dataset
        self.dataset_dir = dataset_dir
        self.apply_downscale = apply_downscale
        self.downscale_shape = downscale_shape
        self.comet_log = comet_log
        self.comet_args = comet_args
        self.logging_dir = logging_dir
        self.core_init_spec = core_init_spec
        self.optimizer = optimizer
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.constant_lr = constant_lr
        self.learning_rate_init = learning_rate_init
        self.learning_rate_final = learning_rate_final
        self.early_stopping = early_stopping
        self.factor = factor
        self.patience = patience
        self.cooldown = cooldown
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.num_train = num_train
        self.num_test = num_test
        self.num_val = num_val
        self.shuffle = shuffle
        self.verbose = verbose
        self.seed = seed

    def fit(self, X, y=None):
        """
        Trains a probabilistic MPS model on a specified dataset
        """
        if self.comet_log:
            from comet_ml import Experiment
        global torch
        import torch
        import torchvision
        import torch.nn as nn

        from torchmps import ProbMPS
        from torchmps.embeddings import DataDomain

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # TODO: Initialize embedding, using embed_spec and domain_spec

        # Initialize model
        my_mps = ProbMPS(
            self.seq_len,
            self.input_dim,
            self.bond_dim,
            complex_params=self.complex_params,
            use_bias=self.use_bias,
            # embed_fun=embedding,
            # domain=embDomain,
        )
        my_mps.to(device)

        # Initialize optimizer and LR scheduler
        optimizer, scheduler = setup_opt_sched(
            my_mps.parameters(),
            self.optimizer,
            self.weight_decay,
            self.momentum,
            self.constant_lr,
            self.learning_rate_init,
            self.learning_rate_final,
            self.early_stopping,
            self.factor,
            self.patience,
            self.cooldown,
            self.max_epochs,
            self.verbose,
        )

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


# trainX, testX = load_dataset()


def getTrainTest(trainX, testX, pool=False, discrete=True, d=2):
    trainX = processMnist(trainX, pool=pool, discrete=discrete)
    testX = processMnist(testX, pool=pool, discrete=discrete)
    return trainX, testX


def discreteEmbedding(input):
    input = input.cpu()
    d1 = torch.tensor(np.ones(input.shape) * (input.numpy() > 0.2))
    d2 = torch.tensor(np.ones(input.shape) * (input.numpy() < 0.2))
    ret = torch.stack([d1, d2], dim=-1)
    return ret.float()


# discreteDomain = DataDomain(False, 2)


def sincosEmbedding(input):
    input = input.cpu()
    d1 = np.cos(input * np.pi / 2)
    d2 = np.sin(input * np.pi / 2)
    ret = torch.stack([d1, d2], dim=-1)
    return ret.float()


# sincosDomain = DataDomain(True, 1, 0)


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


# plot(
#     trainX[:10],
#     testX[:10],
#     hyper_params["epochs"],
#     hyper_params["sequence_length"],
#     hyper_params["bond_dim"],
#     hyper_params["batch_size"],
#     sincosEmbedding,
#     sincosDomain,
#     lr=hyper_params["lr_init"],
# )


def setup_opt_sched(
    parameters,
    optimizer,
    weight_decay,
    momentum,
    constant_lr,
    learning_rate_init,
    learning_rate_final,
    early_stopping,
    factor,
    patience,
    cooldown,
    max_epochs,
    verbose,
):
    assert hasattr(torch.optim, optimizer)

    # Define optimizer
    if optimizer in ["Adam", "AdamW", "Adamax", "Adadelta", "Adagrad"]:
        opt = getattr(torch.optim, optimizer)(
            parameters, lr=learning_rate_init, weight_decay=weight_decay
        )
    elif optimizer in ["SGD", "RMSprop"]:
        opt = getattr(torch.optim, optimizer)(
            parameters,
            lr=learning_rate_init,
            weight_decay=weight_decay,
            momentum=momentum,
        )
    else:
        raise NotImplementedError

    # Define (tweaked) scheduler, whose step method outputs early stopping info
    if constant_lr:
        factor = 1.0
        max_reductions = float("inf")
    else:
        max_reductions = ceil(
            log(learning_rate_final / learning_rate_init) / log(factor)
        )
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, factor=factor, patience=patience, cooldown=cooldown, verbose=verbose
    )
    sched.num_reductions = 0
    sched.__reduce_lr = sched._reduce_lr
    sched._step = sched.step

    def custom_reduce_lr(self, epoch):
        self.num_reductions += 1
        self.__reduce_lr(self, epoch)

    def custom_step(self, metrics, epoch=None):
        self._step(metrics, epoch)
        if not early_stopping:
            return False
        if self.num_reductions >= 1 if constant_lr else max_reductions:
            return True
        return False

    sched.step = custom_step
    sched._reduce_lr = custom_reduce_lr

    return opt, sched


if __name__ == "__main__":
    estimator = ProbMPS_Estimator()
    estimator.fit(None)
