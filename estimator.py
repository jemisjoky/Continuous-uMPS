"""Probabilistic MPS written as a sklearn estimator"""
from time import time
from copy import deepcopy
from types import MethodType
from math import ceil, log, prod

import numpy as np
from sklearn.base import BaseEstimator, DensityMixin

from utils import FakeLogger, null


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
        input_dim=2,
        bond_dim=1,
        complex_params=False,
        use_bias=False,
        embed_spec=None,
        domain_spec=None,
        num_bins=None,
        dataset="mnist",
        dataset_dir="./datasets/",
        apply_downscale=True,
        downscale_shape=(14, 14),
        comet_log=False,
        comet_args={},
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
        slim_eval=False,
        parallel_eval=False,
        max_epochs=100,
        batch_size=128,
        num_train=None,
        num_test=None,
        num_val=None,
        shuffle=True,
        verbose=True,
        save_model=False,
        model_dir="./models/",
        seed=0,
    ):
        self.input_dim = input_dim
        self.bond_dim = bond_dim
        self.complex_params = complex_params
        self.use_bias = use_bias
        self.embed_spec = embed_spec
        self.domain_spec = domain_spec
        self.num_bins = num_bins
        self.dataset = dataset
        self.dataset_dir = dataset_dir
        self.apply_downscale = apply_downscale
        self.downscale_shape = downscale_shape
        self.comet_log = comet_log
        self.comet_args = comet_args
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
        self.slim_eval = slim_eval
        self.parallel_eval = parallel_eval
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.num_train = num_train
        self.num_test = num_test
        self.num_val = num_val
        self.shuffle = shuffle
        self.verbose = verbose
        self.save_model = save_model
        self.model_dir = model_dir
        self.seed = seed

    def fit(self, X=None, y=None):
        """
        Trains a probabilistic MPS model on a specified dataset
        """
        now = time()
        if self.comet_log:
            from comet_ml import Experiment
        global torch
        import torch

        from torchmps import ProbMPS
        from torchmps.embeddings import DataDomain

        # Conditional print function
        global cprint
        cprint = print if self.verbose else null

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        train_g = torch.Generator(device=device)
        train_g.manual_seed(self.seed)
        self.val_g = torch.Generator(device=device)
        self.val_g.manual_seed(self.seed)

        # Import our dataset
        self.dataset = self.dataset.lower()
        if self.dataset in ["mnist", "fashion_mnist"]:
            from datasets import load_mnist

            fashion = self.dataset == "fashion_mnist"
            train, val, test = load_mnist(
                fashion=fashion,
                num_train=self.num_train,
                num_test=self.num_test,
                num_val=self.num_val,
                downscale=self.apply_downscale,
                downscale_shape=self.downscale_shape,
                num_bins=self.num_bins,
                dataset_dir=self.dataset_dir,
                device=device,
            )
            self.seq_len = (
                prod(self.downscale_shape) if self.apply_downscale else 28 ** 2
            )
        else:
            raise NotImplementedError

        # TODO: Initialize embedding, using embed_spec and domain_spec

        # Initialize model
        self.model = ProbMPS(
            self.seq_len,
            self.input_dim,
            self.bond_dim,
            complex_params=self.complex_params,
            use_bias=self.use_bias,
            # embed_fun=embedding,
            # domain=embDomain,
        )
        self.model.to(device)

        # Initialize optimizer and LR scheduler
        optimizer, scheduler = setup_opt_sched(
            self.model.parameters(),
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

        # Initialize Comet data logging
        if self.comet_log:
            # TODO: Make experiment name
            exp_name = ""
            logger = Experiment(exp_name)
        else:
            logger = FakeLogger()

        def optimize_one_epoch():
            """Carries out one epoch of gradient updates over training data"""
            train_loss = 0.0
            for num, batch in enumerate(
                torch.utils.data.DataLoader(
                    train,
                    batch_size=self.batch_size,
                    shuffle=self.shuffle,
                    generator=train_g,
                ),
                start=1,
            ):
                optimizer.zero_grad()
                loss = self.model.loss(
                    *batch, slim_eval=self.slim_eval, parallel_eval=self.parallel_eval
                )
                train_loss += loss.detach()

                loss.backward()
                optimizer.step()

            # Return average loss
            train_loss /= num
            return train_loss

        cprint(f"Initialization time: {time() - now:.2f}s")
        cprint("Starting training...\n")
        now = time()

        # Run the actual training loop
        try:
            for epoch in range(self.max_epochs):
                cprint(f"Epoch {epoch}")

                # Optimize model parameters
                train_loss = optimize_one_epoch()
                cprint(f"  Train loss: {train_loss:.2f}")

                # Evaluate on the validation set
                val_loss = -self.score(val)
                cprint(f"  Val loss:   {val_loss:.2f}")

                # Log data, check for best val loss
                logger.log_metrics({"train_loss": train_loss, "val_loss": val_loss})
                self._check_best(val_loss)
                cprint(f"  Epoch time: {time() - now:.2f}s")
                now = time()

                # Check for early stopping
                if scheduler.step(val_loss):
                    cprint("\nNon-improving val loss, stopping training early")
                    break

        except KeyboardInterrupt:
            cprint("\nTraining stopped early by user")

        # Evaluate test loss using final model, and all losses using best model
        test_loss = -self.score(test)
        self.model.load_state_dict(self.best_model)
        best_train, best_val, best_test = [-self.score(ds) for ds in [train, val, test]]

        cprint(f"\nTrain loss at best: {best_train:.2f}")
        cprint(f"Val loss at best:   {best_val:.2f}")
        cprint(f"Test loss at best:  {best_test:.2f}")
        cprint(f"Test loss at end:   {test_loss:.2f}")
        logger.log_metrics(
            {
                "test_loss": test_loss,
                "best_train": best_train,
                "best_val": best_val,
                "best_test": best_test,
            }
        )

        # Save model to disk if desired
        if self.save_model:
            torch.save(self.model_dir + exp_name + ".model")

        return self

    def score(self, X, y=None):
        """
        Return the average log likelihood of the input dataset
        """
        assert isinstance(X, torch.utils.data.Dataset)
        val_loss = 0.0
        for num, batch in enumerate(
            torch.utils.data.DataLoader(
                X,
                batch_size=self.batch_size,
                shuffle=self.shuffle,
                generator=self.val_g,
            ),
            start=1,
        ):
            with torch.no_grad():
                loss = self.model.loss(
                    *batch, slim_eval=self.slim_eval, parallel_eval=self.parallel_eval
                )
            val_loss += loss.detach()

        # Return average loss, negative to fit with sklearn convention
        return -val_loss / num

    def _check_best(self, val_loss):
        """
        Checks if the current validation loss is best, saves model
        """
        best = val_loss < self.best_loss if hasattr(self, "best_loss") else True
        if best:
            self.best_loss = val_loss
            self.best_model = deepcopy(self.model.state_dict())


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
        cprint(p / Z)
        im = seq_to_array(vals, sizes, sizes)

        plt.imshow(im)
        plt.show()


# code computing the training and test error and saving the resulting graph
def plot(
    trainX, testX, epochs, seq_len, bond_dim, batch_size, embedding, embDomain, lr
):
    # cprint(trainX.shape)

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
    # cprint(emb.shape, vect.shape)
    return emb


def embedding(data, d):

    newEmbed = np.zeros([len(data), len(data[0]), d])
    for s in range(d):
        # cprint(newEmbed[:,:, s].shape)
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

    sched.step = MethodType(custom_step, sched)
    sched._reduce_lr = MethodType(custom_reduce_lr, sched)

    return opt, sched


if __name__ == "__main__":
    estimator = ProbMPS_Estimator(
        num_bins=2,
        batch_size=100,
        num_train=100,
        num_val=100,
        num_test=100,
        verbose=True,
    )
    estimator.fit(None)
