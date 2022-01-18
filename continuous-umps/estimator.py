"""Probabilistic MPS written as a sklearn estimator"""
from time import time
from copy import deepcopy
from types import MethodType
from functools import partial
from math import ceil, log, prod

from comet_ml import Experiment
from sklearn.base import BaseEstimator, DensityMixin
import torch

from torchmps import ProbMPS
from .utils import FakeLogger, null
from torchmps.embeddings import (
    unit_interval,
    trig_embed,
    init_mlp_embed,
    legendre_embed,
    FixedEmbedding,
)


class ProbMPS_Estimator(BaseEstimator, DensityMixin):
    """
    Wraps ProbMPS model to give scikit-learn API, simplifying model training

    Args:
        input_dim: Dimension of the MPS input indices
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
        embed_layers=2,
        embed_hdims=10,
        frameify=False,
        num_bins=None,
        dataset="mnist",
        dataset_dir="./datasets/",
        apply_downscale=True,
        downscale_shape=(14, 14),
        comet_log=False,
        comet_args={},
        project_name="",
        core_init_spec="normal",
        optimizer="Adam",
        weight_decay=0.0001,
        momentum=0.0,
        constant_lr=False,
        learning_rate_init=0.001,
        learning_rate_final=1e-6,
        early_stopping=False,
        factor=0.1,
        patience=1,
        cooldown=0,
        slim_eval=True,
        parallel_eval=False,
        max_calls=20000,
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
        self.embed_layers = embed_layers
        self.embed_hdims = embed_hdims
        self.frameify = frameify
        self.num_bins = num_bins
        self.dataset = dataset
        self.dataset_dir = dataset_dir
        self.apply_downscale = apply_downscale
        self.downscale_shape = downscale_shape
        self.comet_log = comet_log
        self.comet_args = comet_args
        self.project_name = project_name
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
        self.max_calls = max_calls
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
        start_time = time()

        # Modify some experimental parameters in advance of logging
        self.max_epochs = ceil(self.max_calls / self.num_train)
        self.dataset = self.dataset.lower()

        # Initialize embedding using embed_spec and domain_spec
        assert self.embed_spec in [None, "trig", "leg", "nn"]
        if self.embed_spec == "trig":
            assert self.num_bins >= 2
            embed_domain = unit_interval
            raw_embed = partial(trig_embed, emb_dim=self.num_bins)
            embed_fun = FixedEmbedding(raw_embed, embed_domain, frameify=self.frameify)
            self.input_dim = self.num_bins
            continuous = True
        elif self.embed_spec == "leg":
            assert self.num_bins >= 2
            embed_domain = unit_interval
            raw_embed = partial(legendre_embed, emb_dim=self.num_bins)
            embed_fun = FixedEmbedding(raw_embed, embed_domain, frameify=self.frameify)
            self.input_dim = self.num_bins
            continuous = True
        elif self.embed_spec == "nn":
            assert self.embed_hdims is not None
            assert self.embed_layers is not None
            embed_fun = init_mlp_embed(
                self.num_bins,
                num_layers=self.embed_layers,
                hidden_dims=self.embed_hdims,
                frameify=self.frameify,
            )
            embed_domain = unit_interval
            self.input_dim = self.num_bins
            continuous = True
        elif self.embed_spec is None:
            # If we're binning data, set the input dimension to number of bins
            embed_fun = embed_domain = None
            self.input_dim = self.num_bins
            continuous = False
        else:
            raise NotImplementedError

        # Build experiment name
        exp_name = (
            f"nb{self.num_bins}_bd{self.bond_dim}_nt{round(self.num_train / 1000)}k"
        )
        frame_id = "f_" if self.frameify else ""
        if self.embed_spec == "trig":
            exp_name = "trig_" + frame_id + exp_name
        elif self.embed_spec == "leg":
            exp_name = "leg_" + frame_id + exp_name
        elif self.embed_spec == "nn":
            exp_name = f"nn{self.embed_layers}_" + frame_id + exp_name
        if self.core_init_spec == "normal":
            exp_name = exp_name + "_gs"
        elif self.core_init_spec is None:
            exp_name = exp_name + "_eye"

        # Initialize Comet data logging
        if self.comet_log:
            # TODO: Make name
            if self.project_name == "":
                raise ValueError("Must specify experiment name to use comet logging")
            logger = Experiment(project_name=self.project_name)
            logger.log_parameters(self.__dict__)
            logger.set_name(exp_name)
        else:
            logger = FakeLogger()

        # Conditional print function
        global cprint
        cprint = partial(print, flush=True) if self.verbose else null

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        train_g = torch.Generator(device=device)
        train_g.manual_seed(self.seed)
        self.val_g = torch.Generator(device=device)
        self.val_g.manual_seed(self.seed)

        # Import our dataset
        if self.dataset in ["mnist", "fashion_mnist"]:
            from .datasets import load_mnist

            fashion = self.dataset == "fashion_mnist"
            train, val, test = load_mnist(
                fashion=fashion,
                num_train=self.num_train,
                num_test=self.num_test,
                num_val=self.num_val,
                downscale=self.apply_downscale,
                downscale_shape=self.downscale_shape,
                num_bins=self.num_bins,
                continuous=continuous,
                dataset_dir=self.dataset_dir,
                device=device,
            )
            self.seq_len = (
                prod(self.downscale_shape) if self.apply_downscale else 28 ** 2
            )
        else:
            raise NotImplementedError

        # Compute loss adjustment coming from binning and continuous domains
        # This ensures all models begin training with loss at same scale
        if self.embed_spec is not None and embed_domain == unit_interval:
            # Expand unit interval to have effective width of 2
            self.loss_adjust = log(2) * self.seq_len
        else:
            assert self.num_bins is not None
            self.loss_adjust = (log(2) - log(self.num_bins)) * self.seq_len

        # Initialize model
        self.model = ProbMPS(
            self.seq_len,
            self.input_dim,
            self.bond_dim,
            init_method=self.core_init_spec,
            complex_params=self.complex_params,
            use_bias=self.use_bias,
            embed_fun=embed_fun,
            # domain=embed_domain,
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

        def optimize_one_epoch():
            """Carries out one epoch of gradient updates over training data"""
            nonlocal epoch
            batches_per_epoch = ceil(len(train) / self.batch_size)
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
                step_time = time()
                optimizer.zero_grad()
                loss = (
                    self.model.loss(
                        *batch,
                        slim_eval=self.slim_eval,
                        parallel_eval=self.parallel_eval,
                    )
                    + self.loss_adjust
                )
                train_loss += loss.detach()

                loss.backward()
                optimizer.step()
                frac_ep = epoch * batches_per_epoch + num
                logger.log_metrics(
                    {"batch_loss": loss, "step_time": time() - step_time}, epoch=frac_ep
                )
                cprint(f"    Batch {num}: {loss:.2f}", end="\r")

            # Return average loss
            train_loss /= num
            return train_loss

        now = time()
        cprint(f"\n\nExperiment: {exp_name}")
        cprint(f"Initialization time: {now - start_time:.2f}s")
        cprint("Starting training...\n")

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
                self._check_best(val_loss)
                epoch_time = time() - now
                cprint(f"  Epoch time: {epoch_time:.2f}s")
                now = time()
                logger.log_metrics(
                    {
                        "train_loss": train_loss,
                        "val_loss": val_loss,
                        "epoch_time": epoch_time,
                    },
                    epoch=epoch,
                )

                # Check for early stopping
                if scheduler.step(val_loss):
                    cprint("\nNon-improving val loss, stopping training early")
                    break

        except KeyboardInterrupt:
            cprint("\nTraining stopped early by user")

        # Evaluate test loss using final model, and all losses using best model
        test_loss = -self.score(test)
        logger.log_metric("test_loss", test_loss)

        if hasattr(self, "best_model"):
            self.model.load_state_dict(self.best_model)
            best_train, best_val, best_test = [
                -self.score(ds) for ds in [train, val, test]
            ]

            cprint(f"\nTrain loss at best: {best_train:.2f}")
            cprint(f"Val loss at best:   {best_val:.2f}")
            cprint(f"Test loss at best:  {best_test:.2f}")
            cprint(f"Test loss at end:   {test_loss:.2f}")
            cprint(f"Total runtime:      {now - start_time:.1f}s")
            logger.log_metrics(
                {
                    "best_train": best_train,
                    "best_val": best_val,
                    "best_test": best_test,
                }
            )

        # Save model to disk if desired
        if self.save_model:
            torch.save(self.model, self.model_dir + exp_name + ".model")

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
                loss = (
                    self.model.loss(
                        *batch,
                        slim_eval=self.slim_eval,
                        parallel_eval=self.parallel_eval,
                    )
                    + self.loss_adjust
                )
            val_loss += loss.detach()
            cprint(f"    Batch {num}", end="\r")

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
        self.__reduce_lr(epoch)

    def custom_step(self, metrics, epoch=None):
        num_red = self.num_reductions
        self._step(metrics, epoch=epoch)
        if early_stopping and self.num_reductions >= (
            1 if constant_lr else max_reductions
        ):
            return True
        if not constant_lr and self.num_reductions > num_red:
            cprint("### Decreasing LR ###")
        if not early_stopping:
            return False
        return False

    sched.step = MethodType(custom_step, sched)
    sched._reduce_lr = MethodType(custom_reduce_lr, sched)

    return opt, sched
