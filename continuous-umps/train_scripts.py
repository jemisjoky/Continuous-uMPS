from itertools import product

from estimator import ProbMPS_Estimator

default_config = {
    "input_dim": 2,
    "bond_dim": 10,
    "complex_params": False,
    "use_bias": False,
    "embed_spec": None,
    "domain_spec": None,
    "num_bins": 2,
    "dataset": "mnist",
    "dataset_dir": "./datasets/",
    "apply_downscale": True,
    "downscale_shape": (14, 14),
    "comet_log": True,
    "comet_args": {},
    "project_name": "",
    "core_init_spec": "near_eye",
    "optimizer": "Adam",
    "weight_decay": 0.0001,
    "momentum": 0.0,
    "constant_lr": False,
    "learning_rate_init": 0.001,
    "learning_rate_final": 1e-6,
    "early_stopping": False,
    "factor": 0.1,
    "patience": 1,
    "cooldown": 0,
    "slim_eval": True,
    "parallel_eval": False,
    "max_calls": 30000,
    "batch_size": 100,
    "num_train": 10000,
    "num_test": 5000,
    "num_val": 5000,
    "shuffle": True,
    "verbose": True,
    "save_model": False,
    "model_dir": "./models/",
    "seed": 0,
}

# Define the collection of experiments to run
first_run = default_config.copy()
first_run.update(
    # # Debug settings
    # {
    #     "bond_dim": 10,
    #     "max_calls": 1000,
    #     "num_train": 100,
    #     "num_val": 100,
    #     "num_test": 100,
    #     "comet_log": True,
    #     "early_stopping": True,
    #     "patience": 0,
    #     "cooldown": 1,
    #     "num_bins": 2,
    #     "embed_spec": "trig",
    # }
    # Real settings
    {
        "project_name": "continuous_fashion_v1",
        "dataset": "fashion_mnist",
        # "num_bins": 2,
        "bond_dim": 10,
        "max_calls": 50000,
        "num_train": 10000,
        "num_val": 5000,
        "num_test": 5000,
        "comet_log": True,
        "early_stopping": True,
        "patience": 0,
        "cooldown": 1,
        "embed_spec": "trig",
        "core_init_spec": "normal",
    }
)
exp_list = []
bin_list = [2, 3, 4, 5, 10]
for dataset, embed_spec, num_bins in product(
    ["mnist", "fashion_mnist"], [None, "trig"], bin_list
):
    next_run = first_run.copy()
    next_run["dataset"] = dataset
    next_run["embed_spec"] = embed_spec
    next_run["num_bins"] = num_bins
    next_run["project_name"] = (
        "continuous_fashion_v1" if dataset == "fashion_mnist" else "continuous_mnist_v1"
    )
    exp_list.append(next_run)

# Run the experiments
assert all(set(d.keys()).issubset(default_config.keys()) for d in exp_list)
for exp_dict in exp_list:
    estimator = ProbMPS_Estimator(**exp_dict)
    estimator.fit()
