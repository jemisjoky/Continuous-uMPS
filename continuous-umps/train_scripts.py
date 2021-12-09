from itertools import product

from .estimator import ProbMPS_Estimator

# Define the collection of experiments to run
first_run = {}
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
        "bond_dim": 10,
        "max_calls": 50000,
        "num_train": 10000,
        "num_val": 5000,
        "num_test": 5000,
        "comet_log": False,
        # "comet_log": True,
        "early_stopping": True,
        "patience": 0,
        "cooldown": 1,
        "save_model": True,
        "slim_eval": True,
        "frameify": True,
        "embed_spec": "trig",
        "core_init_spec": "normal",
    }
)
exp_list = []
# bin_list = [2]
bin_list = [2, 3, 4, 5, 10]
embed_list = ["leg", "trig"]
dataset_list = ["mnist"]
# embed_list = [None, "nn", "trig", "leg"]
# dataset_list = ["mnist", "fashion_mnist"]
for dataset, embed_spec, num_bins in product(dataset_list, embed_list, bin_list):
    next_run = first_run.copy()
    next_run["dataset"] = dataset
    next_run["embed_spec"] = embed_spec
    next_run["num_bins"] = num_bins
    next_run["project_name"] = (
        "continuous_fashion_v1" if dataset == "fashion_mnist" else "continuous_mnist_v1"
    )
    exp_list.append(next_run)

# Run the experiments
for exp_dict in exp_list:
    estimator = ProbMPS_Estimator(**exp_dict)
    estimator.fit()
