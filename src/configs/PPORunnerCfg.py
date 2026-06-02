import inspect


class BaseConfig:
    def __init__(self) -> None:
        """Initialize all member classes recursively, ignoring built-in names (starting with '__')."""
        self.init_member_classes(self)

    @staticmethod
    def init_member_classes(obj):
        # Iterate over all attribute names
        for key in dir(obj):
            # Skip built-in attributes
            if key == "__class__":
                continue
            # Get the corresponding attribute object
            var = getattr(obj, key)
            # If the attribute is a class, instantiate it and recurse
            if inspect.isclass(var):
                i_var = var()
                # Replace the type with an instance
                setattr(obj, key, i_var)
                BaseConfig.init_member_classes(i_var)


class CartspoleCfgPPO(BaseConfig):
    seed = 1  # random seed for reproducibility
    class_name = "OnPolicyRunner"
    train_algo = "PPO"
    device = "cuda"
    exp_name = "two-carts-pole"
    run_name = ""
    experiment_name = "ppo_experiment"
    play_name = "ppo_play"

    # Config dict consumed directly by OnPolicyRunner
    cfg_dict = {
        # Runner / training-loop configuration
        "class_name": "ActorCritic",
        "num_steps_per_env": 24,
        "max_iterations": 16667,
        "save_interval": 500,
        "experiment_name": "ppo_experiment",
        "run_name": "",
        "resume": False,
        "load_run": -1,
        "checkpoint": -1,
        "resume_path": None,
        # Logging backend
        "logger": "wandb",
        "wandb_project": "two_carts_pole",
        # Observation groups (OnPolicyRunner sets self.cfg["obs_groups"] = ...)
        "obs_groups": {},
        # Used by some OnPolicyRunner versions
        "empirical_normalization": None,
        # Policy network
        "policy": {
            "class_name": "ActorCritic",
            "init_noise_std": 0.3,  # initial exploration noise
            "actor_hidden_dims": [512, 256, 128],
            "critic_hidden_dims": [512, 256, 128],
            "activation": "elu",  # elu / relu / selu / lrelu / tanh / sigmoid ...
            # For an RNN policy, add e.g.:
            # "rnn_type": "lstm",
            # "rnn_hidden_size": 512,
            # "rnn_num_layers": 1,
        },
        # PPO algorithm parameters
        "algorithm": {
            "class_name": "PPO",
            # Loss coefficients
            "value_loss_coef": 1.0,  # balances value loss against policy loss
            "use_clipped_value_loss": True,  # whether to use the clipped value loss
            "clip_param": 0.1,  # PPO clip parameter
            "entropy_coef": 0.0001,  # entropy coefficient to encourage exploration
            # Optimization settings
            "num_learning_epochs": 5,  # epochs per update
            "num_mini_batches": 4,  # number of mini-batches
            "learning_rate": 1e-3,  # learning rate
            "schedule": "adaptive",  # learning-rate schedule (adaptive / fixed)
            # RL hyperparameters
            "gamma": 0.99,  # discount factor
            "lam": 0.95,  # GAE lambda
            "desired_kl": 0.001,  # target KL for the adaptive schedule
            "max_grad_norm": 1.0,  # gradient-clipping threshold
        },
        # Kept for code paths that read train_cfg["runner"]
        "runner": {
            "policy_class_name": "ActorCritic",
            "class_name": "PPO",
            "num_steps_per_env": 24,
            "max_iterations": 16667,
            "save_interval": 1000,
            "experiment_name": "ppo_experiment",
            "run_name": "",
            "resume": False,
            "load_run": -1,
            "checkpoint": -1,
            "resume_path": None,
            # Also set the logger here as a safeguard
            "logger": "wandb",
        },
    }
