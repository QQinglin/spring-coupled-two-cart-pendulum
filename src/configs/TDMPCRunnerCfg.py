from isaaclab.utils.configclass import configclass


@configclass
class TDMPCRunnerCfg:
    # General
    task: str = "two-carts-pole"
    modality: str = "state"
    task_title: str = "Carts2pole Balance"
    device: str = "cuda"
    exp_name: str = "two-carts-pole"
    seed: int = 1

    # Discount and environment
    discount: float = 0.99
    action_repeat: int = 4
    episode_length: int = 500
    train_steps: int = 100000

    # Planning parameters
    iterations: int = 6
    num_samples: int = 512
    num_elites: int = 64
    mixture_coef: float = 0.05
    min_std: float = 0.05
    temperature: float = 0.5
    momentum: float = 0.1

    # Learning parameters
    batch_size: int = 512
    max_buffer_size: int = 100000
    horizon: int = 5
    reward_coef: float = 0.5
    value_coef: float = 0.1
    consistency_coef: float = 2
    rho: float = 0.5
    kappa: float = 0.1
    lr: float = 1e-3
    std_schedule: str = "linear(0.5, 0.05, 25000)"
    horizon_schedule: str = "linear(1, 5, 25000)"
    per_alpha: float = 0.6
    per_beta: float = 0.4
    grad_clip_norm: float = 10
    seed_steps: int = 500
    update_freq: int = 2
    tau: float = 0.01

    # Network architecture
    enc_dim: int = 256
    mlp_dim: int = 512
    latent_dim: int = 50

    # Weights & Biases (wandb) configuration
    use_wandb: bool = False
    wandb_project: str = "none"
    wandb_entity: str = "none"

    # Evaluation and checkpointing
    eval_freq: int = 20000
    eval_episodes: int = 10
    save_video: bool = False
    save_model: bool = False

    # Action / observation dimensions
    action_dim: int = 1
    obs_dim: int = 9
    obs_shape: list[int] = [9]
    num_envs: int = 1
