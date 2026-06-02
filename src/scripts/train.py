"""Training entry point for the two-carts-pole task.

Supports two algorithms, selected via ``--algo``:
    - ``tdmpc``: custom TD-MPC implementation (default)
    - ``ppo``:   PPO via rsl_rl's OnPolicyRunner

Isaac Sim must be launched before any omni/isaaclab modules are imported, so
argument parsing and the AppLauncher call happen at the top of this file.
"""

import argparse
import sys
from pathlib import Path

# Project root (repository top level) and the source root (src/)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from isaaclab.app import AppLauncher

# Parse custom args first. Do not add --headless manually; AppLauncher injects it.
parser = argparse.ArgumentParser(description="Two-carts-pole Training Script")
parser.add_argument("--exp_name", type=str, default="tdmpc_experiment", help="Experiment name")
parser.add_argument("--seed", type=int, default=42, help="Random seed")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--save_video", action="store_true", help="Save evaluation videos (default: False).")
parser.add_argument("--save_model", action="store_true", help="Save model checkpoints (default: False).")
parser.add_argument("--algo", type=str, default="tdmpc", help="Algorithm to train: 'tdmpc' or 'ppo'.")
parser.add_argument(
    "--log_dir", type=str, default=str(PROJECT_ROOT / "logs"), help="Root directory for logs/checkpoints."
)

# AppLauncher injects Kit-related args such as --headless / --renderer
AppLauncher.add_app_launcher_args(parser)
args, hydra_args = parser.parse_known_args()
# Strip parsed args from sys.argv (kept for consistency; Hydra is not used here)
sys.argv = [sys.argv[0]] + hydra_args

# Launch Isaac Sim (do not pass headless=... here to avoid conflicting with args)
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# Kit is running; the imports below may pull in omni.* modules, so they must come after launch.
import os
import random
import time
from datetime import datetime

import numpy as np
import torch
import wandb

from rsl_rl.runners.on_policy_runner import OnPolicyRunner

from algorithm.helper import Episode, ReplayBuffer
from algorithm.tdmpc import TDMPC
from configs.PPORunnerCfg import CartspoleCfgPPO
from configs.TDMPCRunnerCfg import TDMPCRunnerCfg
from envs.carts_pole.cartspole_env import Carts2PoleEnv, Carts2PoleEnvCfg
from utils import logger
from wrappers.ppo_wrapper import RslRlVecEnvWrapper
from wrappers.tdmpc_wrapper import TDMPCEnvWrapper


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def evaluate(env, agent, num_episodes, step, env_step, video):
    """Evaluate a trained agent and optionally save a video."""
    episode_rewards = []
    for i in range(num_episodes):
        obs, _ = env.reset()
        done, ep_reward, t = False, 0, 0
        if video:
            video.init(env, enabled=(i == 0))
        while not done:
            action = agent.plan(obs, eval_mode=True, step=step, t0=t == 0)
            obs, reward, done, _ = env.step(action)
            ep_reward += reward
            if video:
                video.record(env)
            t += 1
        episode_rewards.append(ep_reward.cpu().numpy())
        if video and len(video.frames) > 0:
            video.save(env_step)
    return np.nanmean(episode_rewards)


def train_tdmpc(env, algo_cfg):
    """Train the TD-MPC agent. Requires a CUDA-enabled device."""
    set_seed(algo_cfg.seed)
    # Output directory for this run
    work_dir = Path(algo_cfg.log_dir) / algo_cfg.exp_name / str(algo_cfg.seed)
    algo_cfg.obs_shape = tuple(int(x) for x in algo_cfg.obs_shape)

    agent, buffer = TDMPC(algo_cfg), ReplayBuffer(algo_cfg)
    env = TDMPCEnvWrapper(env, clip_actions=None)

    # Initialize WandB
    wandb_run_name = f"tdmpc_{algo_cfg.exp_name}_{algo_cfg.seed}"
    wandb.init(
        project="two_carts_pole_tdmpc",
        name=wandb_run_name,
        config={
            "algo": "TD-MPC",
            "train_steps": algo_cfg.train_steps,
            "episode_length": algo_cfg.episode_length,
            "seed": algo_cfg.seed,
        },
    )

    # Run training
    L = logger.Logger(work_dir, algo_cfg)
    episode_idx, start_time, mean_reward = 0, time.time(), []
    total_episode = (algo_cfg.train_steps + algo_cfg.episode_length) / algo_cfg.episode_length
    global_step_counter = 0

    for step in range(0, algo_cfg.train_steps + algo_cfg.episode_length, algo_cfg.episode_length):
        # Collect a full trajectory (one episode)
        obs, _ = env.reset()
        episode = Episode(algo_cfg, obs)

        for t in range(algo_cfg.episode_length):
            # Plan an action and step the env; obs order:
            # [sin_theta, cos_theta, pole_vel, cart1_pos, cart1_vel, cart_pos, cart_vel, spring_length, spring_velocity]
            action = agent.plan(obs, step=step, t0=episode.first)
            next_obs, reward, dones, extrals = env.step(action)
            # `episode += (...)` extends the episode buffer with this transition
            episode += (obs, action, reward, dones)
            obs = next_obs
            global_step_counter += 1

        assert len(episode) == algo_cfg.episode_length
        buffer += episode  # store the whole episode at once

        # Update the model. Only start once the buffer has enough data (>= seed_steps).
        train_metrics = {}
        if step >= algo_cfg.seed_steps:
            # Right after seeding, do a large batch of updates; afterwards update once per env step.
            num_updates = algo_cfg.seed_steps if step == algo_cfg.seed_steps else algo_cfg.episode_length
            for i in range(num_updates):
                train_metrics.update(agent.update(buffer, step + i))

        # --- Log the training episode ---
        episode_idx += 1
        env_step = int(step * algo_cfg.action_repeat)
        total_timesteps = int((step + 1) * algo_cfg.episode_length)

        mean_reward.append(episode.cumulative_reward.mean().item())

        common_metrics = {
            "episode": episode_idx,
            "total_timesteps": total_timesteps,
            "total_time": time.time() - start_time,
            "episode_reward": episode.cumulative_reward,
            "env_step": env_step,
        }
        train_metrics.update(common_metrics)

        # WandB logging
        try:
            wandb_payload = {
                "train/episode": episode_idx,
                "train/episode_reward": float(episode.cumulative_reward.mean().item()),
                "train/mean_reward_running": float(np.mean(mean_reward)),
                "train/total_timesteps": total_timesteps,
            }

            # Pull spring/pole debug info out of the env extras (keys set in Carts2PoleEnv._get_rewards)
            if extrals and "episode" in extrals:
                episode_info = extrals["episode"]
                if "debug/spring_length" in episode_info:
                    wandb_payload["train/spring_length"] = float(episode_info["debug/spring_length"])
                if "debug/spring_disp" in episode_info:
                    wandb_payload["train/spring_deformation"] = float(episode_info["debug/spring_disp"])
                if "debug/pole_angle_deg" in episode_info:
                    wandb_payload["train/pole_angle_deg"] = float(episode_info["debug/pole_angle_deg"])
                if "debug/pole_vel_rad_s" in episode_info:
                    wandb_payload["train/pole_vel_rad_s"] = float(episode_info["debug/pole_vel_rad_s"])
                if "debug/cart_vel2" in episode_info:
                    wandb_payload["train/cart_vel2"] = float(episode_info["debug/cart_vel2"])

            # Training loss metrics (only present once updates have started)
            if "mean_action_noise_std" in train_metrics:
                wandb_payload.update(
                    {
                        "train/action_noise_std": float(train_metrics["mean_action_noise_std"]),
                        "train/value_loss": float(train_metrics["mean_value_loss"]),
                        "train/pi_loss": float(train_metrics["mean_pi_loss"]),
                        "train/reward_loss": float(train_metrics["mean_reward_loss"]),
                        "train/weighted_loss": float(train_metrics["weighted_loss"]),
                        "train/total_loss": float(train_metrics["total_loss"]),
                    }
                )

            wandb.log(wandb_payload, step=global_step_counter)
        except Exception as e:
            print(f"Warning: Failed to log to wandb: {e}")

        L.log(train_metrics, category="train")

        # --- Console output ---
        log_string = (
            f"\n{'#' * 60}\n"
            f"  \033[1mLearning episode {train_metrics['episode']}/{total_episode}\033[0m\n"
        )

        if "mean_action_noise_std" in train_metrics:
            log_string += f"{'mean action noise std:':>25} {train_metrics['mean_action_noise_std']:.2f}\n"
            log_string += f"{'mean value loss:':>25} {train_metrics['mean_value_loss']:.2f}\n"
            log_string += f"{'mean pi loss:':>25} {train_metrics['mean_pi_loss']:.2f}\n"
            log_string += f"{'mean reward loss:':>25} {train_metrics['mean_reward_loss']:.2f}\n"
            log_string += f"{'weighted loss:':>25} {train_metrics['weighted_loss']:.2f}\n"
            log_string += f"{'total loss:':>25} {train_metrics['total_loss']:.2f}\n"

        log_string += (
            f"{'episode reward:':>25} {episode.cumulative_reward.item():.2f}\n"
            f"{'mean reward:':>25} {np.mean(mean_reward):.2f}\n"
            f"{'-' * 60}\n"
            f"{'total timesteps:':>25} {train_metrics['total_timesteps']:.2f}\n"
            f"{'total time:':>25} {train_metrics['total_time']:.2f}\n"
            f"{'time elapsed:':>25} {time.strftime('%H:%M:%S', time.gmtime(common_metrics['total_time']))}\n"
        )
        print(log_string)

        if env_step % algo_cfg.eval_freq == 0:
            common_metrics["episode_reward"] = evaluate(
                env, agent, algo_cfg.eval_episodes, step, env_step, L.video
            )
            L.log(common_metrics, category="eval")

    wandb.finish()
    L.finish(agent)
    env.close()


def train_ppo(env, algo_cfg):
    """Train a PPO agent using rsl_rl's OnPolicyRunner."""
    log_root = os.path.join(algo_cfg.log_dir, algo_cfg.experiment_name)
    log_dir = os.path.join(log_root, datetime.now().strftime("%b%d_%H-%M-%S") + "_" + algo_cfg.run_name)

    env = RslRlVecEnvWrapper(env, clip_actions=1)
    runner = OnPolicyRunner(env, algo_cfg.cfg_dict, log_dir, device="cuda")
    runner.learn(algo_cfg.cfg_dict["max_iterations"])
    env.close()


def train(env_cfg, algo_cfg):
    """Dispatch to the TD-MPC or PPO training loop based on ``algo_cfg.train_algo``."""
    env = Carts2PoleEnv(env_cfg)
    if algo_cfg.train_algo == "TDMPC":
        train_tdmpc(env, algo_cfg)
    elif algo_cfg.train_algo == "PPO":
        train_ppo(env, algo_cfg)
    else:
        raise ValueError(f"Unknown train_algo: {algo_cfg.train_algo}")
    print("Training completed successfully")


def main():
    env_cfg = Carts2PoleEnvCfg()
    if args.algo == "tdmpc":
        algo_cfg = TDMPCRunnerCfg()
        # Override config from the CLI
        algo_cfg.exp_name = args.exp_name
        algo_cfg.seed = args.seed
        algo_cfg.save_model = args.save_model
        algo_cfg.save_video = args.save_video
        algo_cfg.train_algo = "TDMPC"
        algo_cfg.device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        algo_cfg = CartspoleCfgPPO()

    # Root directory for logs/checkpoints (used by both training loops)
    algo_cfg.log_dir = args.log_dir

    print(f"[INFO] Exp: {algo_cfg.exp_name}, Seed: {algo_cfg.seed}, Device: {algo_cfg.device}")
    print("torch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    print("CUDA devices:", torch.cuda.device_count())
    print("Current device:", torch.cuda.current_device() if torch.cuda.is_available() else "None")
    print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))

    train(env_cfg, algo_cfg)


if __name__ == "__main__":
    main()
    # Always close Kit, whether or not training raised
    simulation_app.close()
