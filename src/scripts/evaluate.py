"""Evaluate a trained TD-MPC agent in the Isaac Lab CartPole environment.

Launches Isaac Sim, rebuilds the same env/agent used for training, loads a
checkpoint, and rolls out a number of evaluation episodes (optionally saving
a video of the first one).
"""

import argparse
import sys
from pathlib import Path

# Make the project source root (src/) importable when running this script directly
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from isaaclab.app import AppLauncher

# Parse arguments first. AppLauncher injects Kit-related args such as --headless / --renderer,
# so do not add a manual --headless with type=bool.
parser = argparse.ArgumentParser(description="TD-MPC Evaluate Script (Isaac Lab)")
parser.add_argument("--exp_name", type=str, default="tdmpc_experiment", help="Experiment name")
parser.add_argument("--seed", type=int, default=42, help="Random seed")
parser.add_argument("--checkpoint", type=str, required=True, help="Path to .pt checkpoint")
parser.add_argument("--episodes", type=int, default=10, help="Number of evaluation episodes")
parser.add_argument("--save_video", action="store_true", help="Save evaluation video (first episode)")
parser.add_argument("--num_envs", type=int, default=1, help="(Optional) number of envs")
AppLauncher.add_app_launcher_args(parser)
args, hydra_args = parser.parse_known_args()
# Strip parsed args from sys.argv (kept for consistency with train.py; Hydra is not used here)
sys.argv = [sys.argv[0]] + hydra_args

# Launch Isaac Sim (must happen before importing omni/isaaclab modules below)
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import random

import numpy as np
import torch

from algorithm.tdmpc import TDMPC
from configs.TDMPCRunnerCfg import TDMPCRunnerCfg
from envs.carts_pole.cartspole_env import Carts2PoleEnv, Carts2PoleEnvCfg
from utils import logger
from wrappers.tdmpc_wrapper import TDMPCEnvWrapper


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate():
    assert torch.cuda.is_available(), "CUDA is required for evaluation."
    device = "cuda"
    set_seed(args.seed)

    # Build the same cfg used during training
    env_cfg = Carts2PoleEnvCfg()
    algo_cfg = TDMPCRunnerCfg()
    algo_cfg.exp_name = args.exp_name
    algo_cfg.seed = args.seed
    algo_cfg.device = device
    algo_cfg.save_video = args.save_video
    # Output root (same layout as training)
    work_dir = Path("/home/ubuntu22/tdmpc/logs") / algo_cfg.exp_name / str(algo_cfg.seed)
    work_dir.mkdir(parents=True, exist_ok=True)

    # Build env/agent consistently with training
    env = Carts2PoleEnv(env_cfg)
    env = TDMPCEnvWrapper(env, clip_actions=None)
    agent = TDMPC(algo_cfg)
    agent.eval()

    # Load checkpoint (supports both a raw agent state_dict and a {'agent': ...} dict)
    ckpt_path = Path(args.checkpoint)
    assert ckpt_path.exists(), f"Checkpoint not found: {ckpt_path}"
    state = torch.load(ckpt_path, map_location="cpu")
    if isinstance(state, dict) and "agent" in state:
        agent.load_state_dict(state["agent"])
    else:
        agent.load_state_dict(state)
    agent.to(device)

    # Logger is reused only for video/console output (it does not save models)
    L = logger.Logger(work_dir, algo_cfg)
    video_recorder = L.video if args.save_video else None

    # Multi-episode evaluation
    returns = []
    # These two are kept only to match the call signature; they do not affect the result
    step = 0
    env_step = 0

    obs, _ = env.reset()
    done, ep_reward, t = False, 0.0, 0
    enable_video = args.save_video
    if enable_video and video_recorder:
        video_recorder.init(env, enabled=True)
    for ep in range(args.episodes):
        while not done:
            # TD-MPC: plan the action (evaluation mode)
            action = agent.plan(obs, eval_mode=True, step=step, t0=(t == 0))
            obs, reward, done, _ = env.step(action)
            ep_reward += float(reward)
            if enable_video and video_recorder:
                video_recorder.record(env)
            t += 1
        returns.append(ep_reward)
        print(f"[EVAL] Episode {ep+1}/{args.episodes}: return={ep_reward:.2f}")
        step += 1
        env_step += 1
    avg_ret = float(np.mean(returns)) if len(returns) else float("nan")
    print(f"[EVAL] Average return over {args.episodes} episodes: {avg_ret:.2f}")

    if enable_video and video_recorder:
        video_recorder.save(env_step)
    return ep_reward


def main():
    evaluate()
    simulation_app.close()


if __name__ == "__main__":
    main()
