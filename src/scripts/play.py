"""Play a trained RSL-RL (PPO) checkpoint. Launches Isaac Sim first."""
import argparse
import sys
from pathlib import Path

# Make the project source root (src/) importable when running this script directly
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from isaaclab.app import AppLauncher
# local imports
# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=750, help="Length of the recorded video (in steps). Default 750 ≈ 15 seconds at 50Hz.")  # <-- 修改默认值
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--checkpoint", type=str, default=None, help="checkpoint file")
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args
# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""
import gymnasium as gym
import os
import time
import torch
from datetime import datetime

from rsl_rl.runners.on_policy_runner import OnPolicyRunner

from configs.PPORunnerCfg import CartspoleCfgPPO
from envs.carts_pole.cartspole_env import Carts2PoleEnv, Carts2PoleEnvCfg
from wrappers.ppo_wrapper import RslRlVecEnvWrapper
# PLACEHOLDER: Extension template (do not remove this comment)

def main():
   
    env_cfg = Carts2PoleEnvCfg()
    algo_cfg = CartspoleCfgPPO()
    """Play with RSL-RL agent."""
    # override configurations with non-hydra CLI arguments
    env_cfg.scene.num_envs = args_cli.num_envs
    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.sim.device = env_cfg.sim.device
   
    log_root = os.path.join("/home/ubuntu22/tdmpc", 'logs', algo_cfg.play_name)  # 设置默认日志根目录
    log_dir = os.path.join(log_root, datetime.now().strftime('%b%d_%H-%M-%S') + algo_cfg.run_name)  # 附加时间戳和运行名称
    print(f"[INFO] Loading experiment from directory: {log_root}")
   
   
    resume_path = args_cli.checkpoint
   
    env = Carts2PoleEnv(env_cfg)
   
    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,  # 使用新的默认值 750
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        env = gym.wrappers.RecordVideo(env, **video_kwargs)
        
    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    env = RslRlVecEnvWrapper(env, clip_actions=1)
    runner = OnPolicyRunner(env, algo_cfg.cfg_dict, log_dir, device="cuda")
   
    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    runner.load(resume_path)
    # obtain the trained policy for inference
    policy = runner.get_inference_policy(device=env.unwrapped.device)
    # extract the neural network module
    # we do this in a try-except to maintain backwards compatibility.
    try:
        # version 2.3 onwards
        policy_nn = runner.alg.policy
    except AttributeError:
        # version 2.2 and below
        policy_nn = runner.alg.actor_critic
    # extract the normalizer
    dt = 1 / env_cfg.fsp if hasattr(env_cfg, 'fsp') else env_cfg.control.dt  # 更健壮的 dt 获取方式
    # reset environment
    obs = env.get_observations()
    timestep = 0
    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            actions = policy(obs)
            # env stepping
            obs, _, dones, _ = env.step(actions)
            # reset recurrent states for episodes that have terminated
            policy_nn.reset(dones)
        if args_cli.video:
            timestep += 1
            # Exit the play loop after recording one video (约 15 秒)
            if timestep >= args_cli.video_length:  # 使用 >= 防止边界问题
                print("[INFO] Finished recording 15-second video. Exiting.")
                break
        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if sleep_time > 0:
            time.sleep(sleep_time)
    # close the simulator
    env.close()

if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()