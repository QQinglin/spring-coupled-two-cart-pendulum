"""Sim-to-sim deployment of a trained TD-MPC policy in MuJoCo.

Loads a TD-MPC checkpoint trained in Isaac Lab and rolls it out on the
equivalent MuJoCo (MJCF) model of the spring-coupled double-cart pole.
The observation is assembled to exactly match the training-time definition.
"""

import sys
import time
from pathlib import Path

# Make the project source root (src/) importable when running this script directly
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mujoco
import mujoco.viewer
import numpy as np
import torch

from algorithm.tdmpc import TDMPC
from configs.TDMPCRunnerCfg import TDMPCRunnerCfg


def joint_addr(model: mujoco.MjModel, joint_name: str):
    """Return ``(qpos_adr, dof_adr)`` for a joint.

    Both values are scalars (the joint's start address). Prismatic and hinge
    joints are 1-DoF in MuJoCo, so a single address per array is sufficient.
    """
    j_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    if j_id < 0:
        raise RuntimeError(f"Joint '{joint_name}' not found in MJCF.")
    return model.jnt_qposadr[j_id], model.jnt_dofadr[j_id]


def actuator_id(model: mujoco.MjModel, act_name: str):
    """Return the actuator id for ``act_name`` (raises if not found)."""
    a_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, act_name)
    if a_id < 0:
        raise RuntimeError(f"Actuator '{act_name}' not found in MJCF.")
    return a_id


def build_obs(
    data: mujoco.MjData,
    qadr_cart1,
    padr_cart1,
    qadr_spring,
    padr_spring,
    qadr_pole,
    padr_pole,
    sign_cart: float = +1.0,
):
    """Assemble the 9-dim observation, matching the Isaac Lab training layout."""
    device = "cuda"
    cart1_pos = data.qpos[qadr_cart1]
    cart1_vel = data.qvel[padr_cart1]

    spring_len = data.qpos[qadr_spring]
    spring_vel = data.qvel[padr_spring]

    pole_pos = data.qpos[qadr_pole]  # theta
    pole_vel = data.qvel[padr_pole]  # theta_dot

    # cart (the one carrying the pole) = cart1 +/- spring, sign matches training
    cart_pos = cart1_pos + sign_cart * spring_len
    cart_vel = cart1_vel + sign_cart * spring_vel

    sin_theta = np.sin(pole_pos)
    cos_theta = np.cos(pole_pos)

    obs = torch.tensor(
        [
            sin_theta,
            cos_theta,
            pole_vel,
            cart1_pos,
            cart1_vel,
            cart_pos,
            cart_vel,
            spring_len,
            spring_vel,
        ],
        dtype=torch.float32,
        device=device,
    ).unsqueeze(0)
    return obs


if __name__ == "__main__":
    # ----- Model and policy paths (edit to your setup) -----
    XML_PATH = "/home/ubuntu22/tdmpc/src/assets/carts2pole.xml"
    CKPT_PATH = "/home/ubuntu22/tdmpc/logs/tdmpc_experiment/42/models/2025-10-29_12-44-24.pt"

    # Joint / actuator names: must match the MJCF
    J_SLIDER_TO_CART1 = "slider_to_cart1"
    J_CART_TO_CART1 = "cart_to_cart1"
    J_CART_TO_POLE = "cart_to_pole"
    ACTUATOR_NAME = "cart1_actuator"  # actuator that applies force to the cart

    # Sign of the spring term in cart = cart1 (+/-) spring_len (aligns with Isaac Lab)
    SIGN_CART = +1.0  # flip to -1.0 if the direction is reversed

    # Simulation parameters
    simulation_duration = 20  # total duration of the simulation (s)
    simulation_dt = 1 / 100  # physics time step (s)
    control_decimation = 4  # apply a new action every N physics steps

    # Default joint angles: [cart1 along rail, rest spacing L0, pole angle]
    default_angles = np.array([0.0, 0.4, np.pi], dtype=np.float32)

    action_scale = 100  # scaling applied to the policy action

    num_actions = 1  # action dimension
    num_obs = 9  # observation dimension

    obs_np = np.zeros(num_obs, dtype=np.float32)

    # Use the maximum horizon at evaluation time: pass a very large step to plan()
    EVAL_STEP = 10**9

    # ----- Create and load the TD-MPC agent -----
    algo_cfg = TDMPCRunnerCfg()
    agent = TDMPC(algo_cfg)
    sd = torch.load(CKPT_PATH, map_location=algo_cfg.device)
    agent.model.load_state_dict(sd["model"])
    agent.model_target.load_state_dict(sd["model_target"])
    agent.model.eval()
    agent.model_target.eval()

    # ----- Load the MuJoCo model -----
    m = mujoco.MjModel.from_xml_path(XML_PATH)
    d = mujoco.MjData(m)
    m.opt.timestep = simulation_dt

    # Joint addresses / actuator id
    q_cart1, v_cart1 = joint_addr(m, J_SLIDER_TO_CART1)
    q_spring, v_spring = joint_addr(m, J_CART_TO_CART1)
    q_pole, v_pole = joint_addr(m, J_CART_TO_POLE)
    act_id = actuator_id(m, ACTUATOR_NAME)

    # ----- Run the simulation -----
    counter = 0
    t = 0  # TD-MPC warm-start step within an episode (t0=True only on the first step)

    with mujoco.viewer.launch_passive(m, d) as viewer:
        start = time.time()

        # Run one observe + control step up front so there is a force from the first frame
        obs_np = build_obs(
            d,
            q_cart1,
            v_cart1,
            q_spring,
            v_spring,
            q_pole,
            v_pole,
            sign_cart=SIGN_CART,
        )
        act = agent.plan(obs_np, eval_mode=True, step=EVAL_STEP, t0=True)
        if isinstance(act, torch.Tensor):
            act = act.detach().cpu().numpy()
        u = float(act[0]) if np.ndim(act) > 0 else float(act)
        d.ctrl[act_id] = np.clip(u * action_scale, -1e6, 1e6)
        t = 1

        while viewer.is_running() and time.time() - start < simulation_duration:
            step_start = time.time()

            # Advance the physics by one step
            mujoco.mj_step(m, d)
            counter += 1

            # Refresh the action only on control-decimation boundaries
            if counter % control_decimation == 0:
                # 1) Build the observation (identical to the Isaac Lab training layout)
                obs_np = build_obs(
                    d,
                    q_cart1,
                    v_cart1,
                    q_spring,
                    v_spring,
                    q_pole,
                    v_pole,
                    sign_cart=SIGN_CART,
                )

                # 2) TD-MPC planning (evaluation mode)
                act = agent.plan(obs_np, eval_mode=True, step=EVAL_STEP, t0=(t == 0))
                if isinstance(act, torch.Tensor):
                    act = act.detach().cpu().numpy()
                # Action dim = 1, so take the scalar
                u = float(act[0]) if np.ndim(act) > 0 else float(act)

                # 3) Write the control (direct force/effort)
                d.ctrl[act_id] = np.clip(u * action_scale, -1e6, 1e6)

                t += 1  # warm-start for the next iteration

            # Sync the viewer
            viewer.sync()

            # Roughly keep real time
            remain = m.opt.timestep - (time.time() - step_start)
            if remain > 0:
                time.sleep(remain)
