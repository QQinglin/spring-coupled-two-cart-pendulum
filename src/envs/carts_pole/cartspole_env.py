"""Direct RL environment for a spring-coupled double-cart pole.

Physical model:
    - ``cart1`` slides along a rail (``slider_to_cart1``) and is the only
      actively driven joint (a force/effort is applied to it).
    - ``cart`` is connected to ``cart1`` through a spring (``cart_to_cart1``);
      its motion is therefore derived from ``cart1`` and the spring state.
    - the pole hinges on ``cart`` (``cart_to_pole``) and must be balanced upright.

The agent applies an effort on ``cart1`` and is rewarded for keeping the pole
upright while keeping the carts within bounds.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg, ViewerCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import TiledCamera, TiledCameraCfg
from isaaclab.sim import RigidBodyMaterialCfg, SimulationCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils import configclass
from isaaclab.utils.math import sample_uniform

from envs.carts_pole.cartspole import CARTPOLE_CFG


@configclass
class Carts2PoleEnvCfg(DirectRLEnvCfg):
    """Configuration for :class:`Carts2PoleEnv` (scene, sim, spaces, reward scales)."""

    # Environment parameters
    decimation = 2
    episode_length_s = 5.0
    fsp = 120
    device = "cuda"
    action_space = 1
    action_scale = 50  # action scaling factor
    observation_space = 9
    num_envs = 1

    # Simulation configuration (dt derived from fsp)
    sim: SimulationCfg = SimulationCfg(
        dt=1 / fsp,
        render_interval=decimation,
        physics_material=RigidBodyMaterialCfg(
            static_friction=0.01,
            dynamic_friction=0.01,
            restitution=0.0,
            friction_combine_mode="average",
            restitution_combine_mode="average",
        ),
    )

    robot_cfg: ArticulationCfg = CARTPOLE_CFG.replace(prim_path="/World/cartpole")

    tiled_camera: TiledCameraCfg = TiledCameraCfg(
        # Single env can use an absolute path; for multi-env use /World/envs/env_.*/Camera
        prim_path="/World/Camera",
        offset=TiledCameraCfg.OffsetCfg(pos=(-18.0, 0.0, 4.0), rot=(1, 0, 0, 0), convention="world"),
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(focal_length=24.0, horizontal_aperture=20.955),
        width=1280,
        height=720,
    )

    # Viewer settings
    viewer = ViewerCfg(eye=(-12.0, 0.0, 4.0), lookat=(0.0, 0.0, 2.0))

    # Joint names: [slider_to_cart1, cart_to_cart1, cart_to_pole]
    cart_dof_name = ["slider_to_cart1", "cart_to_cart1", "cart_to_pole"]

    # Scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs, env_spacing=4.0, replicate_physics=True)

    max_cart_pos = 8.0
    initial_pole_angle_range = [-0.25, 0.25]

    # Reward scales
    rew_scale_alive = 0.0
    rew_scale_terminated = -10.0
    rew_scale_pole_pos = -0.1
    rew_scale_cart_vel = -0.01
    rew_scale_pole_vel = -0.005

    rew_scale_spring_length = -0.5
    rew_scale_spring_vel = -0.05
    rew_scale_cart1_vel = 0

    rew_scale_cart1_pos = 0
    rew_scale_cart_pos = 0

    alive_bonus = 0.8
    terminated_penalty = -13.0

    w_upright = 4.0
    w_theta_dot = 0.005

    w_pos = 0.01
    w_vel = 2.0

    w_spring = 0.004  # penalize (L - L0)^2
    w_spring_vel = 0.05

    w_action = 0.0005

    spring_original_length = 1.0


class Carts2PoleEnv(DirectRLEnv):
    cfg: Carts2PoleEnvCfg

    def __init__(self, cfg: "Carts2PoleEnvCfg", render_mode: str | None = "rgb_array", **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Cart joint indices: cart_dof_name[0:2] = ['slider_to_cart1', 'cart_to_cart1']
        self._cart_dof_idx, _ = self.cartspole.find_joints(self.cfg.cart_dof_name[0:2])

        # Pole joint index
        self._pole_dof_idx, _ = self.cartspole.find_joints(self.cfg.cart_dof_name[2:])

        self.action_scale = self.cfg.action_scale

        # References to joint position/velocity data
        self.joint_pos = self.cartspole.data.joint_pos
        self.joint_vel = self.cartspole.data.joint_vel

        # cart1: the actively driven cart (slider_to_cart1), shape (num_envs, 1)
        self.cart1_pos = self.joint_pos[:, self._cart_dof_idx[0]].unsqueeze(dim=1)
        self.cart1_vel = self.joint_vel[:, self._cart_dof_idx[0]].unsqueeze(dim=1)

        # Spring (cart_to_cart1): the USD rest length is 1.0, so the absolute spring
        # length is 1.0 plus the joint displacement around that rest position.
        self.spring_length = 1 + self.joint_pos[:, self._cart_dof_idx[1]].unsqueeze(dim=1)
        self.spring_velocity = self.joint_vel[:, self._cart_dof_idx[1]].unsqueeze(dim=1)

        # cart: not directly actuated; its state is cart1 offset by the spring.
        # cart1 sits at a larger axis position than cart, hence the negative sign.
        self.cart_pos = -self.spring_length + self.cart1_pos
        self.cart_vel = -self.spring_velocity + self.cart1_vel

        # Pole (cart_to_pole): angle is 0 when upright
        self.pole_pos = self.joint_pos[:, self._pole_dof_idx[0]].unsqueeze(dim=1)
        self.pole_vel = self.joint_vel[:, self._pole_dof_idx[0]].unsqueeze(dim=1)

    def _setup_scene(self):
        """Spawn the articulation, camera, ground and light, then clone per-env copies."""
        self.cartspole = Articulation(self.cfg.robot_cfg)
        self._tiled_camera = TiledCamera(self.cfg.tiled_camera)

        # Add ground plane to the scene
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())

        # Clone and replicate environments
        self.scene.clone_environments(copy_from_source=False)

        # Explicitly filter collisions for CPU simulation
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])

        # Register articulation and sensors with the scene
        self.scene.articulations["cartpole"] = self.cartspole
        self.scene.sensors["tiled_camera"] = self._tiled_camera

        # Add a dome light
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.raw_action = actions
        self.actions = self.action_scale * actions.clone()  # scale the action

    def _apply_action(self) -> None:
        # Only slider_to_cart1 is actuated; the spring and pole evolve passively
        self.cartspole.set_joint_effort_target(self.actions, joint_ids=self._cart_dof_idx[0])

    def _get_observations(self) -> dict:
        """Build the 9-dim observation from the latest joint state."""
        # cart1 (actively driven)
        self.cart1_pos = self.joint_pos[:, self._cart_dof_idx[0]].unsqueeze(dim=1)
        self.cart1_vel = self.joint_vel[:, self._cart_dof_idx[0]].unsqueeze(dim=1)

        # Spring (absolute length = rest length 1.0 + joint displacement)
        self.spring_length = 1 + self.joint_pos[:, self._cart_dof_idx[1]].unsqueeze(dim=1)
        self.spring_velocity = self.joint_vel[:, self._cart_dof_idx[1]].unsqueeze(dim=1)

        # cart1 sits at a larger axis position than cart, hence the negative sign
        self.cart_pos = -self.spring_length + self.cart1_pos
        self.cart_vel = 0 + self.cart1_vel + self.spring_velocity

        # Pole position and velocity
        self.pole_pos = self.joint_pos[:, self._pole_dof_idx[0]].unsqueeze(dim=1)
        self.pole_vel = self.joint_vel[:, self._pole_dof_idx[0]].unsqueeze(dim=1)

        # Encode the angle as sin/cos so the policy never sees the +/-pi discontinuity
        sin_theta = torch.sin(self.pole_pos)
        cos_theta = torch.cos(self.pole_pos)

        obs = torch.cat(
            (
                # [sin_theta, cos_theta, pole_vel, cart1_pos, cart1_vel, cart_pos, cart_vel, spring_length, spring_velocity]
                sin_theta,
                cos_theta,
                self.pole_vel,
                self.cart1_pos,
                self.cart1_vel,
                self.cart_pos,
                self.cart_vel,
                self.spring_length,
                self.spring_velocity,
            ),
            dim=-1,
        )

        observations = {"policy": obs}  # obs.shape = (num_envs, 9)
        return observations

    def _get_rewards(self) -> torch.Tensor:
        # Log debug statistics into self.extras["episode"] (picked up by the RL framework's logger)
        with torch.no_grad():
            if "episode" not in self.extras:
                self.extras["episode"] = {}

            self.extras["episode"]["debug/pole_upright"] = float(torch.cos(self.pole_pos).mean().item())
            self.extras["episode"]["debug/cart_vel2"] = float(((self.cart_vel ** 2)).mean().item())
            self.extras["episode"]["debug/spring_length"] = float((self.spring_length).mean().item())
            self.extras["episode"]["debug/spring_disp"] = float(
                self.joint_pos[:, self._cart_dof_idx[1]].mean().item()
            )

            self.extras["episode"]["debug/pole_angle_deg"] = float(
                (self.pole_pos * 180 / math.pi).mean().item()
            )
            self.extras["episode"]["debug/pole_angle_abs_deg"] = float(
                (self.pole_pos.abs() * 180 / math.pi).mean().item()
            )
            self.extras["episode"]["debug/pole_vel_rad_s"] = float(self.pole_vel.mean().item())

            # Record reward components
            pole_reward = torch.exp(-5.0 * self.pole_pos**2) * 10.0
            self.extras["episode"]["reward/pole_upright"] = float(pole_reward.mean().item())
            self.extras["episode"]["reward/pole_vel"] = float((-0.5 * self.pole_vel**2).mean().item())

        total_reward = compute_rewards(
            self.cfg.rew_scale_alive,
            self.cfg.rew_scale_terminated,
            # pole
            self.cfg.rew_scale_pole_vel,
            self.cfg.rew_scale_pole_pos,
            # cart
            self.cfg.rew_scale_cart_pos,
            self.cfg.rew_scale_cart_vel,
            # cart1
            self.cfg.rew_scale_cart1_pos,
            self.cfg.rew_scale_cart1_vel,
            # spring
            self.cfg.rew_scale_spring_length,
            self.cfg.rew_scale_spring_vel,
            # pole
            self.pole_pos,
            self.pole_vel,
            # cart (without actuator)
            self.cart_pos,
            self.cart_vel,
            # cart1 (with actuator)
            self.cart1_pos,
            self.cart1_vel,
            # spring
            self.spring_length,
            self.spring_velocity,
            self.cfg.spring_original_length,
            # done
            self.reset_terminated,
            # action
            self.actions,
        )
        return total_reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (terminated, time_out) masks, each of shape (num_envs,)."""
        self.joint_pos = self.cartspole.data.joint_pos
        self.joint_vel = self.cartspole.data.joint_vel

        # Episode ends when the maximum number of steps is reached
        time_out = self.episode_length_buf >= self.max_episode_length - 1

        # Terminate if either cart leaves the rail bound, or the pole tips past +/-90 degrees.
        # Check both cart1 and the derived cart position; torch.any reduces over the joint axis.
        all_carts_pos = torch.cat(
            (self.joint_pos[:, self._cart_dof_idx[0]].unsqueeze(dim=1), self.cart_pos), dim=1
        )
        out_of_bounds = torch.any(torch.abs(all_carts_pos) > self.cfg.max_cart_pos, dim=1)
        out_of_bounds = out_of_bounds | torch.any(
            torch.abs(self.joint_pos[:, self._pole_dof_idx]) > math.pi / 2, dim=1
        )

        self.extras["terminated_mask"] = out_of_bounds
        reset_terminated = out_of_bounds
        return reset_terminated, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        """Reset the given environments to their default state with a randomized pole angle."""
        if env_ids is None:
            env_ids = self.cartspole._ALL_INDICES

        super()._reset_idx(env_ids)

        joint_pos = self.cartspole.data.default_joint_pos[env_ids]
        # Add a random initial angle to the pole
        joint_pos[:, self._pole_dof_idx] += sample_uniform(
            self.cfg.initial_pole_angle_range[0] * math.pi,
            self.cfg.initial_pole_angle_range[1] * math.pi,
            joint_pos[:, self._pole_dof_idx].shape,
            joint_pos.device,
        )

        joint_vel = self.cartspole.data.default_joint_vel[env_ids]

        # default_root_state layout:
        # [0:3] translation (x, y, z), [3:7] orientation quaternion (w, x, y, z),
        # [7:10] linear velocity, [10:13] angular velocity
        default_root_state = self.cartspole.data.default_root_state[env_ids]
        default_root_state[:, :3] += self.scene.env_origins[env_ids]

        self.joint_pos[env_ids] = joint_pos
        self.joint_vel[env_ids] = joint_vel

        self.cartspole.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self.cartspole.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self.cartspole.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)


def compute_rewards(
    rew_scale_alive: float,
    rew_scale_terminated: float,
    rew_scale_pole_vel: float,
    rew_scale_pole_pos: float,
    rew_scale_cart_pos: float,
    rew_scale_cart_vel: float,
    rew_scale_cart1_pos: float,
    rew_scale_cart1_vel: float,
    rew_scale_spring_length: float,
    rew_scale_spring_vel: float,
    pole_pos: torch.Tensor,  # (N, 1)
    pole_vel: torch.Tensor,  # (N, 1)
    cart_pos: torch.Tensor,  # (N, 1)
    cart_vel: torch.Tensor,  # (N, 1)
    cart1_pos: torch.Tensor,  # (N, 1)
    cart1_vel: torch.Tensor,  # (N, 1)
    spring_length: torch.Tensor,  # (N, 1)
    spring_velocity: torch.Tensor,  # (N, 1)
    spring_original_length: torch.Tensor,  # scalar or (N, 1)
    reset_terminated: torch.Tensor,  # (N,) or (N, 1)
    actions: torch.Tensor,  # (N, 1) or (N,)
) -> torch.Tensor:
    """Dense reward: upright bonus + survival, minus bounded (tanh) penalties.

    The penalties are squashed with ``tanh`` so each term saturates and stays
    well-scaled regardless of how large the raw quantity grows.
    """
    # Core objective: keep the pole upright (weight 3)
    pole_reward = torch.cos(pole_pos) * 3.0

    # Survival bonus (weight 0.5)
    survival = 0.5

    # Other penalties (total weight < 1.5)
    penalties = (
        -0.5 * torch.tanh(pole_vel / 5.0) ** 2
        + -0.3 * torch.tanh(cart1_pos / 6.0) ** 2
        + -0.3 * torch.tanh(cart_pos / 6.0) ** 2
        + -0.2 * torch.tanh((spring_length - 1.0) / 0.3) ** 2
        + -0.1 * torch.tanh(spring_velocity / 2.0) ** 2
        + -0.05 * torch.tanh(actions / 200.0) ** 2
    )

    # No termination penalty
    return pole_reward + survival + penalties
