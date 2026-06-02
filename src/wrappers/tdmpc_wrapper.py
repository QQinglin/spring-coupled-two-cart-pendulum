import numpy as np
import torch

from envs.carts_pole.cartspole_env import Carts2PoleEnv, Carts2PoleEnvCfg


class TDMPCEnvWrapper:
    """Adapt a Isaac Lab :class:`Carts2PoleEnv` to the API expected by the TD-MPC agent."""

    def __init__(self, env: Carts2PoleEnv, clip_actions: float | None = None):
        # env: the Isaac Lab DirectRLEnv to wrap
        # clip_actions: action clipping range; None means no clipping
        self.env = env
        self.common_step_counter = self.env.common_step_counter
        self.clip_actions = clip_actions

        # num_envs: number of parallel environments
        # max_episode_length: maximum episode length (in steps)
        self.num_envs = self.env.cfg.num_envs
        self.max_episode_length = env.max_episode_length

        # Action dimension
        self.num_actions = self.env.cfg.action_space

        # Policy observation dimension (num_envs, observation_space)
        self.num_obs = (self.env.cfg.num_envs, self.env.cfg.observation_space)

        # Reset at the start since the RSL-RL runner does not call reset
        self.env.reset()
        self.step_counter = 0

    def __str__(self):
        """Returns the wrapper name and the :attr:`env` representation string."""
        return f"<{type(self).__name__}{self.env}>"

    def __repr__(self):
        """Returns the string representation of the wrapper."""
        return str(self)

    """
    Properties -- Gym.Wrapper
    """

    @property
    def cfg(self) -> object:
        """Returns the configuration class instance of the environment."""
        return self.env.cfg

    @property
    def render_mode(self) -> str | None:
        """Returns the :attr:`Env` :attr:`render_mode`."""
        return self.env.render_mode

    @property
    def observation_space(self):
        """Returns the :attr:`Env` :attr:`observation_space`."""
        return self.env.cfg.observation_space

    @property
    def action_space(self):
        """Returns the :attr:`Env` :attr:`action_space`."""
        return self.env.cfg.action_space

    @classmethod
    def class_name(cls) -> str:
        """Returns the class name of the wrapper."""
        return cls.__name__

    """
    Properties
    """

    def get_observations(self) -> tuple[torch.Tensor, dict]:
        """Returns the current observations as (policy tensor, {"observations": full dict})."""
        obs_dict = self.env._get_observations()
        return obs_dict["policy"], {"observations": obs_dict}

    """
    Operations - MDP
    """

    def seed(self, seed: int = -1) -> int:  # noqa: D102
        return self.env.seed(seed)

    def reset(self) -> tuple[torch.Tensor, dict]:  # noqa: D102
        self.step_counter = 0
        obs_dict, _ = self.env.reset()
        return obs_dict["policy"], {"observations": obs_dict}

    def step(self, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        # Clip actions if a range is configured
        if self.clip_actions is not None:
            actions = torch.clamp(actions, -self.clip_actions, self.clip_actions)
        self.step_counter = self.step_counter + 1

        # Step the underlying env: obs, reward, terminated, truncated, extras
        obs_dict, rew, terminated, truncated, extras = self.env.step(actions)

        rew = rew.squeeze(-1)
        # Combine termination and time-out into a single done flag (RSL-RL compatibility)
        dones = (self.step_counter == self.max_episode_length | truncated).to(dtype=torch.long)

        # Move the policy observation out and keep the full dict in extras
        obs = obs_dict["policy"]
        extras["observations"] = obs_dict

        # Expose time-out info in extras (only needed for infinite-horizon tasks)
        if not self.env.cfg.is_finite_horizon:
            extras["time_outs"] = truncated

        return obs, rew, dones, extras

    def render(self, recompute: bool = False, **_):
        """Grab one RGB frame from the TiledCamera and return it as a uint8 (H, W, 3) array."""
        if recompute:
            self.env.sim.render()

        # Locate the tiled camera
        cam = getattr(self.env, "_tiled_camera", None)
        if cam is None and hasattr(self.env, "scene"):
            cam = self.env.scene.sensors.get("tiled_camera", None)
        if cam is None:
            raise RuntimeError("Tiled camera not found (expect _tiled_camera or scene.sensors['tiled_camera']).")

        # May be a torch.Tensor or numpy array, shape (1, H, W, C) or (H, W, C)
        frame = cam.data.output["rgb"]

        # Convert to numpy and drop the batch dimension
        if torch.is_tensor(frame):
            frame = frame.detach().to("cpu")
            if frame.dim() == 4 and frame.shape[0] == 1:
                frame = frame[0]  # (1, H, W, C) -> (H, W, C)
            frame = frame.numpy()
        else:
            frame = np.asarray(frame)
            if frame.ndim == 4 and frame.shape[0] == 1:
                frame = frame[0]

        # Drop alpha / extra channels
        if frame.ndim == 3 and frame.shape[2] > 3:
            frame = frame[:, :, :3]
        # Grayscale -> 3 channels (just in case)
        if frame.ndim == 2:
            frame = np.stack([frame, frame, frame], axis=-1)
        if frame.ndim == 3 and frame.shape[2] == 1:
            frame = np.repeat(frame, 3, axis=2)

        # Ensure uint8
        if frame.dtype != np.uint8:
            frame = (
                (np.clip(frame, 0.0, 1.0) * 255.0).astype(np.uint8)
                if frame.max() <= 1.0
                else np.clip(frame, 0, 255).astype(np.uint8)
            )

        return frame

    def close(self):  # noqa: D102
        return self.env.close()
