"""On-policy transitions storage for PPO and other on-policy algorithms.

Provides a storage class following the cleanRL PPO pattern for collecting
observations, actions, log-probabilities, rewards, dones, and values over
multiple timesteps and parallel environments.
"""

import torch
import numpy as np
from collections import OrderedDict
from typing import Tuple, Union, Optional


class OnPolicyTransitions:
    """Storage for on-policy experience collection (PPO-style).

    This class manages experience buffers for on-policy RL algorithms, storing:
    - observations (obs)
    - actions
    - log-probabilities (logprobs)
    - rewards
    - dones
    - values

    Shape conventions (following cleanRL PPO):
    - observations: (num_steps, num_envs) + obs_shape
    - actions: (num_steps, num_envs) + action_shape
    - logprobs: (num_steps, num_envs)
    - rewards: (num_steps, num_envs)
    - dones: (num_steps, num_envs)
    - values: (num_steps, num_envs)

    Example:
        >>> import torch
        >>> storage = OnPolicyTransitions(
        ...     num_steps=2048,
        ...     num_envs=4,
        ...     obs_shape=(17,),
        ...     action_shape=(6,),
        ...     device='cuda'
        ... )
        >>> # Fill in data step by step
        >>> for step in range(2048):
        ...     storage.obs[step] = obs_tensor  # shape (4, 17)
        ...     storage.actions[step] = action_tensor  # shape (4, 6)
        ...     storage.logprobs[step] = logprob_tensor  # shape (4,)
        ...     storage.rewards[step] = reward_tensor  # shape (4,)
        ...     storage.dones[step] = done_tensor  # shape (4,)
        ...     storage.values[step] = value_tensor  # shape (4,)
        >>> # Flatten for batch processing
        >>> b_obs = storage.get_flat_obs()
        >>> b_actions = storage.get_flat_actions()
    """

    def __init__(
        self,
        num_steps: int,
        num_envs: int,
        obs_shape: Union[Tuple[int, ...], int],
        action_shape: Union[Tuple[int, ...], int],
        device: Union[str, torch.device] = "cpu",
    ):
        """Initialize on-policy storage buffers.

        Args:
            num_steps: Number of rollout steps per batch.
            num_envs: Number of parallel environments.
            obs_shape: Shape of observation space (without batch dimensions).
            action_shape: Shape of action space (without batch dimensions).
            device: Device to place tensors on ('cpu', 'cuda', etc.).
        """
        self.num_steps = num_steps
        self.num_envs = num_envs
        # normalize shapes to tuple form so callers may pass int or tuple
        if isinstance(obs_shape, int):
            self.obs_shape = (obs_shape,)
        else:
            self.obs_shape = tuple(obs_shape)

        if isinstance(action_shape, int):
            self.action_shape = (action_shape,)
        else:
            self.action_shape = tuple(action_shape)
        self.device = device

        # Initialize storage buffers
        self.obs = torch.zeros((num_steps, num_envs) + self.obs_shape, dtype=torch.float32, device=device)
        self.actions = torch.zeros((num_steps, num_envs) + self.action_shape, dtype=torch.float32, device=device)
        self.logprobs = torch.zeros((num_steps, num_envs), dtype=torch.float32, device=device)
        self.rewards = torch.zeros((num_steps, num_envs), dtype=torch.float32, device=device)
        self.dones = torch.zeros((num_steps, num_envs), dtype=torch.float32, device=device)
        self.values = torch.zeros((num_steps, num_envs), dtype=torch.float32, device=device)

    def to(self, device: Union[str, torch.device]) -> "OnPolicyTransitions":
        """Move all buffers to a specified device.

        Args:
            device: Target device.

        Returns:
            Self, with all tensors moved to the device.
        """
        self.device = device
        self.obs = self.obs.to(device)
        self.actions = self.actions.to(device)
        self.logprobs = self.logprobs.to(device)
        self.rewards = self.rewards.to(device)
        self.dones = self.dones.to(device)
        self.values = self.values.to(device)
        return self

    def get_flat_obs(self) -> torch.Tensor:
        """Flatten observations for batch processing.

        Returns:
            Flattened observations of shape (num_steps * num_envs, *obs_shape).
        """
        return self.obs.reshape((-1,) + self.obs_shape)

    def get_flat_actions(self) -> torch.Tensor:
        """Flatten actions for batch processing.

        Returns:
            Flattened actions of shape (num_steps * num_envs, *action_shape).
        """
        return self.actions.reshape((-1,) + self.action_shape)

    def get_flat_logprobs(self) -> torch.Tensor:
        """Flatten logprobs for batch processing.

        Returns:
            Flattened logprobs of shape (num_steps * num_envs,).
        """
        return self.logprobs.reshape(-1)

    def get_flat_rewards(self) -> torch.Tensor:
        """Flatten rewards for batch processing.

        Returns:
            Flattened rewards of shape (num_steps * num_envs,).
        """
        return self.rewards.reshape(-1)

    def get_flat_dones(self) -> torch.Tensor:
        """Flatten dones for batch processing.

        Returns:
            Flattened dones of shape (num_steps * num_envs,).
        """
        return self.dones.reshape(-1)

    def get_flat_values(self) -> torch.Tensor:
        """Flatten values for batch processing.

        Returns:
            Flattened values of shape (num_steps * num_envs,).
        """
        return self.values.reshape(-1)

    def get_flat_batch(self) -> dict:
        """Get all flattened tensors as a dictionary for convenient batch processing.

        Returns:
            Dictionary with keys: 'obs', 'actions', 'logprobs', 'rewards', 'dones', 'values'.
        """
        flat_obs = self.get_flat_obs()
        flat_actions = self.get_flat_actions()
        flat_logprobs = self.get_flat_logprobs()
        flat_rewards = self.get_flat_rewards()
        flat_dones = self.get_flat_dones()
        flat_values = self.get_flat_values()

        # compute next observations (s2) by shifting in time; last step repeats last obs
        if self.num_steps > 1:
            next_obs = torch.zeros_like(self.obs)
            next_obs[:-1] = self.obs[1:]
            next_obs[-1] = self.obs[-1]
        else:
            next_obs = self.obs.clone()
        flat_next_obs = next_obs.reshape((-1,) + self.obs_shape)

        # placeholder next actions (a2) - zeros (PPO training here doesn't need a2)
        flat_next_actions = torch.zeros_like(flat_actions)

        flat_dsc = 1.0 - flat_dones

        return OrderedDict(
            [
                ("obs", flat_obs),
                ("actions", flat_actions),
                ("logprobs", flat_logprobs),
                ("rewards", flat_rewards),
                ("dones", flat_dones),
                ("values", self.values),
                ("s1", self.obs),
                ("a1", self.actions),
                ("s2", flat_next_obs),
                ("a2", flat_next_actions),
                ("reward", self.rewards),
                ("done", self.dones),
                ("dsc", flat_dsc),
            ]
        )

    def clear(self) -> None:
        """Reset all buffers to zero (useful for sequential collection)."""
        self.obs.zero_()
        self.actions.zero_()
        self.logprobs.zero_()
        self.rewards.zero_()
        self.dones.zero_()
        self.values.zero_()

    def __repr__(self) -> str:
        return (
            f"OnPolicyTransitions(\n"
            f"  num_steps={self.num_steps}, num_envs={self.num_envs},\n"
            f"  obs_shape={self.obs_shape}, action_shape={self.action_shape},\n"
            f"  device={self.device}\n"
            f")"
        )
