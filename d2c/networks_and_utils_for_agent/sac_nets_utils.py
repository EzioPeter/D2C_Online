import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym
from torch.distributions import Normal
from torch.distributions.transformed_distribution import TransformedDistribution
from torch.distributions.transforms import TanhTransform
from typing import Tuple, List, Union, Type, Optional, Sequence
from gymnasium.spaces import Box, Space

ModuleType = Type[nn.Module]
LOG_STD_MIN = -5
LOG_STD_MAX = 2


def miniblock(
        input_size: int,
        output_size: int = 0,
        norm_layer: Optional[ModuleType] = None,
        activation: Optional[ModuleType] = None,
        linear_layer: Type[nn.Linear] = nn.Linear,
) -> List[nn.Module]:
    """Construct a miniblock with given input/output-size, norm layer and \
    activation."""
    layers: List[nn.Module] = [linear_layer(input_size, output_size)]
    if norm_layer is not None:
        layers += [norm_layer(output_size)]  # type: ignore
    if activation is not None:
        layers += [activation()]
    return layers

def sac_make_env(env_id, seed, idx, capture_video, run_name):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed)
        return env

    return thunk

# ALGO LOGIC: initialize agent here:
class CriticNetwork(nn.Module):
    def __init__(
            self,
            observation_space: Union[Box, Space],
            action_space: Union[Box, Space],
            fc_layer_params: Sequence[int] = (), 
            device: Union[str, int, torch.device] = "cpu",
    ):
        super().__init__()
        self.observation_dim = observation_space.shape[0]
        self.action_dim = action_space.shape[0]
        self._layers = []
        hidden_sizes = [self.observation_dim + self.action_dim] + list(fc_layer_params)
        for in_dim, out_dim in zip(hidden_sizes[:-1], hidden_sizes[1:]):
            self._layers += miniblock(in_dim, out_dim, activation=nn.ReLU)
        self._layers += [nn.Linear(hidden_sizes[-1], 1)]

        self.network = nn.Sequential(*self._layers)

    def forward(self, x, a):
        x = torch.cat([x, a], 1)
        x = self.network(x)
        return x


LOG_STD_MAX = 2
LOG_STD_MIN = -5


class ActorNetwork(nn.Module):
    def __init__(
            self,
            observation_space: Union[Box, Space],
            action_space: Union[Box, Space],
            fc_layer_params: Sequence[int] = (),
            device: Union[str, int, torch.device] = "cpu", 
    ):
        super().__init__()
        self.observation_dim = observation_space.shape[0]
        self.action_dim = action_space.shape[0]
        self._device = device
        self._layers = []
        hidden_sizes = [self.observation_dim] + list(fc_layer_params)
        for in_dim, out_dim in zip(hidden_sizes[:-1], hidden_sizes[1:]):
            self._layers += miniblock(in_dim, out_dim, activation=nn.ReLU)
        self._layers += [nn.Linear(hidden_sizes[-1], self.action_dim * 2)]

        self.network = nn.Sequential(*self._layers)
        # action rescaling
        self.register_buffer(
            "action_scale",
            torch.tensor(
                (action_space.high - action_space.low) / 2.0,
                dtype=torch.float32,
            ),
        )
        self.register_buffer(
            "action_bias",
            torch.tensor(
                (action_space.high + action_space.low) / 2.0,
                dtype=torch.float32,
            ),
        )

    def forward(self, x):
        output = self.network(x)
        mean, log_std = torch.chunk(output, 2, dim=-1)
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)  # From SpinUp / Denis Yarats

        return mean, log_std

    def get_action(self, x):
        mean, log_std = self(x)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean

