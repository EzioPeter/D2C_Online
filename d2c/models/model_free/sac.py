"""
Implementation of SAC (Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor)
Paper: https://arxiv.org/abs/1801.01290.pdf
"""
import collections

import copy
from ml_collections import ConfigDict

import torch
from tqdm import trange
import numpy as np
import torch.nn.functional as F
from torch import nn, Tensor
from typing import Union, Tuple, Any, Sequence, Dict, Iterator
from d2c.models.base import BaseAgent, BaseAgentModule
from d2c.utils import networks, utils, policies
from d2c.utils.offpolicyreplaybuffer import ReplayBuffer
from d2c.networks_and_utils_for_agent.sac_nets_utils import ActorNetwork, CriticNetwork

class SACAgent(BaseAgent):
    """
    SAC Agent for online reinforcement learning.
    """
    def __init__(
            self,
            update_actor_freq: int = 2,
            rollout_sim_freq: int = 1000,
            rollout_sim_num: int = 1000,            
            reward_scale: float = 1.0,
            alpha_multiplier: float = 1.0,
            alpha_init_value: float = 0.2,
            automatic_entropy_tuning: bool = True,
            log_alpha_init_value: float = 0.0,
            backup_entropy: bool = True,
            target_entropy: float = 0.0,
            target_update_period: int = 1,
            batch_size: int = 256,
            joint_noise_std: float = 0.0,
            max_traj_length: int = 1000,
            env_seed: int = 42,
            num_envs: int = 1,
            buffer_size: int = 1000000,
            learning_starts: int = 5000,
            **kwargs: Any,
    ) -> None:
        self._update_actor_freq = update_actor_freq
        self._rollout_sim_freq = rollout_sim_freq
        self._rollout_sim_num = rollout_sim_num
        self._reward_scale = reward_scale
        self._alpha_multiplier = alpha_multiplier
        self._alpha_init_value = alpha_init_value
        self._automatic_entropy_tuning = automatic_entropy_tuning
        self._log_alpha_init_value = log_alpha_init_value
        self._backup_entropy = backup_entropy
        self._target_entropy = target_entropy
        self._target_update_period = target_update_period
        self._joint_noise_std = joint_noise_std
        self._max_traj_length = max_traj_length
        self._num_envs = num_envs
        self._buffer_size = buffer_size
        self._learning_starts = learning_starts
        self._p_info = collections.OrderedDict()
        super(SACAgent, self).__init__(**kwargs)
        self._batch_size = batch_size
        self._env_seed = env_seed
        self._target_entropy = -np.prod(self._action_space.shape[0]).item()

    def _get_modules(self) -> utils.Flags:
        model_params_q, n_q_fns = self._model_params.q
        model_params_p = self._model_params.p[0]

        def q_net_factory():
            return CriticNetwork(
                observation_space=self._observation_space,
                action_space=self._action_space,
                fc_layer_params=model_params_q,
                device=self._device,
            )

        def p_net_factory():
            return ActorNetwork(
                observation_space=self._observation_space,
                action_space=self._action_space,
                fc_layer_params=model_params_p,
                device=self._device,
            )

        def log_alpha_net_factory():
            log_alpha = torch.zeros(1, requires_grad=True, device=self._device)
            return log_alpha

        modules = utils.Flags(
            p_net_factory=p_net_factory,
            q_net_factory=q_net_factory,
            n_q_fns=n_q_fns,
            log_alpha_net_factory=log_alpha_net_factory,
            device=self._device,
            automatic_entropy_tuning=self._automatic_entropy_tuning,
        )

        return modules
    
    def _build_fns(self) -> None:
        self._agent_module = AgentModule(modules=self._modules)
        self._q_fns = self._agent_module.q_nets
        self._q_target_fns = self._agent_module.q_target_nets
        self._p_fn = self._agent_module.p_net
        self._p_target_fn = self._agent_module.p_target_net
        if self._automatic_entropy_tuning:
            self._log_alpha_fn = self._agent_module.log_alpha_net

    def _init_vars(self) -> None:
        self._observation_space = self._env._env.single_observation_space
        self._action_space = self._env._env.single_action_space
        self._train_data = ReplayBuffer(
            buffer_size=self._buffer_size,
            observation_space=self._observation_space,
            action_space=self._action_space,
            device=self._device,
            n_envs=self._num_envs,
            handle_timeout_termination=False,
        )
        self._current_state = None
        self._observation_space.dtype = np.float32

    def _build_optimizers(self) -> None:
        opts = self._optimizers
        self._q_optimizer = utils.get_optimizer(opts.q[0])(
            parameters=list(self._q_fns[0].parameters())+list(self._q_fns[1].parameters()),
            lr=opts.q[1],
            weight_decay=self._weight_decays,
        )
        self._p_optimizer = utils.get_optimizer(opts.p[0])(
            parameters=self._p_fn.parameters(),
            lr=opts.p[1],
            weight_decay=self._weight_decays,
        )
        if self._automatic_entropy_tuning:
            self._alpha_optimizer = utils.get_optimizer(opts.alpha[0])(
                parameters=[self._log_alpha_fn],
                lr=opts.alpha[1],
                weight_decay=self._weight_decays,
            )
        else:
            self._alpha = self._alpha_init_value
    
    def _build_alpha_loss(self, batch: Dict) -> Tuple:
        states = batch['s1']

        with torch.no_grad():
            _, log_pi, _ = self._p_fn(states)
        alpha_loss = (-self._log_alpha_fn.exp() * (log_pi + self._target_entropy)).mean()
        self._alpha = self._log_alpha_fn.exp() * self._alpha_multiplier

        info = collections.OrderedDict()
        info['alpha'] = self._alpha
        if self._automatic_entropy_tuning:
            info['alpha_loss'] = alpha_loss
            return alpha_loss, info
        else:
            return 0, info

    def _build_q_loss(self, batch: Dict) -> Tuple[Tensor, Dict]:
        states = batch['s1']
        actions = batch['a1']
        rewards = batch['reward']
        next_states = batch['s2']
        dones = batch['done']        

        with torch.no_grad():
            next_state_actions, next_state_log_pi, _ = self._p_fn(next_states)
            qf1_next_target = self._q_target_fns[0](next_states, next_state_actions)
            qf2_next_target = self._q_target_fns[1](next_states, next_state_actions)
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self._log_alpha_fn.exp() * next_state_log_pi
            next_q_value = rewards.flatten() + (1 - dones.flatten()) * self._discount * (min_qf_next_target).view(-1)


        qf1_a_values = self._q_fns[0](states, actions).view(-1)
        qf2_a_values = self._q_fns[1](states, actions).view(-1)
        qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
        qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
        qf_loss = qf1_loss + qf2_loss

        info = collections.OrderedDict()
        info['Q1_loss'] = qf1_loss.detach().mean()
        info['Q2_loss'] = qf2_loss.detach().mean()
        info['Q_loss'] = qf_loss.detach().mean()
        info['average_qf1'] = qf1_a_values.detach().mean()
        info['average_qf2'] = qf2_a_values.detach().mean()
        info['average_target_q'] = min_qf_next_target.detach().mean()
        
        return qf_loss, info
    
    def _build_p_loss(self, batch: Dict) -> Tuple[Tensor, Dict]:
        states = batch['s1']

        pi, log_pi, _ = self._p_fn(states)
        qf1_pi = self._q_fns[0](states, pi)
        qf2_pi = self._q_fns[1](states, pi)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)
        p_loss = ((self._log_alpha_fn.exp() * log_pi) - min_qf_pi).mean()

        info = collections.OrderedDict()
        info['actor_loss'] = p_loss.detach().mean()
        info['log_pi'] = log_pi.detach().mean()
        return p_loss, info

    def _get_train_batch(self) -> Dict:
        obs = self._current_state
        if self._global_step < self._learning_starts:
            actions = np.array([self._action_space.sample() for _ in range(self._num_envs)])
        else:
            actions, _, _ = self._p_fn(torch.Tensor(obs).to(self._device))
            actions = actions.detach().cpu().numpy()
        next_obs, rewards, terminations, truncations, infos = self._env.step(actions)

        real_next_obs = next_obs.copy()
        for idx, trunc in enumerate(truncations):
            if trunc:
                real_next_obs[idx] = infos["final_observation"][idx]
        self._train_data.add(obs, real_next_obs, actions, rewards, terminations, infos)

        self._current_state = next_obs

        batch = None

        if self._global_step > self._learning_starts:
            batch = self._train_data.sample(self._batch_size)
            batch = batch._samples_to_dict()

        return batch

    def _optimize_step(self, batch: Dict) -> Dict:
        info = collections.OrderedDict()

        if self._global_step > self._learning_starts:
            q_loss, q_info = self._build_q_loss(batch)
            self._q_optimizer.zero_grad()
            q_loss.backward()
            self._q_optimizer.step()

            info.update(q_info)

            if self._global_step % self._update_actor_freq == 0:  # TD 3 Delayed update support
                for _ in range(
                    self._update_actor_freq
                ):  # compensate for the delay by doing 'actor_update_interval' instead of 1
                    p_loss, p_info = self._build_p_loss(batch)

                    self._p_optimizer.zero_grad()
                    p_loss.backward()
                    self._p_optimizer.step()
                    info.update(p_info)   

                    if self._automatic_entropy_tuning:
                        alpha_loss, alpha_info = self._build_alpha_loss(batch)

                        self._alpha_optimizer.zero_grad()
                        alpha_loss.backward()
                        self._alpha_optimizer.step()

                        info.update(alpha_info)

            if self._global_step % self._target_update_period == 0:
                self._update_target_fns(self._q_fns, self._q_target_fns)
                self._update_target_fns(self._p_fn, self._p_target_fn)  

        return info
    
    def _build_test_policies(self) -> None:
        self._test_policies['main'] = self._p_fn
    
    def save(self, ckpt_name: str) -> None:
        pass

    def restore(self, ckpt_name: str) -> None:
        pass


class AgentModule(BaseAgentModule):
    def _build_modules(self) -> None:
        device = self._net_modules.device
        automatic_entropy_tuning = self._net_modules.automatic_entropy_tuning
        self._p_net = self._net_modules.p_net_factory().to(device)
        self._q_nets = nn.ModuleList()
        self._q_target_nets = nn.ModuleList()
        n_q_fns = self._net_modules.n_q_fns
        for _ in range(n_q_fns):
            self._q_nets.append(self._net_modules.q_net_factory().to(device))
        for i in range(n_q_fns):
            self._q_target_nets.append(self._net_modules.q_net_factory().to(device))
            self._q_target_nets[i].load_state_dict(self._q_nets[i].state_dict())
        self._p_target_net = self._net_modules.p_net_factory().to(device)
        self._p_target_net.load_state_dict(self._p_net.state_dict())
        if automatic_entropy_tuning:
            self._log_alpha_net = self._net_modules.log_alpha_net_factory().to(device)
        
    @property
    def q_nets(self) -> nn.ModuleList:
        return self._q_nets
    
    @property
    def q_target_nets(self) -> nn.ModuleList:
        return self._q_target_nets
    
    @property
    def p_net(self) -> nn.Module:
        return self._p_net
    
    @property
    def p_target_net(self) -> nn.Module:
        return self._p_target_net
    
    @property
    def log_alpha_net(self) -> nn.Module:
        return self._log_alpha_net