"""
Implementation of PPO (Proximal Policy Optimization)
Paper: https://arxiv.org/abs/1707.06347
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
from d2c.utils import networks, utils, policies, onpolicytransitions
from d2c.networks_and_utils_for_agent.ppo_nets_utils import ActorNetwork, CriticNetwork

class PPOAgent(BaseAgent):
    """
    PPO Agent for online reinforcement learning.
    """
    def __init__(
            self,
            rollout_sim_freq: int = 1000,
            rollout_sim_num: int = 1000,
            # joint_noise_std: float = 0.0,
            max_traj_length: int = 1000,
            env_seed: int = 42,
            total_timesteps: int = 1000000,
            num_envs: int = 1,
            num_steps: int = 2048,
            anneal_lr: bool = True,
            gae_lambda: float = 0.95,
            num_minibatches: int = 32,
            update_epochs: int = 10,
            norm_adv: bool = True,
            clip_coef: float = 0.2,
            clip_vloss: bool = True,
            ent_coef: float = 0.0,
            vf_coef: float = 0.5,
            max_grad_norm: float = 0.5,
            target_kl: float = None,
            # to be filled in runtime
            batch_size: int = 0,
            mini_batch_size: int = 0,
            num_iterations: int = 0,
            **kwargs: Any,
    ) -> None:
        self._rollout_sim_freq = rollout_sim_freq
        self._rollout_sim_num = rollout_sim_num
        # self._joint_noise_std = joint_noise_std
        self._max_traj_length = max_traj_length
        self._p_info = collections.OrderedDict()
        self._total_timesteps = total_timesteps
        self._num_envs = num_envs
        self._num_steps = num_steps
        self._anneal_lr = anneal_lr
        self._gae_lambda = gae_lambda
        self._num_minibatches = num_minibatches
        self._update_epochs = update_epochs
        self._norm_adv = norm_adv
        self._clip_coef = clip_coef
        self._clip_vloss = clip_vloss
        self._ent_coef = ent_coef
        self._vf_coef = vf_coef
        self._max_grad_norm = max_grad_norm
        self._target_kl = target_kl
        self._batch_size = batch_size
        self._mini_batch_size = mini_batch_size
        self._num_iterations = num_iterations
        self._env_seed = env_seed
        super(PPOAgent, self).__init__(**kwargs)
        
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

        modules = utils.Flags(
            p_net_factory=p_net_factory,
            q_net_factory=q_net_factory,
            n_q_fns=n_q_fns,
            device=self._device,
        )

        return modules
    
    def _build_agent(self) -> None:
        """Builds agent components."""
        self._init_vars()
        self._build_fns()
        self._build_optimizers()
        self._global_step = 0
        self._train_info = collections.OrderedDict()
        self._test_policies = collections.OrderedDict()
        self._build_test_policies()

    def _build_fns(self) -> None:
        self._agent_module = AgentModule(modules=self._modules)
        self._q_fns = self._agent_module.q_nets
        self._q_target_fns = self._agent_module.q_target_nets
        self._p_fn = self._agent_module.p_net
        self._p_target_fn = self._agent_module.p_target_net

    def _init_vars(self) -> None:
        self._batch_size = int(self._num_envs * self._num_steps)
        self._mini_batch_size = int(self._batch_size // self._num_minibatches)
        self._num_iterations = self._total_timesteps // self._batch_size
        self._observation_space = self._env._env.single_observation_space
        self._action_space = self._env._env.single_action_space
        self._train_data = onpolicytransitions.OnPolicyTransitions( 
            num_steps=self._num_steps,
            num_envs=self._num_envs,
            obs_shape=self._observation_space.shape,
            action_shape=self._action_space.shape,
            device=self._device,
        )
        self._current_iteration = 0
        self._total_iterations = 0
        self._next_obs = None
        self._next_dones = None

        self._episode_rewards = 0.0

    def _build_optimizers(self) -> None:
        opts = self._optimizers
        self._optimizer = utils.get_optimizer(opts.p[0])(
            parameters=list(list(self._q_fns[0].parameters()) + list(self._p_fn.parameters())),
            lr=opts.p[1],
            weight_decay=self._weight_decays,
        )
        self._optimizer.param_groups[0]['eps']=1e-5

    def _build_q_loss(self, batch: Dict) -> Tuple[Tensor, Dict]:
        states = batch['s1']
        actions = batch['a1']
        rewards = batch['reward']
        next_states = batch['s2']
        dsc = batch['dsc']
        
        qf1_pred = self._q_fns[0](states, actions)
        qf2_pred = self._q_fns[1](states, actions)

        new_next_actions, next_log_pi = self._p_fn(next_states)

        target_q_values = torch.min(
            self._q_target_fns[0](next_states, new_next_actions),
            self._q_target_fns[1](next_states, new_next_actions),
        )

        q_target = rewards + dsc * self._discount * target_q_values

        qf1_loss = F.mse_loss(qf1_pred, q_target.detach())
        qf2_loss = F.mse_loss(qf2_pred, q_target.detach())
        qf_loss = qf1_loss + qf2_loss

        info = collections.OrderedDict()
        info['Q1_loss'] = qf1_loss.detach().mean()
        info['Q2_loss'] = qf2_loss.detach().mean()
        info['Q_loss'] = qf_loss.detach().mean()
        info['average_qf1'] = qf1_pred.detach().mean()
        info['average_qf2'] = qf2_pred.detach().mean()
        info['average_target_q'] = target_q_values.detach().mean()
        
        return qf_loss, info
    
    def _build_p_loss(self, batch: Dict) -> Tuple[Tensor, Dict]:
        states = batch['s1']
        actions = batch['a1']
        rewards = batch['reward']
        next_states = batch['s2']
        dsc = batch['dsc']

        new_actions = self.new_actions
        log_pi = self.log_pi

        q_new_actions = torch.min(
            self._q_fns[0](states, new_actions),
            self._q_fns[1](states, new_actions),
        )
        p_loss = (self.alpha * log_pi - q_new_actions).mean()

        info = collections.OrderedDict()
        info['actor_loss'] = p_loss.detach().mean()
        info['log_pi'] = log_pi.detach().mean()
        return p_loss, info

    def train_step(self) -> None:
        train_batch = self._get_train_batch()
        info = self._optimize_step(train_batch)
        for key, val in info.items():
            self._train_info[key] = val.item()
        # self._global_step += self._num_envs * self._num_steps

    def _get_train_batch(self) -> Dict:        
        with torch.no_grad():
            # # self._current_state = self._env.reset(seed=self._env_seed) # use it for debug, gym
            # self._current_state, _ = self._env.reset(seed=self._env_seed) # use it for debug, gymnasium
            # # self._current_state = self._env.reset()
            # self._next_obs = self._current_state
            # self._next_obs = torch.Tensor(self._next_obs).to(self._device)
            # self._next_dones = torch.zeros(self._num_envs,).to(self._device)

            if self._anneal_lr:
                frac = 1.0 - (self._current_iteration - 1.0) / self._total_iterations
                lrnow = frac * self._optimizers.p[1]
                self._optimizer.param_groups[0]['lr'] = lrnow

            for step in range(0, self._num_steps):
                self._global_step += self._num_envs
                state = self._next_obs

                self._train_data.obs[step] = state
                self._train_data.dones[step] = self._next_dones

                action, logprob, _, = self._p_fn(state)
                value = self._q_fns[0](state)

                self._train_data.values[step] = value.flatten()
                self._train_data.actions[step] = action
                self._train_data.logprobs[step] = logprob

                # next_state, reward, done, info = self._env.step(action.cpu().numpy()) # gym
                next_state, reward, termination, truncation, infos = self._env.step(action.cpu().numpy()) # gymnasium
                done = np.logical_or(termination, truncation)

                self._next_dones = torch.Tensor(done).to(self._device)
                self._next_obs = torch.Tensor(next_state).to(self._device)

                self._train_data.rewards[step] = torch.Tensor(reward).to(self._device)

                if "final_info" in infos:
                    for info in infos["final_info"]:
                        if info and "episode" in info:
                            print(f"global_step={self._global_step}, episodic_return={info['episode']['r']}")

        batch = self._train_data.get_flat_batch()
        
        return batch

    def _optimize_step(self, batch: Dict) -> Dict:
        info = collections.OrderedDict()

        b_obs, b_logprobs, b_actions, b_advantages, b_returns, b_values = self.get_training_batch(batch)

        b_inds = np.arange(self._batch_size)
        clipfracs = []

        for epoch in range(self._update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, self._batch_size, self._mini_batch_size):
                end = start + self._mini_batch_size
                mb_inds = b_inds[start:end]

                _, newlogprobs, entropy = self._p_fn(b_obs[mb_inds], b_actions[mb_inds])
                newvalue = self._q_fns[0](b_obs[mb_inds])
                logratio = newlogprobs - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > self._clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if self._norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)
                
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - self._clip_coef, 1 +  self._clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                newvalue = newvalue.view(-1)
                if self._clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -self._clip_coef,
                        self._clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()

                loss = pg_loss - self._ent_coef * entropy_loss + self._vf_coef * v_loss

                self._optimizer.zero_grad()

                loss.backward()
                nn.utils.clip_grad_norm_(list(list(self._q_fns[0].parameters()) + list(self._p_fn.parameters())), self._max_grad_norm)
                self._optimizer.step()

            if self._target_kl is not None and old_approx_kl > self._target_kl:
                break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
        
        return info
    
    def _build_test_policies(self) -> None:
        policy = self._p_fn
        self._test_policies['main'] = policy
    
    def get_advantage(self, batch: Dict) -> Tensor:
        with torch.no_grad():
            next_values = self._q_fns[0](self._next_obs).reshape(1, -1)
            advantages = torch.zeros_like(batch['reward']).to(self._device)
            lastgaelam = 0
            for t in reversed(range(self._num_steps)):
                if t == self._num_steps - 1:
                    nextnonterminal = 1.0 - self._next_dones
                    nextvalues = next_values
                else:
                    nextnonterminal = 1.0 - batch['dones'][t + 1]
                    nextvalues = batch['values'][t + 1]
                delta = batch['reward'][t] + self._discount * nextvalues * nextnonterminal - batch['values'][t]
                advantages[t] = lastgaelam = delta + self._discount * self._gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + batch['values']
        return advantages, returns

    def get_training_batch(self, batch: Dict) -> Dict:
        training_batch_obs = batch['obs'].reshape((-1,) + self._observation_space.shape)
        training_batch_logprobs = batch['logprobs'].reshape(-1)
        training_batch_actions = batch['actions'].reshape((-1,) + self._action_space.shape)
        advantages, returns = self.get_advantage(batch)
        training_advantages = advantages.reshape(-1)
        training_returns = returns.reshape(-1)
        training_values = batch['values'].reshape(-1)
        return training_batch_obs, training_batch_logprobs, training_batch_actions, training_advantages, training_returns, training_values

    def save(self, ckpt_name: str) -> None:
        pass

    def restore(self, ckpt_name: str) -> None:
        pass


class AgentModule(BaseAgentModule):
    def _build_modules(self) -> None:
        device = self._net_modules.device
        self._q_nets = nn.ModuleList()
        n_q_fns = self._net_modules.n_q_fns
        for _ in range(n_q_fns):
            self._q_nets.append(self._net_modules.q_net_factory().to(device))
        self._p_net = self._net_modules.p_net_factory().to(device)
        self._q_target_nets = copy.deepcopy(self._q_nets)    
        self._p_target_net = copy.deepcopy(self._p_net)
        
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