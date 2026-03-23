"""
Implementation of MBPO (Model-Based Policy Optimization).
Paper: https://arxiv.org/abs/1906.08253
"""
import collections
import copy
import os
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from d2c.envs import LeaEnv
from d2c.envs.learned.dynamics import make_dynamics
from d2c.models.base import BaseAgent, BaseAgentModule
from d2c.networks_and_utils_for_agent.sac_nets_utils import (
    ActorNetwork,
    CriticNetwork,
)
from d2c.utils import utils
from d2c.utils.replaybuffer import ReplayBuffer


class MBPOAgent(BaseAgent):
    """MBPO agent with online environment interaction and model rollouts."""

    def __init__(
            self,
            config: Any,
            update_actor_freq: int = 2,
            reward_scale: float = 1.0,
            alpha_multiplier: float = 1.0,
            alpha_init_value: float = 0.2,
            automatic_entropy_tuning: bool = True,
            backup_entropy: bool = True,
            target_entropy: Optional[float] = -1.0,
            target_update_period: int = 1,
            buffer_size: int = 1_000_000,
            model_buffer_size: int = 4_000,
            learning_starts: int = 5_000,
            model_train_freq: int = 250,
            model_train_steps: int = 250,
            rollout_freq: int = 250,
            rollout_batch_size: int = 1_000,
            rollout_schedule: Tuple[int, int, int, int] = (20_000, 150_000, 1, 1),
            rollout_batch_size_schedule: Optional[Tuple[int, int, int, int]] = None,
            model_buffer_retain_rollouts: int = 4,
            real_data_ratio: float = 0.0,
            real_data_ratio_schedule: Optional[Tuple[int, int, float, float]] = None,
            num_sac_updates_per_step: int = 10,
            sac_updates_every_steps: int = 1,
            alpha_min: float = 1e-4,
            alpha_max: float = 1.0,
            grad_clip_norm: Optional[float] = None,
            **kwargs: Any,
    ) -> None:
        self._config = config
        self._update_actor_freq = update_actor_freq
        self._reward_scale = reward_scale
        self._alpha_multiplier = alpha_multiplier
        self._alpha_init_value = alpha_init_value
        self._automatic_entropy_tuning = automatic_entropy_tuning
        self._backup_entropy = backup_entropy
        self._target_entropy = target_entropy
        self._target_update_period = target_update_period
        self._buffer_size = buffer_size
        self._model_buffer_size = model_buffer_size
        self._learning_starts = learning_starts
        self._model_train_freq = model_train_freq
        self._model_train_steps = model_train_steps
        self._rollout_freq = rollout_freq
        self._rollout_batch_size = rollout_batch_size
        self._rollout_schedule = rollout_schedule
        self._rollout_batch_size_schedule = rollout_batch_size_schedule
        self._model_buffer_retain_rollouts = model_buffer_retain_rollouts
        self._real_data_ratio = real_data_ratio
        self._real_data_ratio_schedule = real_data_ratio_schedule
        self._num_sac_updates_per_step = num_sac_updates_per_step
        self._sac_updates_every_steps = sac_updates_every_steps
        self._alpha_min = alpha_min
        self._alpha_max = alpha_max
        self._grad_clip_norm = grad_clip_norm
        self._step_info = collections.OrderedDict()
        super(MBPOAgent, self).__init__(**kwargs)
        self._target_entropy = -float(self._action_space.shape[0])


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
            return torch.zeros(1, device=self._device, dtype=torch.float32)

        return utils.Flags(
            p_net_factory=p_net_factory,
            q_net_factory=q_net_factory,
            n_q_fns=n_q_fns,
            log_alpha_net_factory=log_alpha_net_factory,
            device=self._device,
            automatic_entropy_tuning=self._automatic_entropy_tuning,
        )

    def _build_fns(self) -> None:
        self._agent_module = AgentModule(modules=self._modules)
        self._q_fns = self._agent_module.q_nets
        self._q_target_fns = self._agent_module.q_target_nets
        self._p_fn = self._agent_module.p_net
        self._p_target_fn = self._agent_module.p_target_net
        if self._automatic_entropy_tuning:
            self._log_alpha_fn = self._agent_module.log_alpha_net

    def _init_vars(self) -> None:
        if hasattr(self._env, '_env') and hasattr(self._env._env, 'single_observation_space'):
            self._observation_space = self._env._env.single_observation_space
            self._action_space = self._env._env.single_action_space
        else:
            self._observation_space = self._env.observation_space
            self._action_space = self._env.action_space
        self._observation_space.dtype = np.float32

        self._real_buffer = ReplayBuffer(
            state_dim=self._observation_space.shape[0],
            action_dim=self._action_space.shape[0],
            max_size=self._buffer_size,
            device=self._device,
        )
        self._min_model_buffer_size = max(int(self._model_buffer_size), 1)
        self._model_buffer = ReplayBuffer(
            state_dim=self._observation_space.shape[0],
            action_dim=self._action_space.shape[0],
            max_size=self._min_model_buffer_size,
            device=self._device,
        )
        self._model_env = LeaEnv(self._config)
        self._dynamics = None
        self._current_state = None
        self._sac_update_step = 0
        self._last_model_train_step = 0
        self._last_rollout_step = 0
        self._rollout_length = int(self._rollout_schedule[2])
        self._current_rollout_batch_size = int(self._rollout_batch_size)
        self._current_real_data_ratio = float(self._real_data_ratio)

    def _build_optimizers(self) -> None:
        opts = self._optimizers
        self._q_optimizer = utils.get_optimizer(opts.q[0])(
            parameters=self._q_fns.parameters(),
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
            self._alpha = torch.tensor(
                self._alpha_init_value,
                device=self._device,
                dtype=torch.float32,
            )

    def _get_alpha(self) -> Tensor:
        if self._automatic_entropy_tuning:
            multiplier = max(self._alpha_multiplier, 1e-8)
            min_log_alpha = np.log(max(self._alpha_min / multiplier, 1e-8))
            max_log_alpha = np.log(max(self._alpha_max / multiplier, 1e-8))
            return torch.clamp(self._log_alpha_fn, min=min_log_alpha, max=max_log_alpha).exp() * multiplier
        return self._alpha

    def _build_alpha_loss(self, batch: Dict) -> Tuple[Tensor, Dict]:
        states = batch['s1']
        with torch.no_grad():
            _, log_pi, _ = self._p_fn(states)
            log_pi = log_pi.view(-1)
        alpha_loss = -(self._log_alpha_fn * (log_pi + self._target_entropy)).mean()

        info = collections.OrderedDict()
        info['alpha'] = self._get_alpha().detach()
        info['alpha_loss'] = alpha_loss.detach()
        return alpha_loss, info

    def _build_q_loss(self, batch: Dict) -> Tuple[Tensor, Dict]:
        states = batch['s1']
        actions = batch['a1']
        rewards = batch['reward'].view(-1)
        next_states = batch['s2']
        dones = batch['done'].view(-1)
        alpha = self._get_alpha().detach()

        with torch.no_grad():
            next_actions, next_log_pi, _ = self._p_fn(next_states)
            next_log_pi = next_log_pi.view(-1)
            target_q1 = self._q_target_fns[0](next_states, next_actions).view(-1)
            target_q2 = self._q_target_fns[1](next_states, next_actions).view(-1)
            target_q = torch.minimum(target_q1, target_q2)
            if self._backup_entropy:
                target_q = target_q - alpha * next_log_pi
            td_target = self._reward_scale * rewards + (1.0 - dones) * self._discount * target_q

        q1_pred = self._q_fns[0](states, actions).view(-1)
        q2_pred = self._q_fns[1](states, actions).view(-1)
        q1_loss = F.mse_loss(q1_pred, td_target)
        q2_loss = F.mse_loss(q2_pred, td_target)
        q_loss = q1_loss + q2_loss

        info = collections.OrderedDict()
        info['Q1'] = q1_pred.detach().mean()
        info['Q2'] = q2_pred.detach().mean()
        info['Q_target'] = td_target.detach().mean()
        info['Q1_loss'] = q1_loss.detach()
        info['Q2_loss'] = q2_loss.detach()
        info['Q_loss'] = q_loss.detach()
        return q_loss, info

    def _build_p_loss(self, batch: Dict) -> Tuple[Tensor, Dict]:
        states = batch['s1']
        alpha = self._get_alpha().detach()

        actions, log_pi, _ = self._p_fn(states)
        log_pi = log_pi.view(-1)
        q1_pi = self._q_fns[0](states, actions).view(-1)
        q2_pi = self._q_fns[1](states, actions).view(-1)
        q_pi = torch.minimum(q1_pi, q2_pi)
        p_loss = (alpha * log_pi - q_pi).mean()

        info = collections.OrderedDict()
        info['actor_loss'] = p_loss.detach()
        info['log_pi'] = log_pi.detach().mean()
        info['Q_in_actor_loss'] = q_pi.detach().mean()
        return p_loss, info

    def _optimize_q(self, batch: Dict) -> Dict:
        q_loss, info = self._build_q_loss(batch)
        self._q_optimizer.zero_grad()
        q_loss.backward()
        if self._grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(self._q_fns.parameters(), self._grad_clip_norm)
        self._q_optimizer.step()
        return info

    def _optimize_p(self, batch: Dict) -> Dict:
        p_loss, info = self._build_p_loss(batch)
        self._p_optimizer.zero_grad()
        p_loss.backward()
        if self._grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(self._p_fn.parameters(), self._grad_clip_norm)
        self._p_optimizer.step()
        return info

    def _optimize_alpha(self, batch: Dict) -> Dict:
        alpha_loss, info = self._build_alpha_loss(batch)
        self._alpha_optimizer.zero_grad()
        alpha_loss.backward()
        if self._grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_([self._log_alpha_fn], self._grad_clip_norm)
        self._alpha_optimizer.step()
        with torch.no_grad():
            multiplier = max(self._alpha_multiplier, 1e-8)
            min_log_alpha = np.log(max(self._alpha_min / multiplier, 1e-8))
            max_log_alpha = np.log(max(self._alpha_max / multiplier, 1e-8))
            self._log_alpha_fn.clamp_(min=min_log_alpha, max=max_log_alpha)
        info['alpha'] = self._get_alpha().detach()
        return info

    def _sample_action(self, observation: np.ndarray) -> np.ndarray:
        if self._real_buffer.size < self._learning_starts:
            return np.stack([self._action_space.sample() for _ in range(observation.shape[0])]).astype(np.float32)
        obs_tensor = torch.as_tensor(observation, device=self._device, dtype=torch.float32)
        with torch.no_grad():
            actions, _, _ = self._p_fn(obs_tensor)
        return actions.cpu().numpy().astype(np.float32)

    @staticmethod
    def _extract_terminal_observation(
            infos: Dict,
            index: int,
            fallback: np.ndarray,
    ) -> np.ndarray:
        if not isinstance(infos, dict):
            return fallback
        for key in ('final_observation', 'final_obs'):
            if key in infos and infos[key][index] is not None:
                return np.asarray(infos[key][index], dtype=np.float32)
        return fallback

    def _collect_real_transition(self) -> None:
        observation = np.asarray(self._current_state, dtype=np.float32)
        if observation.ndim == 1:
            observation = observation.reshape(1, -1)
        action = self._sample_action(observation)
        next_obs, reward, terminated, truncated, infos = self._env.step(action)

        next_obs = np.asarray(next_obs, dtype=np.float32)
        reward = np.asarray(reward, dtype=np.float32).reshape(-1)
        terminated = np.asarray(terminated, dtype=np.float32).reshape(-1)
        truncated = np.asarray(truncated, dtype=np.float32).reshape(-1)
        done = np.logical_or(terminated > 0.0, truncated > 0.0).astype(np.float32)

        for idx in range(observation.shape[0]):
            terminal_next_obs = self._extract_terminal_observation(
                infos=infos,
                index=idx,
                fallback=next_obs[idx],
            )
            self._real_buffer.add(
                state=observation[idx],
                action=action[idx],
                next_state=terminal_next_obs,
                next_action=np.zeros_like(action[idx], dtype=np.float32),
                reward=reward[idx],
                done=done[idx],
            )

        self._current_state = next_obs
        self._step_info['env_reward'] = torch.as_tensor(reward.mean(), device=self._device)
        self._step_info['real_buffer_size'] = torch.as_tensor(float(self._real_buffer.size), device=self._device)

    def _compute_rollout_length(self, env_step: int) -> int:
        min_step, max_step, min_length, max_length = self._rollout_schedule
        if env_step <= min_step:
            return int(min_length)
        if env_step >= max_step:
            return int(max_length)
        ratio = (env_step - min_step) / float(max_step - min_step)
        length = min_length + ratio * (max_length - min_length)
        return int(length)

    @staticmethod
    def _compute_scheduled_value(
            env_step: int,
            schedule: Optional[Tuple[int, int, float, float]],
            default: float,
            cast_type,
    ):
        if schedule is None:
            return cast_type(default)
        min_step, max_step, min_value, max_value = schedule
        if env_step <= min_step:
            return cast_type(min_value)
        if env_step >= max_step:
            return cast_type(max_value)
        ratio = (env_step - min_step) / float(max_step - min_step)
        value = min_value + ratio * (max_value - min_value)
        return cast_type(value)

    def _update_data_mix_schedule(self, env_step: int) -> None:
        self._current_rollout_batch_size = max(
            1,
            self._compute_scheduled_value(
                env_step=env_step,
                schedule=self._rollout_batch_size_schedule,
                default=self._rollout_batch_size,
                cast_type=int,
            ),
        )
        self._current_real_data_ratio = float(np.clip(
            self._compute_scheduled_value(
                env_step=env_step,
                schedule=self._real_data_ratio_schedule,
                default=self._real_data_ratio,
                cast_type=float,
            ),
            0.0,
            1.0,
        ))

    def _target_model_buffer_size(self, rollout_length: int) -> int:
        rollout_length = max(int(rollout_length), 1)
        retain_rollouts = max(int(self._model_buffer_retain_rollouts), 1)
        rollout_capacity = rollout_length * self._current_rollout_batch_size * retain_rollouts
        return max(self._min_model_buffer_size, rollout_capacity)

    def _resize_model_buffer(self, target_capacity: int) -> None:
        target_capacity = max(int(target_capacity), self._min_model_buffer_size)
        if target_capacity == self._model_buffer.capacity:
            return

        new_buffer = ReplayBuffer(
            state_dim=self._observation_space.shape[0],
            action_dim=self._action_space.shape[0],
            max_size=target_capacity,
            device=self._device,
        )
        if self._model_buffer.size > 0:
            capacity = self._model_buffer.capacity
            size = self._model_buffer.size
            ptr = self._model_buffer._ptr
            start = (ptr - size) % capacity
            indices = (start + np.arange(size)) % capacity
            if size > target_capacity:
                indices = indices[-target_capacity:]
            tensor_indices = torch.as_tensor(indices.astype(np.int64), device=self._device)
            batch = collections.OrderedDict(
                (k, torch.clone(v[tensor_indices]))
                for k, v in self._model_buffer.data.items()
            )
            new_buffer.add_transitions(
                state=batch['s1'],
                action=batch['a1'],
                next_state=batch['s2'],
                next_action=batch['a2'],
                reward=batch['reward'],
                done=batch['done'],
                cost=batch['cost'],
            )
        self._model_buffer = new_buffer

    def _refresh_dynamics_data(self) -> bool:
        if self._dynamics is None or self._real_buffer.size < 2:
            return False
        self._real_buffer._shuffle_indices = None
        self._dynamics._train_data = self._real_buffer
        self._dynamics._train_test_split()
        return len(self._dynamics._train_indices) > 0

    def _sync_model_env(self) -> None:
        self._model_env._dynamics_model = self._dynamics
        self._model_env._d_fns = self._dynamics.dynamics_fns

    def _maybe_train_dynamics(self) -> None:
        env_step = self._global_step + 1
        if self._real_buffer.size < self._learning_starts:
            return
        if env_step - self._last_model_train_step < self._model_train_freq:
            return

        if self._dynamics is None:
            self._dynamics = make_dynamics(config=self._config, data=self._real_buffer)
        if not self._refresh_dynamics_data():
            return

        for _ in range(self._model_train_steps):
            self._dynamics.train_step()
        if len(self._dynamics._test_indices) > 0:
            self._dynamics.test_step()
        self._sync_model_env()
        self._last_model_train_step = env_step

        self._step_info['dynamics_global_step'] = torch.as_tensor(
            float(self._dynamics.global_step),
            device=self._device,
        )
        for key, value in self._dynamics._train_info.items():
            self._step_info[f'dynamics_{key}'] = torch.as_tensor(value, device=self._device)

    @staticmethod
    def _pick_model_predictions(
            next_state_list,
            reward_list,
            done_list,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        next_states = np.stack(next_state_list, axis=0)
        rewards = np.stack(reward_list, axis=0)
        dones = np.stack(done_list, axis=0).astype(np.float32)
        batch_size = next_states.shape[1]
        model_indices = np.random.randint(next_states.shape[0], size=batch_size)
        batch_indices = np.arange(batch_size)
        return (
            next_states[model_indices, batch_indices].astype(np.float32),
            rewards[model_indices, batch_indices].astype(np.float32),
            dones[model_indices, batch_indices].astype(np.float32),
        )

    def _rollout_model(self) -> int:
        if self._dynamics is None or self._current_rollout_batch_size <= 0:
            return 0
        initial_batch = self._real_buffer.sample_batch(self._current_rollout_batch_size)
        states = initial_batch['s1']
        self._model_env.action_past = np.zeros(
            (states.shape[0], self._action_space.shape[0]),
            dtype=np.float32,
        )

        transitions_added = 0
        for _ in range(self._rollout_length):
            if states.shape[0] == 0:
                break
            with torch.no_grad():
                actions, _, _ = self._p_fn(states)
            next_state_list, reward_list, done_list = self._model_env.step_raw(states, actions)
            next_states, rewards, dones = self._pick_model_predictions(
                next_state_list=next_state_list,
                reward_list=reward_list,
                done_list=done_list,
            )
            actions_np = actions.detach().cpu().numpy().astype(np.float32)
            states_np = states.detach().cpu().numpy().astype(np.float32)
            batch_size = states_np.shape[0]
            self._model_buffer.add_transitions(
                state=states_np,
                action=actions_np,
                next_state=next_states,
                next_action=np.zeros_like(actions_np, dtype=np.float32),
                reward=rewards.reshape(batch_size),
                done=dones.reshape(batch_size),
            )
            transitions_added += batch_size

            alive = dones < 0.5
            if not np.any(alive):
                break
            states = torch.as_tensor(next_states[alive], device=self._device, dtype=torch.float32)
            self._model_env.action_past = self._model_env.action_past[alive]
        return transitions_added

    def _maybe_rollout_model(self) -> None:
        env_step = self._global_step + 1
        if self._dynamics is None or self._real_buffer.size < self._learning_starts:
            return
        if env_step - self._last_rollout_step < self._rollout_freq:
            return

        self._update_data_mix_schedule(env_step)
        self._rollout_length = self._compute_rollout_length(env_step)
        self._last_rollout_step = env_step
        if self._rollout_length <= 0:
            return

        self._resize_model_buffer(self._target_model_buffer_size(self._rollout_length))
        transitions_added = self._rollout_model()
        self._step_info['rollout_length'] = torch.as_tensor(float(self._rollout_length), device=self._device)
        self._step_info['model_rollout_transitions'] = torch.as_tensor(
            float(transitions_added),
            device=self._device,
        )
        self._step_info['rollout_batch_size'] = torch.as_tensor(
            float(self._current_rollout_batch_size),
            device=self._device,
        )
        self._step_info['model_buffer_size'] = torch.as_tensor(
            float(self._model_buffer.size),
            device=self._device,
        )
        self._step_info['model_buffer_capacity'] = torch.as_tensor(
            float(self._model_buffer.capacity),
            device=self._device,
        )

    def _sample_mixed_batch(self) -> Dict:
        if self._model_buffer.size == 0 or self._current_real_data_ratio >= 1.0:
            return self._real_buffer.sample_batch(self._batch_size)

        real_batch_size = int(round(self._batch_size * self._current_real_data_ratio))
        real_batch_size = min(max(real_batch_size, 1), self._batch_size)
        model_batch_size = self._batch_size - real_batch_size
        if model_batch_size <= 0:
            return self._real_buffer.sample_batch(self._batch_size)

        real_batch = self._real_buffer.sample_batch(real_batch_size)
        model_batch = self._model_buffer.sample_batch(model_batch_size)
        return collections.OrderedDict(
            (k, torch.cat([real_batch[k], model_batch[k]], dim=0))
            for k in real_batch.keys()
        )

    def _ready_for_updates(self) -> bool:
        return self._real_buffer.size >= max(self._learning_starts, self._batch_size)

    def _get_train_batch(self) -> Optional[Dict]:
        self._step_info = collections.OrderedDict()
        self._collect_real_transition()
        self._update_data_mix_schedule(self._global_step + 1)
        self._maybe_train_dynamics()
        self._maybe_rollout_model()
        self._step_info['real_data_ratio'] = torch.as_tensor(
            float(self._current_real_data_ratio),
            device=self._device,
        )

        if not self._ready_for_updates():
            self._step_info['model_buffer_size'] = torch.as_tensor(
                float(self._model_buffer.size),
                device=self._device,
            )
            return None
        return self._sample_mixed_batch()

    def _optimize_step(self, batch: Optional[Dict]) -> Dict:
        info = collections.OrderedDict(self._step_info)
        if batch is None:
            return info
        if (self._global_step + 1) % self._sac_updates_every_steps != 0:
            return info

        for update_idx in range(self._num_sac_updates_per_step):
            train_batch = batch if update_idx == 0 else self._sample_mixed_batch()
            q_info = self._optimize_q(train_batch)
            info.update(q_info)

            self._sac_update_step += 1
            if self._sac_update_step % self._update_actor_freq == 0:
                p_info = self._optimize_p(train_batch)
                info.update(p_info)
                if self._automatic_entropy_tuning:
                    alpha_info = self._optimize_alpha(train_batch)
                    info.update(alpha_info)

            if self._sac_update_step % self._target_update_period == 0:
                self._update_target_fns(self._q_fns, self._q_target_fns)
                self._update_target_fns(self._p_fn, self._p_target_fn)

        info['sac_update_step'] = torch.as_tensor(float(self._sac_update_step), device=self._device)
        info['model_buffer_size'] = torch.as_tensor(float(self._model_buffer.size), device=self._device)
        info['model_buffer_capacity'] = torch.as_tensor(float(self._model_buffer.capacity), device=self._device)
        return info

    def _build_test_policies(self) -> None:
        self._test_policies['main'] = self._p_fn

    def save(self, ckpt_name: str) -> None:
        torch.save(self._agent_module.state_dict(), ckpt_name + '.pth')
        torch.save(self._agent_module.p_net.state_dict(), ckpt_name + '_policy.pth')
        if self._dynamics is not None:
            self._dynamics.save(ckpt_name + '_dynamics')

    def restore(self, ckpt_name: str) -> None:
        self._agent_module.load_state_dict(
            torch.load(ckpt_name + '.pth', map_location=self._device, weights_only=True)
        )
        dynamics_ckpt = ckpt_name + '_dynamics.pth'
        if os.path.exists(dynamics_ckpt):
            if self._dynamics is None:
                self._dynamics = make_dynamics(config=self._config, data=None)
            self._dynamics.restore(ckpt_name + '_dynamics')
            self._sync_model_env()


class AgentModule(BaseAgentModule):
    """Container of trainable modules used by MBPO."""

    def _build_modules(self) -> None:
        device = self._net_modules.device
        self._q_nets = nn.ModuleList()
        n_q_fns = self._net_modules.n_q_fns
        for _ in range(n_q_fns):
            self._q_nets.append(self._net_modules.q_net_factory().to(device))
        self._q_target_nets = copy.deepcopy(self._q_nets)
        self._p_net = self._net_modules.p_net_factory().to(device)
        self._p_target_net = self._net_modules.p_net_factory().to(device)
        self._p_target_net.load_state_dict(self._p_net.state_dict())
        if self._net_modules.automatic_entropy_tuning:
            self._log_alpha_net = nn.Parameter(self._net_modules.log_alpha_net_factory())

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
    def log_alpha_net(self) -> nn.Parameter:
        return self._log_alpha_net
