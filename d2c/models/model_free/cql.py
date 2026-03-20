"""
Implementation of CQL (Conservative Q-Learning for Offline Reinforcement Learning).
Paper: https://arxiv.org/abs/2006.04779
"""
import collections
import copy
import math
import torch
import torch.nn.functional as F
from torch import nn, Tensor
from typing import Tuple, Any, Dict, Optional
from d2c.models.base import BaseAgent, BaseAgentModule
from d2c.utils import networks, utils, policies


class CQLAgent(BaseAgent):
    """Implementation of Conservative Q-Learning.

    This implementation follows the repository's offline actor-critic pattern and
    uses a stochastic policy as in SAC, plus the conservative Q regularization term.

    :param int update_actor_freq: the update frequency of actor network.
    :param float alpha_multiplier: the multiplier of the entropy coefficient.
    :param float alpha_init_value: the initial value of the entropy coefficient.
    :param bool automatic_entropy_tuning: whether to tune the entropy coefficient.
    :param float target_entropy: the target entropy of the policy.
    :param bool backup_entropy: whether to use entropy-augmented Bellman backup.
    :param int target_update_period: the update frequency of target Q networks.
    :param int cql_n_actions: number of sampled actions per state for CQL regularization.
    :param float cql_temp: the temperature used in the log-sum-exp conservative loss.
    :param float cql_alpha: the weight of the conservative loss.
    :param int policy_bc_steps: the number of initial steps using behavior-cloning
        regularization in the policy update.
    :param float grad_clip_norm: gradient clipping threshold. Set to ``None`` to
        disable clipping.

    .. seealso::

        Please refer to :class:`~d2c.models.base.BaseAgent` for more detailed
        explanation.
    """

    def __init__(
            self,
            update_actor_freq: int = 2,
            alpha_multiplier: float = 1.0,
            alpha_init_value: float = 0.2,
            automatic_entropy_tuning: bool = True,
            target_entropy: float = 0.0,
            backup_entropy: bool = False,
            target_update_period: int = 1,
            cql_n_actions: int = 10,
            cql_temp: float = 1.0,
            cql_alpha: float = 5.0,
            cql_q_next_with_next_states: bool = True,
            policy_bc_steps: int = 0,
            actor_bc_weight: float = 0.0,
            grad_clip_norm: Optional[float] = 10.0,
            **kwargs: Any,
    ) -> None:
        self._update_actor_freq = update_actor_freq
        self._alpha_multiplier = alpha_multiplier
        self._alpha_init_value = alpha_init_value
        self._automatic_entropy_tuning = automatic_entropy_tuning
        self._target_entropy = target_entropy
        self._backup_entropy = backup_entropy
        self._target_update_period = target_update_period
        self._cql_n_actions = cql_n_actions
        self._cql_temp = cql_temp
        self._cql_alpha = cql_alpha
        self._cql_q_next_with_next_states = cql_q_next_with_next_states
        self._policy_bc_steps = policy_bc_steps
        self._actor_bc_weight = actor_bc_weight
        self._grad_clip_norm = grad_clip_norm
        self._p_info = collections.OrderedDict()
        super(CQLAgent, self).__init__(**kwargs)
        if self._target_entropy == 0.0:
            self._target_entropy = -float(self._a_dim)

    def _get_modules(self) -> utils.Flags:
        model_params_q, n_q_fns = self._model_params.q
        model_params_p = self._model_params.p[0]

        def q_net_factory():
            return networks.CriticNetwork(
                observation_space=self._observation_space,
                action_space=self._action_space,
                fc_layer_params=model_params_q,
                device=self._device,
            )

        def p_net_factory():
            return networks.ActorNetwork(
                observation_space=self._observation_space,
                action_space=self._action_space,
                fc_layer_params=model_params_p,
                device=self._device,
            )

        def log_alpha_net_factory():
            return torch.tensor(
                [math.log(self._alpha_init_value)],
                dtype=torch.float32,
                device=self._device,
            )

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
        if self._automatic_entropy_tuning:
            self._log_alpha_fn = self._agent_module.log_alpha_net

    def _init_vars(self) -> None:
        pass

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
                self._alpha_init_value * self._alpha_multiplier,
                dtype=torch.float32,
                device=self._device,
            )

    def _get_alpha(self) -> Tensor:
        if self._automatic_entropy_tuning:
            return self._log_alpha_fn.exp() * self._alpha_multiplier
        return self._alpha

    def _sample_policy_actions(self, states: Tensor, n_actions: int) -> Tuple[Tensor, Tensor]:
        _, sampled_actions, log_pi = self._p_fn.sample_n(states, n=n_actions)
        return sampled_actions.transpose(0, 1), log_pi.sum(dim=-1).transpose(0, 1)

    def _sample_random_actions(self, batch_size: int, n_actions: int) -> Tuple[Tensor, Tensor]:
        action_range = self._a_max - self._a_min
        random_actions = torch.rand(
            batch_size, n_actions, self._a_dim, device=self._device
        ) * action_range + self._a_min
        random_log_prob = -torch.log(action_range).sum()
        random_log_probs = torch.ones(
            (batch_size, n_actions),
            dtype=random_actions.dtype,
            device=self._device,
        ) * random_log_prob
        return random_actions, random_log_probs

    @staticmethod
    def _reshape_q_values(q_values: Tensor, batch_size: int, n_actions: int) -> Tensor:
        return q_values.view(batch_size, n_actions)

    def _compute_q_values(self, q_fn: nn.Module, states: Tensor, actions: Tensor) -> Tensor:
        batch_size, n_actions, _ = actions.shape
        state_tile = states.unsqueeze(1).repeat(1, n_actions, 1).view(batch_size * n_actions, -1)
        action_flat = actions.reshape(batch_size * n_actions, -1)
        q_values = q_fn(state_tile, action_flat)
        return self._reshape_q_values(q_values, batch_size, n_actions)

    def _build_alpha_loss(self, batch: Dict) -> Tuple[Tensor, Dict]:
        states = batch['s1']
        with torch.no_grad():
            _, _, log_pi = self._p_fn(states)
            log_pi = log_pi.sum(dim=-1)
        alpha_loss = -(self._log_alpha_fn * (log_pi + self._target_entropy)).mean()
        alpha = self._get_alpha()

        info = collections.OrderedDict()
        info['alpha'] = alpha.detach()
        info['alpha_loss'] = alpha_loss.detach()
        return alpha_loss, info

    def _build_q_loss(self, batch: Dict) -> Tuple[Tensor, Dict]:
        states = batch['s1']
        actions = batch['a1']
        rewards = batch['reward']
        next_states = batch['s2']
        discounts = batch['dsc']
        alpha = self._get_alpha().detach()

        with torch.no_grad():
            _, next_actions, next_log_pi = self._p_fn(next_states)
            next_log_pi = next_log_pi.sum(dim=-1)
            target_q1 = self._q_target_fns[0](next_states, next_actions)
            target_q2 = self._q_target_fns[1](next_states, next_actions)
            target_q = torch.minimum(target_q1, target_q2)
            if self._backup_entropy:
                target_q = target_q - alpha * next_log_pi
            td_target = rewards + discounts * self._discount * target_q

        q1_pred = self._q_fns[0](states, actions)
        q2_pred = self._q_fns[1](states, actions)
        q1_bellman_loss = F.mse_loss(q1_pred, td_target)
        q2_bellman_loss = F.mse_loss(q2_pred, td_target)

        batch_size = states.shape[0]
        random_actions, random_log_probs = self._sample_random_actions(batch_size, self._cql_n_actions)
        curr_actions, curr_log_pi = self._sample_policy_actions(states, self._cql_n_actions)
        next_actions_samples, next_log_pi_samples = self._sample_policy_actions(next_states, self._cql_n_actions)
        curr_actions = curr_actions.detach()
        curr_log_pi = curr_log_pi.detach()
        next_actions_samples = next_actions_samples.detach()
        next_log_pi_samples = next_log_pi_samples.detach()

        q1_rand = self._compute_q_values(self._q_fns[0], states, random_actions) - random_log_probs
        q2_rand = self._compute_q_values(self._q_fns[1], states, random_actions) - random_log_probs
        q1_curr = self._compute_q_values(self._q_fns[0], states, curr_actions) - curr_log_pi
        q2_curr = self._compute_q_values(self._q_fns[1], states, curr_actions) - curr_log_pi
        q_next_states = next_states if self._cql_q_next_with_next_states else states
        q1_next = self._compute_q_values(self._q_fns[0], q_next_states, next_actions_samples) - next_log_pi_samples
        q2_next = self._compute_q_values(self._q_fns[1], q_next_states, next_actions_samples) - next_log_pi_samples

        q1_cat = torch.cat([q1_rand, q1_curr, q1_next], dim=1)
        q2_cat = torch.cat([q2_rand, q2_curr, q2_next], dim=1)
        cql_q1_loss = (
            torch.logsumexp(q1_cat / self._cql_temp, dim=1).mean() * self._cql_temp
            - q1_pred.mean()
        )
        cql_q2_loss = (
            torch.logsumexp(q2_cat / self._cql_temp, dim=1).mean() * self._cql_temp
            - q2_pred.mean()
        )

        q_loss = (
            q1_bellman_loss
            + q2_bellman_loss
            + self._cql_alpha * (cql_q1_loss + cql_q2_loss)
        )

        info = collections.OrderedDict()
        info['Q1'] = q1_pred.detach().mean()
        info['Q2'] = q2_pred.detach().mean()
        info['Q_target'] = td_target.detach().mean()
        info['Q1_bellman_loss'] = q1_bellman_loss.detach()
        info['Q2_bellman_loss'] = q2_bellman_loss.detach()
        info['cql_Q1_loss'] = cql_q1_loss.detach()
        info['cql_Q2_loss'] = cql_q2_loss.detach()
        info['Q_loss'] = q_loss.detach()
        info['r_mean'] = rewards.detach().mean()
        info['dsc'] = discounts.detach().mean()
        return q_loss, info

    def _build_p_loss(self, batch: Dict) -> Tuple[Tensor, Dict]:
        states = batch['s1']
        actions = batch['a1']
        alpha = self._get_alpha().detach()

        _, sampled_actions, log_pi = self._p_fn(states)
        log_pi = log_pi.sum(dim=-1)
        q1_pi = self._q_fns[0](states, sampled_actions)
        q2_pi = self._q_fns[1](states, sampled_actions)
        min_q_pi = torch.minimum(q1_pi, q2_pi)
        rl_loss = (alpha * log_pi - min_q_pi).mean()
        policy_log_prob = self._p_fn.get_log_density(states, actions).sum(dim=-1)
        bc_loss = -policy_log_prob.mean()
        p_loss = rl_loss + self._actor_bc_weight * bc_loss
        if self._global_step < self._policy_bc_steps:
            p_loss = bc_loss

        info = collections.OrderedDict()
        info['actor_loss'] = p_loss.detach()
        info['actor_rl_loss'] = rl_loss.detach()
        info['actor_bc_loss'] = bc_loss.detach()
        info['log_pi'] = log_pi.detach().mean()
        info['Q_in_actor_loss'] = min_q_pi.detach().mean()
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
        return info

    def _optimize_step(self, batch: Dict) -> Dict:
        info = collections.OrderedDict()
        q_info = self._optimize_q(batch)
        info.update(q_info)

        if self._global_step % self._update_actor_freq == 0:
            self._p_info = self._optimize_p(batch)
            info.update(self._p_info)
            if self._automatic_entropy_tuning:
                alpha_info = self._optimize_alpha(batch)
                info.update(alpha_info)

        if self._global_step % self._target_update_period == 0:
            self._update_target_fns(self._q_fns, self._q_target_fns)

        return info

    def _build_test_policies(self) -> None:
        policy = policies.DeterministicSoftPolicy(a_network=self._p_fn)
        self._test_policies['main'] = policy

    def save(self, ckpt_name: str) -> None:
        torch.save(self._agent_module.state_dict(), ckpt_name + '.pth')
        torch.save(self._agent_module.p_net.state_dict(), ckpt_name + '_policy.pth')

    def restore(self, ckpt_name: str) -> None:
        self._agent_module.load_state_dict(
            torch.load(ckpt_name + '.pth', map_location=self._device, weights_only=True)
        )


class AgentModule(BaseAgentModule):
    """Container of all trainable modules used by CQL."""

    def _build_modules(self) -> None:
        device = self._net_modules.device
        self._q_nets = nn.ModuleList()
        n_q_fns = self._net_modules.n_q_fns
        for _ in range(n_q_fns):
            self._q_nets.append(self._net_modules.q_net_factory().to(device))
        self._q_target_nets = copy.deepcopy(self._q_nets)
        self._p_net = self._net_modules.p_net_factory().to(device)
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
    def log_alpha_net(self) -> nn.Parameter:
        return self._log_alpha_net
