import datetime
import logging
import math
import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Tuple

import tyro
import torch
from torch.utils.tensorboard import SummaryWriter

sys.path.append('../../')

from d2c.evaluators import offpolicy_bm_eval
from d2c.envs import benchmark_env
from d2c.models import make_agent
from d2c.utils import utils
from d2c.utils.logger import WandbLogger
from example.benchmark.config import make_config

# If you live in China mainland and want to use wandb, you can use this wandb mirror to solve the problem of network.
os.environ["WANDB_BASE_URL"] = "https://api.bandw.top"

logging.basicConfig(level=logging.INFO)
now_time = datetime.datetime.now().strftime('%y-%m-%d-%H-%M-%S')


@dataclass
class Args:
    env_name: str = 'HalfCheetah-v2'
    total_train_steps: int = 1_000_000
    seed: int = 1
    agent_ckpt_name: str = 'mbpo-watchdog'
    wandb_mode: str = 'online'
    summary_freq: int = 100
    print_freq: int = 1000
    save_freq: int = 10000
    eval_freq: int = 10000
    disable_eval: bool = False
    q_loss_threshold: float = 1e5
    q_abs_threshold: float = 2e3
    alpha_threshold: float = 10.0
    consecutive_alerts: int = 2
    intervention_cooldown: int = 1000
    max_interventions: int = 20
    pause_on_alert: bool = True


def _get_env_specific_mbpo_overrides(env_name: str) -> dict:
    env_name_lower = env_name.lower()
    if 'halfcheetah' in env_name_lower:
        return {'model.mbpo.hyper_params.rollout_schedule': [20_000, 150_000, 1, 1]}
    if 'walker2d' in env_name_lower:
        return {'model.mbpo.hyper_params.rollout_schedule': [20_000, 150_000, 1, 1]}
    if 'hopper' in env_name_lower:
        return {'model.mbpo.hyper_params.rollout_schedule': [20_000, 150_000, 1, 15]}
    if 'ant' in env_name_lower:
        return {'model.mbpo.hyper_params.rollout_schedule': [20_000, 100_000, 1, 25]}
    if 'humanoid' in env_name_lower:
        return {'model.mbpo.hyper_params.rollout_schedule': [20_000, 300_000, 1, 25]}
    return {}


def _build_config(args: Args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    prefix = 'env.external.'
    command_args = {
        prefix + 'benchmark_name': 'gym',
        prefix + 'data_source': 'mujoco',
        prefix + 'env_name': args.env_name,
        prefix + 'data_name': '',
        prefix + 'state_normalize': False,
        prefix + 'score_normalize': False,
        'model.model_name': 'mbpo',
        'train.data_loader_name': None,
        'train.device': device,
        'train.seed': args.seed,
        'train.total_train_steps': args.total_train_steps,
        'train.batch_size': 256,
        'train.agent_ckpt_name': args.agent_ckpt_name,
        'train.summary_freq': args.summary_freq,
        'train.print_freq': args.print_freq,
        'train.save_freq': args.save_freq,
        'train.eval_freq': args.eval_freq,
    }
    command_args.update(_get_env_specific_mbpo_overrides(args.env_name))
    wandb = {
        'project': 'test',
        'name': command_args['env.external.env_name'] + '_watchdog_seed=' + str(args.seed) + '_' + now_time,
        'reinit': False,
        'mode': args.wandb_mode,
    }
    command_args.update({'train.wandb': wandb})
    return make_config(command_args)


class MBPOWatchdog:
    def __init__(self, args: Args) -> None:
        self._args = args
        self._alert_streak = 0
        self._intervention_count = 0
        self._last_intervention_step = -10**18

    @staticmethod
    def _metric(info: Dict, key: str) -> float:
        value = info.get(key)
        if value is None:
            return 0.0
        return float(value)

    def _find_alerts(self, info: Dict) -> List[str]:
        alerts = []
        monitored_keys = ['Q1', 'Q2', 'Q_target', 'Q_loss', 'Q1_loss', 'Q2_loss', 'actor_loss', 'alpha']
        for key in monitored_keys:
            value = info.get(key)
            if value is None:
                continue
            value = float(value)
            if not math.isfinite(value):
                alerts.append(f'{key}=non_finite')
        if abs(self._metric(info, 'Q1')) >= self._args.q_abs_threshold:
            alerts.append(f'|Q1|>={self._args.q_abs_threshold:g}')
        if abs(self._metric(info, 'Q2')) >= self._args.q_abs_threshold:
            alerts.append(f'|Q2|>={self._args.q_abs_threshold:g}')
        if abs(self._metric(info, 'Q_target')) >= self._args.q_abs_threshold:
            alerts.append(f'|Q_target|>={self._args.q_abs_threshold:g}')
        if self._metric(info, 'Q_loss') >= self._args.q_loss_threshold:
            alerts.append(f'Q_loss>={self._args.q_loss_threshold:g}')
        if self._metric(info, 'alpha') >= self._args.alpha_threshold:
            alerts.append(f'alpha>={self._args.alpha_threshold:g}')
        return alerts

    def _apply_intervention(self, agent, step: int, alerts: List[str]) -> None:
        self._intervention_count += 1
        self._last_intervention_step = step

        emergency_ckpt = f"{agent._config.model_config.train.agent_ckpt_dir}_watchdog_step{step}"
        agent.save(emergency_ckpt)

        logging.warning(
            'Watchdog detected instability at step %d due to %s. '
            'Saved emergency checkpoint without changing training hyper-parameters.',
            step,
            ', '.join(alerts),
        )
        logging.warning('Emergency checkpoint saved at %s.pth', emergency_ckpt)
        metric_summary = {
            key: float(agent._train_info[key])
            for key in (
                'Q1', 'Q2', 'Q_target', 'Q_loss', 'Q1_loss', 'Q2_loss',
                'actor_loss', 'alpha', 'model_buffer_size', 'model_rollout_transitions',
                'rollout_length', 'sac_update_step', 'dynamics_global_step'
            )
            if key in agent._train_info
        }
        logging.warning('Watchdog metric snapshot: %s', metric_summary)
        if self._args.pause_on_alert:
            raise RuntimeError(
                f'Watchdog paused training at step {step}. Alerts: {alerts}. '
                f'Emergency checkpoint: {emergency_ckpt}.pth'
            )

    def maybe_intervene(self, agent, step: int, info: Dict) -> Tuple[bool, List[str]]:
        alerts = self._find_alerts(info)
        if alerts:
            self._alert_streak += 1
        else:
            self._alert_streak = 0
            return False, []

        if step - self._last_intervention_step < self._args.intervention_cooldown:
            return False, alerts
        if self._alert_streak < self._args.consecutive_alerts:
            return False, alerts
        if self._intervention_count >= self._args.max_interventions:
            raise RuntimeError(
                f'Watchdog exceeded max_interventions={self._args.max_interventions} at step {step}. '
                f'Latest alerts: {alerts}'
            )

        self._apply_intervention(agent, step, alerts)
        self._alert_streak = 0
        return True, alerts

    @property
    def intervention_count(self) -> int:
        return self._intervention_count

    @property
    def alert_streak(self) -> int:
        return self._alert_streak


def main(args: Args) -> None:
    config = _build_config(args)
    env = benchmark_env(config=config)
    agent = make_agent(config=config, env=env, data=None)
    evaluator = None if args.disable_eval else offpolicy_bm_eval(agent=agent, env=env, config=config)
    watchdog = MBPOWatchdog(args)

    train_cfg = config.model_config.train
    agent_ckpt_dir = train_cfg.agent_ckpt_dir
    utils.maybe_makedirs(os.path.dirname(agent_ckpt_dir))
    train_summary_dir = agent_ckpt_dir + '_train_log_watchdog'
    train_summary_writer = SummaryWriter(train_summary_dir)
    wandb_logger = WandbLogger(
        project=train_cfg.wandb.project,
        entity=getattr(train_cfg.wandb, 'entity', None),
        name=train_cfg.wandb.name,
        config={},
        dir_=train_summary_dir,
        reinit=train_cfg.wandb.reinit,
        mode=train_cfg.wandb.mode,
    )

    time_st_total = datetime.datetime.now()
    try:
        agent._current_state, _ = env.reset(seed=train_cfg.seed)
        step = agent.global_step
        while step < train_cfg.total_train_steps:
            agent.train_step()
            step = agent.global_step
            intervened, alerts = watchdog.maybe_intervene(agent, step, agent._train_info)

            watchdog_info = {
                'watchdog_interventions': watchdog.intervention_count,
                'watchdog_alert_streak': watchdog.alert_streak,
            }
            if intervened:
                watchdog_info['watchdog_last_intervention_step'] = step
            for key, value in watchdog_info.items():
                train_summary_writer.add_scalar(key, value, step)
            WandbLogger.write_summary(dict(global_step=step, **watchdog_info))

            if step % train_cfg.summary_freq == 0 or step == train_cfg.total_train_steps:
                agent.write_train_summary(train_summary_writer)
            if step % train_cfg.print_freq == 0 or step == train_cfg.total_train_steps:
                agent.print_train_info()
                if alerts:
                    logging.info('Watchdog alerts at step %d: %s', step, ', '.join(alerts))
            if (step % train_cfg.eval_freq == 0 or step == train_cfg.total_train_steps) and evaluator is not None:
                try:
                    eval_info = evaluator.eval(step)
                except Exception:
                    logging.info('Something wrong when evaluating the policy!')
                else:
                    eval_info.update(global_step=step)
                    eval_info.update(watchdog_info)
                    wandb_logger.write_summary(eval_info)
            if step % train_cfg.save_freq == 0:
                agent.save(agent_ckpt_dir)
                logging.info('Agent saved at %s.', agent_ckpt_dir)

        agent.save(agent_ckpt_dir)
        time_cost = datetime.datetime.now() - time_st_total
        logging.info('Watchdog training finished, time cost %s.', time_cost)
    finally:
        train_summary_writer.close()
        wandb_logger.finish()
        env.close()


if __name__ == '__main__':
    args = tyro.cli(Args)
    main(args)
