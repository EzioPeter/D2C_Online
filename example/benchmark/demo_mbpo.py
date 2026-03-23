import datetime
import os
import sys
from dataclasses import dataclass

import torch
import tyro

sys.path.append('../../')

import logging

from d2c.evaluators import offpolicy_bm_eval
from d2c.envs import benchmark_env
from d2c.models import make_agent
from d2c.trainers import OffPolicyTrainer as Trainer
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
    agent_ckpt_name: str = 'mbpo-online'
    wandb_mode: str = 'online'


def _get_env_specific_mbpo_overrides(env_name: str) -> dict:
    env_name_lower = env_name.lower()
    if 'halfcheetah' in env_name_lower:
        return {'model.mbpo.hyper_params.rollout_schedule': [20_000, 150_000, 1, 5]}
    if 'walker2d' in env_name_lower:
        return {'model.mbpo.hyper_params.rollout_schedule': [20_000, 150_000, 1, 1]}
    if 'hopper' in env_name_lower:
        return {'model.mbpo.hyper_params.rollout_schedule': [20_000, 150_000, 1, 15]}
    if 'ant' in env_name_lower:
        return {'model.mbpo.hyper_params.rollout_schedule': [20_000, 100_000, 1, 25]}
    if 'humanoid' in env_name_lower:
        return {'model.mbpo.hyper_params.rollout_schedule': [20_000, 300_000, 1, 25]}
    return {}


def main(args: Args) -> None:
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
    }
    command_args.update(_get_env_specific_mbpo_overrides(args.env_name))
    wandb = {
        'project': 'test',
        'name': command_args['env.external.env_name'] + '_seed=' + str(args.seed) + '_' + now_time,
        'reinit': False,
        'mode': args.wandb_mode,
    }
    command_args.update({'train.wandb': wandb})

    config = make_config(command_args)
    env = benchmark_env(config=config)
    agent = make_agent(config=config, env=env, data=None)
    evaluator = offpolicy_bm_eval(agent=agent, env=env, config=config)
    trainer = Trainer(agent=agent, train_data=None, config=config, env=env, evaluator=evaluator)
    trainer.train()


if __name__ == '__main__':
    args = tyro.cli(Args)
    main(args)
