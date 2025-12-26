import sys
import datetime
sys.path.append('../../')
import torch
import logging
from d2c.trainers import OffPolicyTrainer as Trainer
from d2c.models import make_agent
from d2c.envs import benchmark_env
from d2c.evaluators import offpolicy_bm_eval
from example.benchmark.config import make_config

# If you live in China mainland and want to use wandb, you can use this wandb mirror to solve the problem of network.
import os
os.environ["WANDB_BASE_URL"] = "https://api.bandw.top"

logging.basicConfig(level=logging.INFO)
nowTime = datetime.datetime.now().strftime('%y-%m-%d-%H-%M-%S')

import numpy as np
import tyro
from dataclasses import dataclass

@dataclass
class Args:
    env_name: str = 'HalfCheetah-v2'
    wandb_mode: str = 'online'

def main(args: Args):
    seed = np.random.randint(0, 100) 
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    prefix = 'env.external.'
    command_args = {
        prefix + 'benchmark_name': 'gym',
        prefix + 'data_source': 'mujoco',
        prefix + 'env_name': 'HalfCheetah-v2',
        prefix + 'data_name': '',
        prefix + 'state_normalize': False,
        prefix + 'score_normalize': False,
    }
    command_args.update({
        'model.model_name': 'sac',
        'train.data_loader_name': None,
        'train.device': device,
        'train.seed': seed,
        'train.total_train_steps': 1000000,
        'train.batch_size': 256,
        'train.agent_ckpt_name': '1225'
    })
    command_args.update({
        prefix + 'env_name': args.env_name,
    })
    wandb = {
        'project': 'test',
        'name': command_args['env.external.env_name']+'_seed='+str(command_args['train.seed'])+'_'+nowTime,
        'reinit': False,
        'mode': args.wandb_mode
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
