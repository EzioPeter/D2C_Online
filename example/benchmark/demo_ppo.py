import sys
import datetime
sys.path.append('../../')
import torch
import logging
from d2c.trainers import OnPolicyTrainer as Trainer
from d2c.models import make_agent
from d2c.envs import benchmark_env, LeaEnv
from d2c.data import Data
from d2c.evaluators import bm_eval, onpolicy_bm_eval
from example.benchmark.config import make_config

# If you live in China mainland and want to use wandb, you can use this wandb mirror to solve the problem of network.
import os
os.environ["WANDB_BASE_URL"] = "https://api.bandw.top"

logging.basicConfig(level=logging.INFO)
nowTime = datetime.datetime.now().strftime('%y-%m-%d-%H-%M-%S')

import numpy as np
import random
import tyro
from dataclasses import dataclass

@dataclass
class Args:
    env_name: str = 'HalfCheetah-v2'

def main(args: Args):
    seed = np.random.randint(0, 100) 
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    prefix = 'env.external.'
    command_args = {
        prefix + 'benchmark_name': 'gym',
        prefix + 'data_source': 'mujoco',
        prefix + 'env_name': 'HalfCheetah-v2',
        prefix + 'state_normalize': False,
        prefix + 'score_normalize': False,
    }
    command_args.update({
        'model.model_name': 'ppo',
        'train.device': device,
        'train.seed': 42,
        'train.total_train_steps': 1000000,
        'train.batch_size': 256,
        'train.agent_ckpt_name': '1211'
    })
    command_args.update({
        prefix + 'env_name': args.env_name,
    })
    wandb = {
        'project': 'test',
        'name': command_args['env.external.env_name']+'_seed='+str(command_args['train.seed'])+'_'+nowTime,
        'reinit': False,
        'mode': 'online'
    }
    command_args.update({'train.wandb': wandb})

    config = make_config(command_args)
    env = benchmark_env(config=config)

    # agent with an empty buffer
    agent = make_agent(config=config, env=env, data=None)
    # envaluate in the real env
    evaluator = onpolicy_bm_eval(agent=agent, env=env, config=config)
    # train in the sim env
    trainer = Trainer(agent=agent, train_data=None, config=config, env=env, evaluator=evaluator)
    trainer.train()


if __name__ == '__main__':
    args = tyro.cli(Args)
    main(args)
