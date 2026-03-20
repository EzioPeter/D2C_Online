import os
import pytest
import torch
import numpy as np
from easydict import EasyDict
from torch.utils.tensorboard import SummaryWriter
from d2c.models.model_free.cql import CQLAgent
from d2c.data import Data
from d2c.envs import LeaEnv
from d2c.utils.utils import abs_file_path, maybe_makedirs
from d2c.utils.config import ConfigBuilder
from example.benchmark.config.app_config import app_config


class TestCql:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    work_abs_dir = abs_file_path(__file__, '../../../example/benchmark')
    model_config_path = os.path.join(work_abs_dir, 'config', 'model_config.json5')
    prefix = 'env.external.'
    command_args = {
        prefix + 'benchmark_name': 'd4rl',
        prefix + 'data_source': 'mujoco',
        prefix + 'env_name': 'HalfCheetah-v2',
        prefix + 'data_name': 'halfcheetah_medium_expert-v2',
    }
    command_args.update({
        'train.data_loader_name': None,
        'train.device': device,
    })
    cfg_builder = ConfigBuilder(
        app_config=app_config,
        model_config_path=model_config_path,
        work_abs_dir=work_abs_dir,
        command_args=command_args,
    )
    config = cfg_builder.build_config()

    env = LeaEnv(config)
    model_params = EasyDict({'q': [[256, 256], 2], 'p': [[256, 256], ]})
    optimizers = EasyDict({'q': ['adam', 3e-4], 'p': ['adam', 1e-4], 'alpha': ['adam', 3e-4]})
    train_data = Data(config).data

    def test_cql_agent(self):
        if not os.path.exists('./temp/cql'):
            os.makedirs('./temp/cql')
        agent = CQLAgent(
            env=self.env,
            model_params=self.model_params,
            optimizers=self.optimizers,
            train_data=self.train_data,
            device=self.device,
        )
        summary_writer = SummaryWriter('./temp/cql/train_log')
        for _ in range(5):
            agent.train_step()
            agent.write_train_summary(summary_writer)
            agent.print_train_info()
        agent_ckpt_dir = './temp/cql/agent/agent'
        maybe_makedirs(os.path.dirname(agent_ckpt_dir))
        agent.save(agent_ckpt_dir)
        agent.restore(agent_ckpt_dir)

        policy = agent.test_policies['main']
        obs = np.random.random((64, 17))
        action = policy(obs)
        assert action.shape == (64, 6)


if __name__ == '__main__':
    pytest.main(__file__)
