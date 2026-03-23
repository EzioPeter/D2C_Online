import os
import tempfile

import pytest
import torch

from d2c.envs import benchmark_env
from d2c.models import make_agent
from d2c.utils.config import ConfigBuilder
from d2c.utils.utils import abs_file_path
from example.benchmark.config.app_config import app_config


class TestMbpo:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    work_abs_dir = abs_file_path(__file__, '../../../example/benchmark')
    model_config_path = os.path.join(work_abs_dir, 'config', 'model_config.json5')
    prefix = 'env.external.'
    command_args = {
        prefix + 'benchmark_name': 'gym',
        prefix + 'data_source': 'classic',
        prefix + 'env_name': 'Pendulum-v1',
        prefix + 'data_name': '',
        prefix + 'state_normalize': False,
        prefix + 'score_normalize': False,
        'model.model_name': 'mbpo',
        'model.mbpo.hyper_params.model_params': {'q': [[64, 64], 2], 'p': [[64, 64]]},
        'model.mbpo.hyper_params.buffer_size': 1000,
        'model.mbpo.hyper_params.model_buffer_size': 1000,
        'model.mbpo.hyper_params.learning_starts': 8,
        'model.mbpo.hyper_params.model_train_freq': 4,
        'model.mbpo.hyper_params.model_train_steps': 1,
        'model.mbpo.hyper_params.rollout_freq': 4,
        'model.mbpo.hyper_params.rollout_batch_size': 8,
        'model.mbpo.hyper_params.rollout_schedule': [0, 1, 1, 1],
        'model.mbpo.hyper_params.real_data_ratio': 0.5,
        'model.mbpo.hyper_params.num_sac_updates_per_step': 1,
        'env.learned.prob.model_params': [[32, 32], 2],
        'train.data_loader_name': None,
        'train.device': device,
        'train.batch_size': 8,
        'train.seed': 1,
    }
    cfg_builder = ConfigBuilder(
        app_config=app_config,
        model_config_path=model_config_path,
        work_abs_dir=work_abs_dir,
        command_args=command_args,
    )
    config = cfg_builder.build_config()

    def test_mbpo_agent(self):
        env = benchmark_env(config=self.config)
        agent = make_agent(config=self.config, env=env, data=None)
        agent._current_state, _ = env.reset(seed=self.config.model_config.train.seed)

        for _ in range(24):
            agent.train_step()

        assert agent._real_buffer.size >= 24
        assert agent._dynamics is not None
        assert agent._model_buffer.size > 0

        policy = agent.test_policies['main']
        obs, _ = env.reset(seed=0)
        obs_tensor = torch.as_tensor(obs, device=agent._device, dtype=torch.float32)
        with torch.no_grad():
            action, _, _ = policy(obs_tensor)
        assert action.shape == (1, agent._action_space.shape[0])

        with tempfile.TemporaryDirectory() as tmp_dir:
            ckpt = os.path.join(tmp_dir, 'agent')
            agent.save(ckpt)
            agent.restore(ckpt)


if __name__ == '__main__':
    pytest.main(__file__)
