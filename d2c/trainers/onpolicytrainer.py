"""OnPolicy Trainer for RL models."""

import os
import time
import logging
import json5
import copy
from typing import Any, Union, Optional, Callable, Tuple, Dict
from torch.utils.tensorboard import SummaryWriter
from d2c.trainers.base import BaseTrainer
from d2c.models import BaseAgent
from d2c.envs import LeaEnv
from d2c.evaluators import BaseEval
from d2c.utils.replaybuffer import ReplayBuffer
from d2c.utils import utils
from d2c.envs.learned.dynamics import make_dynamics
from d2c.utils.logger import WandbLogger
from d2c.utils.config import ConfigBuilder

import torch

class OnPolicyTrainer(BaseTrainer):
    """Implementation of the onpolicy trainer.

    :param evaluator: the evaluation for testing the training polices. It should \
        contain an external perfect env such as :class:`~d2c.evaluators.sim.benchmark.BMEval`. \
        You can input an evaluator when training in the benchmark experiments.

    .. seealso::

        Please refer to :class:`~d2c.trainers.base.BaseTrainer`
        for more detailed explanation.
    """

    def __init__(
            self,
            agent: Union[BaseAgent, Any],
            train_data: ReplayBuffer,
            config: Union[Any, utils.Flags],
            env: LeaEnv = None,
            evaluator: Union[Any, BaseEval] = None
    ) -> None:
        super(OnPolicyTrainer, self).__init__(agent, env, train_data, config)
        self._train_steps = self._train_cfg.total_train_steps
        self._summary_freq = self._train_cfg.on_policy_summary_freq
        self._print_freq = self._train_cfg.on_policy_print_freq
        self._save_freq = self._train_cfg.on_policy_save_freq
        self._agent_name = self._model_cfg.model.model_name
        self._evaluator = evaluator
        self._eval_freq = self._train_cfg.on_policy_eval_freq

    def train(self) -> None:
        _custom_train = self._build_train_schedule()
        _custom_train()

    def _train_behavior(self) -> None:
        pass

    def _train_dynamics(self) -> None:
        pass

    def _train_q(self) -> None:
        pass

    def _train_vae_s(self) -> None:
        pass

    def _train_agent(self) -> None:
        agent_ckpt_dir = self._train_cfg.agent_ckpt_dir
        utils.maybe_makedirs(os.path.dirname(agent_ckpt_dir))
        train_summary_dir = agent_ckpt_dir + '_train_log'
        train_summary_writer = SummaryWriter(train_summary_dir)
        wandb_logger = self._build_wandb_logger(dir_=train_summary_dir)

        time_st_total = time.time()
        iteration = 0
        total_iterations = self._agent._prepare_for_train(self._train_steps, self._train_cfg.seed)

        while iteration < total_iterations + 1:
            iteration = iteration + 1
            self._agent._current_iteration = iteration
            self._agent._total_iterations = total_iterations
            self._agent.train_step()
            if iteration % self._summary_freq == 0 or iteration == self._train_steps:
                self._agent.write_train_summary(train_summary_writer)
            if iteration % self._print_freq == 0 or iteration == self._train_steps:
                self._agent.print_train_info()
            if iteration % self._eval_freq == 0 or iteration == self._train_steps:
                if self._evaluator is not None:
                    try:
                        eval_info = self._evaluator.eval(self._agent._global_step)
                    except:
                        logging.info('Something wrong when evaluating the policy!')
                    else:
                        eval_info.update(global_step=self._agent._global_step)
                        wandb_logger.write_summary(eval_info)
            if iteration % self._save_freq == 0:
                self._agent.save(agent_ckpt_dir)
                logging.info(f'Agent saved at {agent_ckpt_dir}.')
        self._agent.save(agent_ckpt_dir)
        train_summary_writer.close()
        wandb_logger.finish()
        time_cost = time.time() - time_st_total
        logging.info('Training finished, time cost %.4gs.', time_cost)

    @staticmethod
    def check_ckpt(_model_ckpt_dir: str) -> Tuple[Optional[SummaryWriter], str]:
        """Determine if the model files exist.

        When calling the :meth:`train` method, it will check if the models have been trained
        and decide if to create a file writer.

        :param str _model_ckpt_dir: the file path of the model that will be trained.

        :return: a file_writer for recording the model training information. If the
            model has already been trained, it will return ``None``.

        """
        _train_summary_dir = _model_ckpt_dir+'_train_log'
        if os.path.exists(f'{_model_ckpt_dir}.pth'):
            logging.info(f'Checkpoint found at {_model_ckpt_dir}')
            train_summary_writer = None
        else:
            logging.info(f'No trained checkpoint, train the {_model_ckpt_dir}')
            utils.maybe_makedirs(os.path.dirname(_model_ckpt_dir))
            train_summary_writer = SummaryWriter(
                _train_summary_dir
            )
        return train_summary_writer, _train_summary_dir

    def _build_train_schedule(self) -> Callable:
        train_fn_dict = dict(
            b=self._train_behavior,
            d=self._train_dynamics,
            q=self._train_q,
            vae_s=self._train_vae_s,
            agent=self._train_agent,
        )
        train_sche = self._model_cfg.model[self._agent_name].train_schedule

        def custom_train():
            for x in train_sche:
                train_fn_dict[x]()

        return custom_train

    def _build_wandb_logger(
            self,
            dir_: Optional[str] = None,
            name: Optional[str] = None,
            _config: Optional[Dict] = None,
    ) -> WandbLogger:
        _params = copy.deepcopy(self._model_cfg.train.wandb)
        if dir_ is not None:
            utils.maybe_makedirs(dir_)
            _params.update(dir_=dir_)
        if name is not None:
            _params.update(name=name)
        if _config is None:
            _config = ConfigBuilder.main_hyper_params(self._model_cfg)
        _params.update(config=_config)
        return WandbLogger(**_params)





