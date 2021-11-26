################################################################################
# Copyright (c) 2021 ContinualAI.                                              #
# Copyrights licensed under the MIT License.                                   #
# See the accompanying LICENSE file for terms.                                 #
#                                                                              #
# Date: 2020-01-25                                                             #
# Author(s): Antonio Carta                                                     #
# E-mail: contact@continualai.org                                              #
# Website: avalanche.continualai.org                                           #
################################################################################
from copy import copy
from rich import print
import sys
from typing import List, TYPE_CHECKING

import torch
from torchvision.utils import save_image

from avalanche.evaluation.metric_results import MetricValue, TensorImage
from avalanche.logging import StrategyLogger
from avalanche.evaluation.metric_utils import stream_type

from rich.console import Console
from rich.table import Table
from rich.live import Live

from tqdm import tqdm

if TYPE_CHECKING:
    from avalanche.training import BaseStrategy


class RichTextLogger(StrategyLogger):
    """
    The `TextLogger` class provides logging facilities
    printed to a user specified file. The logger writes
    metric results after each training epoch, evaluation
    experience and at the end of the entire evaluation stream.

    .. note::
        To avoid an excessive amount of printed lines,
        this logger will **not** print results after
        each iteration. If the user is monitoring
        metrics which emit results after each minibatch
        (e.g., `MinibatchAccuracy`), only the last recorded
        value of such metrics will be reported at the end
        of the epoch.

    .. note::
        Since this logger works on the standard output,
        metrics producing images or more complex visualizations
        will be converted to a textual format suitable for
        console printing. You may want to add more loggers
        to your `EvaluationPlugin` to better support
        different formats.
    """
    def __init__(self, logdir, file=sys.stdout, interactive=True):
        """
        Creates an instance of `TextLogger` class.

        :param logdir: a directory to save metric value not serializable to
            strings (e.g. images).
        :param file: destination file to which print metrics
            (default=sys.stdout).
        :param interactive: whether to use the interactive display mode,
            suggested for printing in the terminal. Disable it if you want to
            log the output in a logging file to reduce the visual clutter such
            as progress bars. (default=True)
        """
        super().__init__()
        self.logdir = logdir
        self.file = file
        self.interactive = interactive

        # state
        self.metric_vals = {}
        self.live = None
        self.curr_table = None
        self.is_table_initialized = False
        self._pbar = None

    def log_metric(self, metric_value: 'MetricValue', callback: str) -> None:
        name = metric_value.name
        x = metric_value.x_plot
        val = metric_value.value
        self.metric_vals[name] = (name, x, val)

    def _val_to_str(self, m_val):
        if isinstance(m_val, torch.Tensor):
            return '\n' + str(m_val)
        elif isinstance(m_val, float):
            return f'{m_val:.4f}'
        else:
            return str(m_val)

    def print_current_metrics(self, strategy):
        if strategy.is_training:
            self.train_print_current_metrics()
        else:
            self.eval_print_current_metrics(strategy)

    def train_print_current_metrics(self):
        sorted_vals = sorted(self.metric_vals.values(),
                             key=lambda x: x[0])
        for name, x, val in sorted_vals:
            if isinstance(val, TensorImage):
                str_name = '_'.join(name.split('/'))
                fname = f'{self.logdir}/{str_name}_x_{x}.png'
                save_image(val.image, fname)
                val = f'<serialized to {fname}>.'
            else:
                val = self._val_to_str(val)
            name = name.split('/')[0]
            print(f'\t{name} = {val}', file=self.file, flush=True)

    def eval_print_current_metrics(self, strategy):
        sorted_vals = sorted(self.metric_vals.values(),
                             key=lambda x: x[0])
        if not self.is_table_initialized:
            self.curr_table.add_column("Exp/Task")
            for name, _, _ in sorted_vals:
                # keep only the metric name.
                # Other info is redundant
                name = name.split('/')[0]
                self.curr_table.add_column(name)
            self.is_table_initialized = True

        vals = []
        exp_id = strategy.experience.current_experience
        task_id = strategy.experience.task_label
        vals.append(f"E{exp_id}T{task_id}")
        for name, x, val in sorted_vals:
            if isinstance(val, TensorImage):
                str_name = '_'.join(name.split('/'))
                fname = f'{self.logdir}/{str_name}_x_{x}.png'
                save_image(val.image, fname)
                val = f'<serialized to {fname}>.'
            else:
                val = self._val_to_str(val)
            vals.append(val)
        self.curr_table.add_row(*vals)
        self.live.refresh()
        # print(self.curr_table)

    def after_training_epoch(self, strategy: 'BaseStrategy',
                             metric_values: List['MetricValue'], **kwargs):
        self._end_progress()
        super().after_training_epoch(strategy, metric_values, **kwargs)
        print(f'\t[italic]Epoch {strategy.clock.train_exp_epochs} ended.', file=self.file, flush=True)
        self.print_current_metrics(strategy)
        self.metric_vals = {}

    def after_eval_exp(self, strategy: 'BaseStrategy',
                       metric_values: List['MetricValue'], **kwargs):
        super().after_eval_exp(strategy, metric_values, **kwargs)
        self.print_current_metrics(strategy)
        self.metric_vals = {}

    def before_training(self, strategy: 'BaseStrategy',
                        metric_values: List['MetricValue'], **kwargs):
        super().before_training(strategy, metric_values, **kwargs)
        print('[bold blue]Start of training phase', file=self.file, flush=True)

    def before_eval_dataset_adaptation(self, strategy: 'BaseStrategy',
                    metric_values: List['MetricValue'], **kwargs):
        super().before_eval(strategy, metric_values, **kwargs)
        if self.live is None:
            print('[bold blue]Start of eval phase', file=self.file, flush=True)

            stream = stream_type(strategy.experience)
            self.curr_table = Table(title=f"Eval Metrics - {stream} stream")
            self.is_table_initialized = False
            self.live = Live(self.curr_table, refresh_per_second=4)
            self.live.start()

    def after_training(self, strategy: 'BaseStrategy',
                       metric_values: List['MetricValue'], **kwargs):
        super().after_training(strategy, metric_values, **kwargs)
        print('[bold blue]End of training phase', file=self.file, flush=True)

    def after_training_iteration(self, strategy: 'BaseStrategy',
                                 metric_values: List['MetricValue'], **kwargs):
        if self.interactive:
            self._progress.update()
            self._progress.refresh()
        super().after_training_iteration(strategy, metric_values, **kwargs)

    def before_training_epoch(self, strategy: 'BaseStrategy',
                              metric_values: List['MetricValue'], **kwargs):
        super().before_training_epoch(strategy, metric_values, **kwargs)
        self._progress.total = len(strategy.dataloader)

    def after_eval(self, strategy: 'BaseStrategy',
                   metric_values: List['MetricValue'], **kwargs):
        super().after_eval(strategy, metric_values, **kwargs)
        self.live.stop()
        self.live = None

        print('\t[italic]-- stream metrics', file=self.file, flush=True)
        sorted_vals = sorted(self.metric_vals.values(),
                             key=lambda x: x[0])
        for name, x, val in sorted_vals:
            if isinstance(val, TensorImage):
                str_name = '_'.join(name.split('/'))
                save_image(val.image, f'{self.logdir}/{str_name}_x_{x}.png')
                val = f'<serialized to {str_name}>.'
            else:
                val = self._val_to_str(val)
            name = name.split('/')[0]
            print(f'\t{name} = {val}', file=self.file, flush=True)
        print('[bold blue]End of eval phase', file=self.file, flush=True)
        self.metric_vals = {}

    @property
    def _progress(self):
        if self._pbar is None:
            self._pbar = tqdm(leave=True, position=0, file=sys.stdout)
        return self._pbar

    def _end_progress(self):
        if self._pbar is not None:
            self._pbar.close()
            self._pbar = None
