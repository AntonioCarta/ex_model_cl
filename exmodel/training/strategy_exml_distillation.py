import copy

import torch
import torch.nn.functional as F
from torch.utils.data import random_split, Dataset

from avalanche.benchmarks import Experience
from avalanche.benchmarks.utils import AvalancheDataset, AvalancheConcatDataset
from avalanche.benchmarks.utils.data_loader import GroupBalancedInfiniteDataLoader
from avalanche.benchmarks.utils.dataset_utils import ConstantSequence
from avalanche.models import avalanche_forward

from avalanche.training.strategies import BaseStrategy
from torch.nn import Module
from torch.optim import Optimizer

from exmodel.sampler import ReplayBuffer, ModelInversionBuffer, DataImpressionBuffer, BufferFromPath


class ExMLDistillation(BaseStrategy):
    def __init__(self, model: Module, optimizer: Optimizer, bias_normalization, reset_model,
                 loss_type, ce_loss, num_iter_per_exp, init_from_expert, **kwargs):
        super().__init__(model, optimizer, **kwargs)
        assert loss_type in {'mse', 'kldiv'}

        # strategy hyperparams
        self.bias_normalization = bias_normalization
        self.reset_model = reset_model
        self.loss_type = loss_type
        self.ce_loss = ce_loss
        self.num_iter_per_exp = num_iter_per_exp
        self.init_from_expert = init_from_expert

        # strategy state
        self.current_exp_num_iters = 0
        self.init_model = None
        self.expert_model = None
        self.prev_model = None
        self.prev_classes = []
        self.logits_target = None
        self.eval_streams = None

    def _after_forward(self, **kwargs):
        super()._after_forward(**kwargs)
        is_single_task = len(set([el for arr in self.experience.benchmark.task_labels for el in arr])) == 1
        self.logits_targets = torch.zeros_like(self.mb_output)
        curr_classes = self.experience.classes_in_this_experience

        # current model
        if is_single_task:
            self.logits_targets[:, curr_classes] += self.get_masked_bn_logits(self.expert_model, curr_classes)
        else:
            curr_classes = list(range(self.experience.benchmark.n_classes))
            self.logits_targets[:] += self.get_masked_bn_logits(self.expert_model, curr_classes)

        if self.prev_model is not None:
            is_single_task = self.experience.task_label == 0
            if is_single_task:
                self.logits_targets[:, curr_classes] = self.logits_targets[:, curr_classes]
                # Ex-model distillation. Combines ex-model targets with previous CL model.
                self.logits_targets[:, self.prev_classes] += self.get_masked_bn_logits(self.prev_model, self.prev_classes)

                intersection = list(set(curr_classes).intersection(self.prev_classes))
                if len(intersection) > 0:
                    self.logits_targets[:, intersection] = 0.5 * self.logits_targets[:, intersection]

            else:
                # MT scenario. Experts have separate heads. Assumes increasing classes also for MT
                prev_classes = list(range(self.experience.benchmark.n_classes))
                curr_task = self.experience.task_label
                curr_task_mask = self.mb_task_id == curr_task

                # if sum(curr_task_mask).item() > 0:
                # heads from CL model
                self.logits_targets[:] = self.get_masked_bn_logits(self.prev_model, prev_classes)
                # head from new expert
                new_head_targets = self.get_masked_bn_logits(self.expert_model, curr_classes)[curr_task_mask]
                self.logits_targets[curr_task_mask] += new_head_targets

    def criterion(self):
        if self.is_training:
            if self.loss_type == 'mse':
                ll = self.mse_loss()
            elif self.loss_type == 'kldiv':
                ll = self.kldiv_loss()
            else:
                raise ValueError("Unknown loss type")

            if self.ce_loss:
                ll += F.cross_entropy(self.mb_output, self.mb_y)
            return ll
        else:
            return F.cross_entropy(self.mb_output, self.mb_y)

    def get_masked_bn_logits(self, model, selected_classes):
        with torch.no_grad():
            curr_logits = avalanche_forward(model, self.mb_x, self.mb_task_id)[:, selected_classes]
            if self.bias_normalization:
                curr_logits = curr_logits - curr_logits.mean(dim=1).unsqueeze(1)
        return curr_logits

    def mse_loss(self):
        return F.mse_loss(self.mb_output, self.logits_targets)

    def kldiv_loss(self):
        return F.kl_div(self.mb_output, self.logits_targets, reduction='mean', log_target=True)

    def train_dataset_adaptation(self, **kwargs):
        super().train_dataset_adaptation(**kwargs)
        buffer = self.make_buffer()
        self.adapted_dataset = buffer

    def _after_train_dataset_adaptation(self, **kwargs):
        super()._after_train_dataset_adaptation()
        if self.reset_model:
            if self.init_model is None:  # first experience.
                self.init_model = copy.deepcopy(self.model)
            else:
                self.model = copy.deepcopy(self.init_model)

    def _before_training_exp(self, **kwargs):
        if self.init_from_expert and self.clock.train_exp_counter == 0:
            self.model = copy.deepcopy(self.expert_model)
            self.stop_training()

        self.current_exp_num_iters = 0
        super()._before_training_exp(**kwargs)

    def _after_training_exp(self, **kwargs):
        super()._after_training_exp(**kwargs)
        self.prev_model = copy.deepcopy(self.model)
        self.prev_model.eval()
        self.prev_classes.extend(self.experience.classes_in_this_experience)

    def make_buffer(self):
        """ Prepare the data used for the ex-model distillation. """
        assert NotImplementedError()

    def train(self, experiences, eval_streams=None, expert_models=None, **kwargs):
        """ Train ex-model strategy.

        :param experiences:
        :param eval_streams:
        :param expert_models:
        :param kwargs:
        :return:
        """
        self.is_training = True
        self.model.train()
        self.model.to(self.device)

        # Normalize training and eval data.
        if isinstance(experiences, Experience):
            experiences = [experiences]
        if isinstance(expert_models, torch.nn.Module):
            expert_models = [expert_models]
        if eval_streams is None:
            eval_streams = [experiences]
        for i, exp in enumerate(eval_streams):
            if isinstance(exp, Experience):
                eval_streams[i] = [exp]
        self.eval_streams = eval_streams

        self._before_training(**kwargs)
        for self.expert_model, self.experience in zip(expert_models, experiences):
            self.expert_model = self.expert_model.to(self.device)
            self.expert_model.eval()
            self.train_exp(self.experience, eval_streams, **kwargs)
            self.expert_model.to('cpu')
        self._after_training(**kwargs)

        res = self.evaluator.get_last_metrics()
        return res

    def _before_training_iteration(self, **kwargs):
        if self.current_exp_num_iters == self.num_iter_per_exp:
            self.stop_training()
        elif (self.current_exp_num_iters % 1000) == 999:
            _prev_state = (
                self.experience,
                self.adapted_dataset,
                self.dataloader,
                self.is_training)

            for exp in self.eval_streams:
                self.eval(exp)

            # restore train-state variables and training mode.
            self.experience, self.adapted_dataset = _prev_state[:2]
            self.dataloader = _prev_state[2]
            self.is_training = _prev_state[3]
            self.model.train()

        self.current_exp_num_iters += 1
        super()._before_training_iteration(**kwargs)

    def make_train_dataloader(self, num_workers=0, shuffle=True,
                              pin_memory=True, **kwargs):
        """
        Called after the dataset adaptation. Initializes the data loader.
        :param num_workers: number of thread workers for the data loading.
        :param shuffle: True if the data should be shuffled, False otherwise.
        :param pin_memory: If True, the data loader will copy Tensors into CUDA
            pinned memory before returning them. Defaults to True.
        """
        self.dataloader = GroupBalancedInfiniteDataLoader(
            [self.adapted_dataset],
            num_workers=num_workers,
            batch_size=self.train_mb_size,
            pin_memory=pin_memory)


class AuxDataED(ExMLDistillation):
    def __init__(self, model: Module, optimizer: Optimizer, aux_data, bias_normalization,
                 reset_model, loss_type, ce_loss, **kwargs):
        super().__init__(model, optimizer, bias_normalization, reset_model,
                         loss_type, ce_loss,**kwargs)
        self.aux_data = aux_data

    @property
    def mb_y(self):
        if self.is_training:
            return self.logits_targets.argmax(dim=1)
        else:
            return super().mb_y

    def make_buffer(self):
        t = ConstantSequence(0, len(self.aux_data))
        buffer = AvalancheDataset(self.aux_data, task_labels=t)
        return buffer


class ReplayED(ExMLDistillation):
    def __init__(self, model: Module, optimizer: Optimizer, bias_normalization, reset_model,
                 buffer_size, num_iter_per_exp,
                 **kwargs):
        assert 'train_epochs' not in kwargs
        # Set num train epochs to an unreasonably high number.
        # We fix the number of maximum iterations instead.
        super().__init__(model, optimizer, bias_normalization, reset_model, train_epochs=10**20,
                         num_iter_per_exp=num_iter_per_exp, **kwargs)
        # buffer params.
        self.buffer_size = buffer_size

        # strategy state
        self.cum_data = []

    def make_buffer(self):
        numexp = len(self.cum_data) + 1
        bufsize = self.buffer_size // numexp
        curr_buffer = self.make_buffer_exp(bufsize)

        # subsample previous data
        for i, data in enumerate(self.cum_data):
            removed_els = len(data) - bufsize
            if removed_els > 0:
                data, _ = random_split(data, [bufsize, removed_els])
            self.cum_data[i] = data

        self.cum_data.append(curr_buffer)
        return AvalancheConcatDataset(self.cum_data)

    def make_buffer_exp(self, bufsize):
        return ReplayBuffer(self.expert_model, self.experience, bufsize).buffer


class GaussianNoiseED(ExMLDistillation):
    def __init__(self, img_shape, **kwargs):
        super().__init__(**kwargs)
        self.img_shape = img_shape

    @property
    def mb_y(self):
        if self.is_training:
            return self.logits_targets.argmax(dim=1)
        else:
            return super().mb_y

    def make_buffer(self):
        class GND(Dataset):
            def __init__(self, shape):
                self.shape = shape

            def __getitem__(self, index):
                return torch.randn(*self.shape), 0

            def __len__(self):
                return 10 ** 3

        data = GND(self.img_shape)
        t = ConstantSequence(0, len(data))
        return AvalancheDataset(data, targets=t)


class BufferPrecomputedED(ReplayED):
    def __init__(self, model: Module, optimizer: Optimizer, bias_normalization, reset_model,
                 num_iter_per_exp=10**6, logdir=None, transform=None,
                 **kwargs):
        super().__init__(model, optimizer, bias_normalization, reset_model, 1000, num_iter_per_exp, **kwargs)
        self.transform = transform
        self.logdir = logdir

    def make_buffer_exp(self, bufsize):
        return BufferFromPath(
            model=self.expert_model,
            experience=self.experience,
            logdir=self.logdir,
            transform=self.transform
        ).buffer


class ModelInversionED(ReplayED):
    def __init__(self, model: Module, optimizer: Optimizer, bias_normalization, reset_model,
                 buffer_size, num_iter_per_exp,
                 buffer_iter, buffer_lr, buffer_wd, transform,
                 lambda_blur, lambda_bns, buffer_mb_size, temperature,
                 max_buffer_size=None,
                 **kwargs):
        super().__init__(model, optimizer, bias_normalization, reset_model, buffer_size, num_iter_per_exp, **kwargs)
        self.buffer_iter = buffer_iter
        self.buffer_lr = buffer_lr
        self.buffer_wd = buffer_wd
        self.transform = transform
        self.lambda_blur = lambda_blur
        self.lambda_bns = lambda_bns
        self.buffer_mb_size = buffer_mb_size
        self.temperature = temperature
        self.max_buffer_size = max_buffer_size if max_buffer_size is not None else buffer_size

    def make_buffer_exp(self, bufsize):
        return ModelInversionBuffer(
            self.expert_model, self.experience,
            transform=self.transform,
            buffer_size=min(bufsize, self.max_buffer_size),
            device=self.device,
            n_iter=self.buffer_iter,
            lr=self.buffer_lr,
            weight_decay=self.buffer_wd,
            lambda_blur=self.lambda_blur,
            lambda_bns=self.lambda_bns,
            train_batch_size=self.buffer_mb_size,
            temperature=self.temperature,
            verbose=True).buffer


class DataImpressionED(ReplayED):
    def __init__(self, model: Module, optimizer, bias_normalization, reset_model,
                 buffer_size, num_iter_per_exp,
                 buffer_iter, buffer_lr,
                 buffer_beta, buffer_wd,
                 get_classifier,
                 lambda_blur, lambda_bns, buffer_mb_size, temperature,
                 transform, max_buffer_size=None,
                 **kwargs):
        super().__init__(model, optimizer, bias_normalization, reset_model, buffer_size, num_iter_per_exp, **kwargs)
        self.buffer_iter = buffer_iter
        self.buffer_lr = buffer_lr
        self.buffer_beta = buffer_beta
        self.buffer_wd = buffer_wd
        self.get_classifier = get_classifier
        self.transform = transform
        self.lambda_blur = lambda_blur
        self.lambda_bns = lambda_bns
        self.buffer_mb_size = buffer_mb_size
        self.temperature = temperature
        self.max_buffer_size = max_buffer_size if max_buffer_size is not None else buffer_size

    def make_buffer_exp(self, bufsize):
        return DataImpressionBuffer(
            self.expert_model, self.experience,
            transform=self.transform,
            buffer_size=min(bufsize, self.max_buffer_size),
            device=self.device,
            n_iter=self.buffer_iter,
            lr=self.buffer_lr,
            beta=self.buffer_beta,
            weight_decay=self.buffer_wd,
            get_classifier=self.get_classifier,
            lambda_blur=self.lambda_blur,
            lambda_bns=self.lambda_bns,
            train_batch_size=self.buffer_mb_size,
            temperature=self.temperature,
            verbose=True).buffer
