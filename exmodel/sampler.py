"""
    Ex-model Sampling.
    Given an experience (the original data) and the trained model
    they generate mini-batches to train a new model.

    NOTE: the original experience is given as a convenience but should NOT
    be used directly since in ex-model scenarios the model has no direct
    access to the original data.
"""
import os

import numpy as np
import torch
from torch import nn
from torch.distributions import Dirichlet
from torch.nn import BatchNorm2d
from torch.optim import Adam
from torch.utils.data import random_split, DataLoader
import torch.nn.functional as F
from torchvision.utils import save_image, make_grid

from avalanche.benchmarks.utils import AvalancheTensorDataset, AvalancheDataset
from avalanche.benchmarks.utils.dataset_utils import ConstantSequence

from exmodel.models import GaussianBlur
import random


class ExModelSampler:
    def __init__(self, buffer, batch_size):
        """ ExModelSampler. """
        self.buffer = buffer
        self.dataloader = DataLoader(self.buffer, batch_size=batch_size,
                                     shuffle=True)
        self.iter_dl = iter(self.dataloader)

    def sample(self):
        """ Sample a mini-batch.

        :return: mini-batch
        """
        try:
            return next(self.iter_dl)
        except StopIteration:
            self.iter_dl = iter(self.dataloader)
            return next(self.iter_dl)


class ExMLBuffer:
    def __init__(self, expert_model, experience, buffer_size=100):
        """ Buffer for ex-model strategies.

        :param expert_model:
        :param experience:
        :param buffer_size:
        """
        expert_model.eval()

        self.expert_model = expert_model
        self.experience = experience
        self.buffer_size = buffer_size
        self.buffer = []

    def __getitem__(self, item):
        return self.buffer[item]

    def __len__(self):
        return len(self.buffer)


class ReplayBuffer(ExMLBuffer):
    def __init__(self, expert_model, experience, buffer_size=100):
        """ Buffer that stores a subsample of the experience's data.

        :param expert_model: trained model (nn.Module)
        :param experience: `Experience` used to train `model`.
        :param buffer_size:
        """
        super().__init__(expert_model, experience, buffer_size)

        data = self.experience.dataset
        rem_len = len(data) - buffer_size
        self.buffer, _ = random_split(data, [buffer_size, rem_len])
        t = ConstantSequence(self.experience.task_label, len(self.buffer))
        self.buffer = AvalancheDataset(self.buffer, task_labels=t)


class AuxDataBuffer(ExMLBuffer):
    def __init__(self, expert_model, experience, aux_data, transform, buffer_size=100):
        """ Buffer that stores a subsample of the experience's data.

        :param model: trained model (nn.Module)
        :param experience: `Experience` used to train `model`.
        :param aux_data: auxiliary data.
        :param buffer_size:
        """
        super().__init__(expert_model, experience, buffer_size)
        self.transform = transform

        rem_len = len(aux_data) - buffer_size
        self.buffer, _ = random_split(aux_data, [buffer_size, rem_len])

        t = ConstantSequence(self.experience.task_label, buffer_size)
        self.buffer = AvalancheDataset(self.buffer, task_labels=t, transform=self.transform)


class SyntheticBuffer(ExMLBuffer):
    def __init__(self, expert_model, experience, transform, buffer_size=100,
                 device='cpu', n_iter=1000, lr=0.1, weight_decay=.001,
                 lambda_blur=0.001, lambda_bns=1.0, train_batch_size=512,
                 verbose=False):
        """ Extracts samples from the model by optimizing random noise.

        :param expert_model:
        :param experience:
        :param transform:
        :param buffer_size:
        :param device:
        :param n_iter:
        :param lr:
        :param weight_decay:
        :param temperature:
        :param lambda_blur:
        :param lambda_bns:
        :param train_batch_size:
        :param verbose:
        """
        super().__init__(expert_model, experience, buffer_size)
        self.expert_model.eval()

        self.device = device
        self.n_iter = n_iter
        self.lr = lr
        self.train_batch_size = train_batch_size
        self.transform = transform
        self.verbose = verbose

        self.weight_decay = weight_decay
        self.lambda_blur = lambda_blur
        self.lambda_bns = lambda_bns

        if self.transform is None:  # no transform
            self.transform = lambda x: x

        self.buffer = None  # generated samples

        self.bn_hooks = []

        self.mb_bns_loss = 0
        if self.lambda_bns > 0:
            self._set_bns_hooks()
        self._optimize_buffer()

    def _criterion(self, y_pred, y_curr):
        raise NotImplementedError()

    def _set_bns_hooks(self):
        batchnorm_modules = list(filter(lambda x: isinstance(x, BatchNorm2d), list(self.expert_model.modules())))

        def batchnorm_hook(module, input, output):
            assert len(input) == 1
            in_mu = torch.mean(input[0], dim=(0, 2, 3))
            in_var = torch.var(input[0], dim=(0, 2, 3), unbiased=False)
            self.mb_bns_loss = SyntheticBuffer.bn_stat_loss(in_mu, in_var, module.running_mean, module.running_var).mean()

        for mod in batchnorm_modules:
            h = mod.register_forward_hook(batchnorm_hook)
            self.bn_hooks.append(h)

    @staticmethod
    def bn_stat_loss(mu1, var1, mu2, var2):
        kd = F.mse_loss(mu2, mu1)
        kd += F.mse_loss(var2, var1)
        return kd

    def _init_buffer(self):
        curr_classes = self.experience.classes_in_this_experience
        img_shape = self.experience.dataset[0][0].shape

        if self.pat_per_class == 0:
            # buffer_size < num_classes. Select a random subset of classes.
            self.pat_per_class = 1
            curr_classes = random.sample(curr_classes, self.buffer_size)

        gen_x = nn.Parameter(torch.randn(self.buffer_size, *img_shape, requires_grad=True, device=self.device))
        gen_y = torch.zeros(self.buffer_size, dtype=torch.long, device=self.device)
        gen_t = ConstantSequence(self.experience.task_label, gen_x.shape[0])
        for idx, yl in enumerate(curr_classes):
            y_curr = gen_y[idx * self.pat_per_class:(idx + 1) * self.pat_per_class]
            y_curr[:] = yl
        return gen_x, gen_y, gen_t

    def _make_dataset(self):
        self.gen_x.requires_grad = False
        self.buffer = AvalancheTensorDataset(
            self.gen_x.to('cpu'), self.gen_y.to('cpu'),
            targets=1, task_labels=self.gen_t, transform=self.transform)

    def _optimize_buffer(self):
        self.expert_model.eval()

        curr_classes = self.experience.classes_in_this_experience
        n_classes = len(curr_classes)
        img_shape = self.experience.dataset[0][0].shape
        flat_size = np.prod(img_shape)
        blur_filter = GaussianBlur(img_shape[0], 7, 1).to(self.device)

        self.pat_per_class = self.buffer_size // n_classes
        if self.pat_per_class > 0:  # make buffer_size a multiple of n_classes
            self.buffer_size = self.pat_per_class * n_classes

        gen_x, gen_y, gen_t = self._init_buffer()
        opt = Adam([gen_x], lr=self.lr, weight_decay=0)
        # opt = SGD([gen_x], lr=self.lr, weight_decay=0, momentum=0.9)
        # optim_data = TensorDataset(gen_x, gen_y)
        for it in range(self.n_iter):
            opt.zero_grad()
            rand_perm = torch.randperm(self.buffer_size)

            loss = 0
            # dl = DataLoader(optim_data, batch_size=self.train_batch_size)
            for base_idx in range(0, self.buffer_size, self.train_batch_size):
                x_curr = gen_x[rand_perm][base_idx:base_idx+self.train_batch_size]
                y_curr = gen_y[rand_perm][base_idx:base_idx+self.train_batch_size]

                self.mb_bns_loss = 0
                x_aug = self.transform(x_curr)
                y_pred = self.expert_model(x_aug)

                # norm penalty
                if self.weight_decay == 0.0:
                    norm = 0.0
                else:
                    norm = (x_curr * x_curr).sum()
                    # norm = torch.abs(x_curr).sum()

                # blur penalty
                if self.lambda_blur == 0.0:
                    blur = 0.0
                else:
                    x_noborder = x_curr.reshape(-1, flat_size)
                    x_blur = blur_filter(x_curr).reshape(-1, flat_size)
                    # blur = F.mse_loss(x_noborder, x_blur, reduction='sum')
                    blur = torch.norm(x_noborder - x_blur, dim=1).sum()

                # bns loss
                if self.lambda_bns == 0.0:
                    bns_loss = 0.0
                else:
                    bns_loss = self.mb_bns_loss  # computed inside hooks
                    # mult = x_curr.shape[0] * x_curr.shape[2] * x_curr.shape[3]
                    # x_mean, x_var = x_curr.mean(dim=(0, 2, 3)), x_curr.var(dim=(0, 2, 3))
                    # bns_loss = F.mse_loss(x_mean, torch.zeros_like(x_mean), reduction='sum') * mult
                    # bns_loss += F.mse_loss(x_var, torch.ones_like(x_var), reduction='sum') * mult

                # total loss
                loss = self._criterion(y_pred, y_curr) + \
                       self.weight_decay * norm + \
                       self.lambda_blur * blur + \
                       self.lambda_bns * bns_loss
                loss.backward()
            opt.step()
            gen_x.data.clip_(-3, +3)
            # gen_x.data[:] -= 10 * self.weight_decay * blur_filter(gen_x)

            if in_notebook() and self.verbose and it % 500 == 499:
                import matplotlib.pyplot as plt
                sx = gen_x.detach()[::10]
                plt.figure(figsize=(24, 96))
                img = make_grid(sx.detach().cpu(), nrow=10, scale_each=True)
                npimg = img.numpy()
                plt.imshow(np.transpose(npimg, (1, 2, 0)), interpolation='nearest')

            if self.verbose and it % 100 == 0:
                self.gen_x, self.gen_y, self.gen_t = gen_x.detach(), gen_y.detach(), gen_t
                self._make_dataset()
                print(f"\t(it={it}) loss: {loss:.6f}, blur: {blur: .6f}, norm: {norm: .6f}, bns: {bns_loss: .6f}")
                sampler = ExModelSampler(self, batch_size=32)
                sx, sy, _ = sampler.sample()
                sx, sy = sx.to(self.device), sy.to(self.device)
                px = F.softmax(self.expert_model(sx), dim=1)
                px_y = torch.gather(px, 1, sy.reshape(-1, 1))[:, 0].mean()
                print(f"mean_acc: {px_y:.4f}")
                os.makedirs('./logs/tmp_logs', exist_ok=True)
                save_image(sx, f'./logs/tmp_logs/genbuff_it{it}.jpg')

        self.gen_x, self.gen_y, self.gen_t = gen_x.detach(), gen_y.detach(), gen_t
        self._make_dataset()

        for handle in self.bn_hooks:
            handle.remove()


class BufferFromPath(ExMLBuffer):
    def __init__(self, model, experience, logdir, transform):
        super().__init__(model, experience, None)
        exp_id = experience.current_experience
        gen_x = torch.load(f'{logdir}/buffer_x_e{exp_id}.pt').to('cuda')
        gen_y = torch.load(f'{logdir}/buffer_y_e{exp_id}.pt').to('cuda')
        gen_t = ConstantSequence(self.experience.task_label, gen_x.shape[0])
        self.buffer = AvalancheTensorDataset(gen_x, gen_y, targets=1, task_labels=gen_t, transform=transform)


class ModelInversionBuffer(SyntheticBuffer):
    def __init__(self, expert_model, experience, transform, buffer_size=100,
                 device='cpu', n_iter=1000, lr=0.1, weight_decay=.001,
                 lambda_blur=0.001, lambda_bns=1.0, train_batch_size=512, verbose=False,
                 temperature=1.0):
        """ Extracts samples from the model by optimizing random noise to reach
        low error when using model to predict the class.

        :param expert_model:
        :param experience:
        :param transform:
        :param buffer_size:
        :param device:
        :param n_iter:
        :param lr:
        :param weight_decay:
        :param temperature:
        :param lambda_blur:
        :param lambda_bns:
        :param train_batch_size:
        :param verbose:
        """
        self.temperature = temperature
        super().__init__(expert_model=expert_model, experience=experience, transform=transform,
                         buffer_size=buffer_size, device=device, n_iter=n_iter, lr=lr,
                         weight_decay=weight_decay, lambda_blur=lambda_blur, lambda_bns=lambda_bns,
                         train_batch_size=train_batch_size, verbose=verbose)

    def _criterion(self, y_pred, y_curr):
        return F.cross_entropy(y_pred / self.temperature, y_curr, reduction='sum')


def _default_get_classifier(model):
    return model.classifier.weight


class DataImpressionBuffer(SyntheticBuffer):
    def __init__(self, expert_model, experience, transform, buffer_size=100,
                 device='cpu', n_iter=1000, lr=0.1, weight_decay=.001,
                 lambda_blur=0.001, lambda_bns=1.0, train_batch_size=512, verbose=False,
                 temperature=1.0, beta=0.1, get_classifier=_default_get_classifier):
        """ Data Impression.

        :param expert_model:
        :param experience:
        :param transform:
        :param buffer_size:
        :param device:
        :param n_iter:
        :param lr:
        :param weight_decay:
        :param lambda_blur:
        :param lambda_bns:
        :param train_batch_size:
        :param verbose:
        :param tau:
        :param beta:
        :param get_classifier:
        """
        self.beta = beta
        self.temperature = temperature
        self._get_classifier = get_classifier
        super().__init__(expert_model=expert_model, experience=experience, transform=transform,
                         buffer_size=buffer_size, device=device, n_iter=n_iter, lr=lr,
                         weight_decay=weight_decay, lambda_blur=lambda_blur, lambda_bns=lambda_bns,
                         train_batch_size=train_batch_size, verbose=verbose)

    def _criterion(self, y_pred, y_curr):
        return F.kl_div(F.log_softmax(y_pred / self.temperature), y_curr, reduction='sum')

    def _init_buffer(self):
        try:
            num_classes = self.experience.scenario.n_classes
        except:
            # core50nc doesn't have num_classes attribute
            # Avalanche bug: https://github.com/ContinualAI/avalanche/issues/758
            numexp = len(self.experience.scenario.train_stream)
            classesl = []
            for s in self.experience.scenario.classes_in_experience['train'][:numexp][0]:
                classesl.extend(s)
            num_classes = max(classesl) + 1
            # num_classes = max([el for s in list(self.experience.scenario.classes_in_experience['train']) for el in s]) + 1

        curr_classes = self.experience.classes_in_this_experience
        img_shape = self.experience.dataset[0][0].shape

        gen_x = nn.Parameter(torch.randn(self.buffer_size, *img_shape, device=self.device))
        gen_y = torch.zeros(self.buffer_size, num_classes, device=self.device)
        gen_t = ConstantSequence(self.experience.task_label, gen_x.shape[0])

        W = self._get_classifier(self.expert_model)
        W = W / torch.norm(W, dim=1, keepdim=True)  # row normalization
        self.class_sim = (W @ W.T)

        for idx, yl in enumerate(curr_classes):
            if self.verbose:
                print(f"generating class {yl}")

            idx_start = idx*self.pat_per_class
            idx_end = (idx+1)*self.pat_per_class
            y_curr = gen_y[idx_start:idx_end]
            print(f"idx: ({idx_start}, {idx_end})")

            # sample labels
            concentration = self.class_sim[yl]
            minc, maxc = torch.min(concentration), torch.max(concentration)
            concentration = (concentration - minc) / (maxc - minc)
            concentration = self.beta * concentration + 0.0001
            dist = Dirichlet(concentration)
            for ii in range(self.pat_per_class):
                y_curr[ii] = dist.sample()
        return gen_x, gen_y, gen_t

    def _optimize_buffer(self):
        super()._optimize_buffer()
        self._make_dataset()

    def _make_dataset(self):
        self.gen_x.requires_grad = False
        self.target_classes_gen_y = torch.max(self.gen_y, dim=1)[1]
        self.buffer = AvalancheTensorDataset(
            self.gen_x.to('cpu'), self.target_classes_gen_y.to('cpu'),
            targets=1, task_labels=self.gen_t, transform=self.transform)
