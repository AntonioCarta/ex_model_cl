import os
import torch
from torchvision.datasets import FashionMNIST
from torchvision.transforms import Normalize, RandomHorizontalFlip, RandomCrop, \
    RandomRotation, CenterCrop, ColorJitter, RandomResizedCrop
from torchvision.transforms import Resize, ToTensor, Compose
from torchvision.models import resnet18, mobilenet_v2

from avalanche.benchmarks import PermutedMNIST, SplitMNIST, SplitCIFAR10, \
    CORe50, SplitCIFAR100, nc_benchmark
from avalanche.benchmarks.classic.ccifar100 import _get_cifar100_dataset, \
    _default_cifar100_eval_transform, _get_cifar10_dataset
from avalanche.benchmarks.classic.cmnist import _get_mnist_dataset
from avalanche.benchmarks.datasets import default_dataset_location
from avalanche.benchmarks.utils import AvalancheConcatDataset
from avalanche.models import SimpleMLP, SimpleCNN
from exmodel.benchmarks import ImageNet128, ImageNet32, ExModelNIC50Cifar100, NIC50Cifar100
from exmodel.models import LeNet5, MTLeNet5
from exmodel.gem_resnet import ResNet18, MTResNet18, MTMaskedResNet18
from torch import nn
from torchvision import transforms

from exmodel.sampler import _default_get_classifier
from exmodel.utils import to_json

SEED_BENCHMARKS = 1234  # old seed. Don't use it anymore.
SEED_BENCHMARK_RUNS = [1234, 2345, 3456, 5678, 6789]  # 5 different seeds to randomize class orders


class GaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        """ pytorch transform. for additive gaussian noise. """
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        noise = torch.randn(tensor.size(), device=tensor.device)
        return tensor + noise * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + \
               '(mean={0}, std={1})'.format(self.mean, self.std)


def load_buffer_transform(args):
    if 'cifar100' in args.scenario:
        return Compose([RandomCrop(32, padding=4), RandomHorizontalFlip(), RandomRotation(15)])
    elif 'cifar10' in args.scenario:
        return Compose([ColorJitter(), RandomResizedCrop(32, scale=(0.8, 1.0)),
                        RandomHorizontalFlip(), RandomRotation(15)])
    elif 'core50' in args.scenario:
        return Compose([transforms.RandomCrop(128, padding=4), transforms.RandomHorizontalFlip(), RandomRotation(15)])
    elif 'mnist' in args.scenario:
        return Compose([RandomCrop(32, padding=4)])
    else:
        raise ValueError(f"No available transform for scenario: {args.scenario}")


def in_notebook():
    try:
        from IPython import get_ipython
        if 'IPKernelApp' not in get_ipython().config:  # pragma: no cover
            return False
    except ImportError:
        return False
    return True


class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor):
                Tensor image of size (C, H, W) or batch of images (B, C, H, W)
                to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        m = torch.tensor(self.mean, device=tensor.device).reshape(-1, 1, 1)
        s = torch.tensor(self.std, device=tensor.device).reshape(-1, 1, 1)
        tensor.mul_(s).add_(m)
        return tensor


def get_aux_data(args):
    if 'core50' in args.scenario:
        return ImageNet128()
    elif 'mnist' in args.scenario:
        trs = Compose([Resize(32), ToTensor(), Normalize((0.1307,), (0.3081,))])
        return FashionMNIST(root=default_dataset_location('fashion_mnist'), download=True, transform=trs)
    elif 'cifar100' in args.scenario:
        return ImageNet32()
    elif 'cifar10' in args.scenario:
        return ImageNet32()
    else:
        raise ValueError(f"No available auxiliary data for scenario {args.scenario}")


def get_classifier_weights_from_args(args):
    gc = _default_get_classifier
    if 'resnet' in args.model:
        gc = lambda m: m.linear.weight
    elif 'mobile' in args.model:
        gc = lambda m: m.classifier[1].weight
    elif 'lenet' in args.model:
        gc = lambda m: m.classifier[0].weight
    return gc


def get_classifier_from_args(args):
    gc = _default_get_classifier
    if 'resnet' in args.model:
        gc = lambda m: m.linear
    elif 'mobile' in args.model:
        gc = lambda m: m.classifier[1]
    elif 'lenet' in args.model:
        gc = lambda m: m.classifier[0]
    return gc


class ModelClassifierWrapper(nn.Module):
    def __init__(self, model, get_classifier):
        super().__init__()
        self.model = model
        self.get_classifier = get_classifier

    @property
    def classifier(self):
        return self.get_classifier(self.model)

    def forward(self, x):
        return self.model(x)


def with_classifier_attribute(model, args):
    return ModelClassifierWrapper(model, get_classifier_weights_from_args(args))


def load_scenario(args, run_id):
    CURR_SEED = SEED_BENCHMARK_RUNS[run_id]
    core50_normalization = Normalize(mean=[0.485, 0.456, 0.406],
                  std=[0.229, 0.224, 0.225])
    core50_train_transforms = Compose([
        RandomHorizontalFlip(p=0.5),
        RandomCrop(size=128, padding=1),
        RandomRotation(15),
        ToTensor(),
        core50_normalization
    ])
    core50_eval_transforms = Compose([
        CenterCrop(size=128),
        ToTensor(), core50_normalization
    ])

    cifar100_train_transform = Compose([
        RandomCrop(32, padding=4),
        RandomHorizontalFlip(),
        RandomRotation(15),
        ToTensor(),
        Normalize((0.5071, 0.4865, 0.4409),
                  (0.2673, 0.2564, 0.2762))
    ])

    transforms = Compose([ToTensor(), Normalize((0.1307,), (0.3081,))])
    if args.model == 'lenet':  # LeNet wants 32x32 images
        transforms = Compose([Resize(32), ToTensor(), Normalize((0.1307,), (0.3081,))])

    if args.scenario == 'pmnist':
        scenario = PermutedMNIST(n_experiences=args.permutations, seed=CURR_SEED, train_transform=transforms, eval_transform=transforms)
    elif args.scenario == 'split_mnist':
        scenario = SplitMNIST(n_experiences=5, return_task_id=False, seed=CURR_SEED, train_transform=transforms, eval_transform=transforms)
    elif args.scenario == 'mt_mnist':
        mnist_train, mnist_test = _get_mnist_dataset()
        scenario = nc_benchmark(train_dataset=mnist_train, test_dataset=mnist_test,
            n_experiences=5, task_labels=True, seed=CURR_SEED,
            class_ids_from_zero_in_each_exp=False,
            train_transform=transforms, eval_transform=transforms)
    elif args.scenario == 'joint_mnist':
        scenario = SplitMNIST(n_experiences=1, return_task_id=False, seed=CURR_SEED, train_transform=transforms, eval_transform=transforms)
    elif args.scenario == 'joint_cifar10':
        scenario = SplitCIFAR10(n_experiences=1, return_task_id=False, seed=CURR_SEED)
    elif args.scenario == 'split_cifar10':
        scenario = SplitCIFAR10(n_experiences=5, return_task_id=False, seed=CURR_SEED)
    elif args.scenario == 'mt_cifar10':
        from avalanche.benchmarks.classic.ccifar10 import _default_cifar10_eval_transform, _default_cifar10_train_transform
        cifar_train, cifar_test = _get_cifar10_dataset(None)
        scenario = nc_benchmark(
            train_dataset=cifar_train,
            test_dataset=cifar_test,
            n_experiences=5,
            task_labels=True,
            seed=CURR_SEED,
            class_ids_from_zero_in_each_exp=False,
            train_transform=_default_cifar10_train_transform,
            eval_transform=_default_cifar10_eval_transform)
    elif args.scenario == 'joint_cifar100':
        scenario = SplitCIFAR100(n_experiences=1, return_task_id=False, seed=CURR_SEED, train_transform=cifar100_train_transform)
    elif args.scenario == 'split_cifar100' or args.scenario == 'scifar100_single_exp':
        scenario = SplitCIFAR100(n_experiences=10, return_task_id=False, seed=CURR_SEED, train_transform=cifar100_train_transform)
    elif args.scenario == 'mt_scifar100':
        cifar_train, cifar_test = _get_cifar100_dataset()
        scenario = nc_benchmark(
            train_dataset=cifar_train, test_dataset=cifar_test,
            n_experiences=10, task_labels=True, seed=CURR_SEED,
            class_ids_from_zero_in_each_exp=False,
            train_transform=cifar100_train_transform,
            eval_transform=_default_cifar100_eval_transform)
    elif args.scenario == 'core50_nc':
        scenario = CORe50(scenario='nc',
                          train_transform=core50_train_transforms,
                          eval_transform=core50_eval_transforms, run=run_id)
    elif args.scenario == 'joint_core50':
        core50nc = CORe50(scenario='nc')
        train_cat = AvalancheConcatDataset([e.dataset for e in core50nc.train_stream])
        test_cat = AvalancheConcatDataset([e.dataset for e in core50nc.test_stream])
        scenario = nc_benchmark(train_cat, test_cat, n_experiences=1, task_labels=False)
    elif args.scenario == 'nic_core50':
        scenario = CORe50(scenario='nic',
                          train_transform=core50_train_transforms,
                          eval_transform=core50_eval_transforms, run=run_id)
    elif args.scenario == 'ni_core50':
        scenario = CORe50(scenario='ni',
                          train_transform=core50_train_transforms,
                          eval_transform=core50_eval_transforms, run=run_id)
    elif args.scenario == 'nic_cifar100':
        scenario = NIC50Cifar100(seed=CURR_SEED)
    else:
        raise ValueError(f"Unknown scenario name: {args.scenario}")

    return scenario


def load_model(args) -> nn.Module:
    num_classes = 10
    if 'core50' in args.scenario:
        num_classes = 50

    orig_file = os.path.join(args.logdir, args.model, args.scenario, 'orig_model.pt')
    if os.path.exists(orig_file):
        print("loading from fixed initialization.")
        return torch.load(orig_file)

    if args.model == 'simple_mlp':
        return SimpleMLP(hidden_size=args.hs)
    elif args.model == 'simple_cnn':
        return SimpleCNN()
    elif args.model == 'lenet':
        input_channels = 1 if 'mnist' in args.scenario else 3
        if 'mt' in args.scenario:
            return MTLeNet5(10)
        return LeNet5(10, input_channels)
    elif args.model == 'resnet':
        if 'mt' in args.scenario:
            return MTResNet18(nclasses=num_classes)
        return ResNet18(nclasses=num_classes)
    elif args.model == 'masked_resnet':
        return MTMaskedResNet18(nclasses=num_classes)
    elif args.model == 'resnet_pretrained':
        return resnet18(pretrained=True)
    elif args.model == 'mobilenet_pretrained':
        model = mobilenet_v2(pretrained=True)
        # switch head to correct number of classes
        model.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(model.last_channel, num_classes),
        )
        return model
    elif args.model == 'mobilenet_imnet_fixed':
        model = mobilenet_v2(pretrained=True)
        for p in model.parameters():
            p.requires_grad = False
        # switch head to trainable classifier
        model.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(model.last_channel, num_classes),
        )
        return model
    else:
        raise ValueError("Wrong model name.")


def load_exmodels(args, scenario):
    curr_scenario = args.scenario
    expert_arch = args.model
    if args.model == 'masked_resnet':
        expert_arch = 'resnet'

    if args.scenario == 'mt_cifar10':
        curr_scenario = 'split_cifar10'
    if curr_scenario == 'nic_cifar100':
        logdir = '/raid/carta/ex_model_cl/logs/pret_models', args.model, args.scenario, f'run{args.run_id}'
        return ExModelNIC50Cifar100(logdir, SEED_BENCHMARK_RUNS[args.run_id]).trained_models
    try:
        models = []
        for i in range(len(scenario.train_stream)):
            model_fname = os.path.join('/raid/carta/ex_model_cl/logs/pret_models', expert_arch, curr_scenario,
                                       f'run{args.run_id}', f'model_e{i}.pt')
            model = torch.load(model_fname).to('cpu')
            model.eval()
            models.append(model)
    except FileNotFoundError:
        print(f"File not found: {model_fname}")
        print("please train separate models before launching ex-model experiment.")
        raise
    return models


def train_loop(log_dir, scenario, strategy, peval_on_test=True):
    for i, experience in enumerate(scenario.train_stream):
        if peval_on_test:
            evals = [[experience], scenario.test_stream]
        else:
            evals = [[experience]]
        strategy.train(
            experiences=experience,
            eval_streams=evals,
            expert_models=scenario.trained_models[i],
            pin_memory=True, num_workers=8)
        model_fname = os.path.join(log_dir,
                                   f'model_e{experience.current_experience}.pt')
        torch.save(strategy.model, model_fname)

        if isinstance(scenario, ExModelNIC50Cifar100):
            strategy.eval(scenario.cat_train_set)
        else:
            strategy.eval(scenario.train_stream[:])
        strategy.eval(scenario.test_stream[:])
    with open(os.path.join(log_dir, 'metrics.json'), 'w') as f:
        f.write(to_json(strategy.evaluator.get_all_metrics(), ignore_errors=True))
