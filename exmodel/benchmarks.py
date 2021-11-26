from typing import Any

from torchvision import transforms
from torchvision.datasets import ImageNet

from avalanche.benchmarks.datasets import default_dataset_location

SEED_BENCHMARK_RUNS = [1234, 2345, 3456, 5678, 6789]  # 5 different seeds to randomize class orders


class ExModelScenario:
    def __init__(self, original_scenario, trained_models) -> None:
        """ Ex-model scenario.
        Each experience is a quadruple
        <original_exp, trained_model, generator, buffer>.

        The original experience should NOT BE USED DURING TRAINING.
        Instead, data must be extracted from the original model, generator,
        or buffers.

        :param original_scenario:
        :param trained_models: list of filenames
        """
        mm = []
        for model in trained_models:
            model = model.eval().to('cpu')
            mm.append(model)

        self.scenario = original_scenario
        self.trained_models = mm
        self.train_stream = original_scenario.train_stream
        self.test_stream = original_scenario.test_stream


class ImageNet32(ImageNet):
    def __init__(self, root: str = None, split: str = 'train', **kwargs: Any):
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        if root is None:
            root = default_dataset_location('imagenet')

        if split == 'train':
            transform = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.Resize(32),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])
        else:
            transform = transforms.Compose([
                transforms.Scale(256),
                transforms.CenterCrop(224),
                transforms.Resize(32),
                transforms.ToTensor(),
                normalize,
            ])
        super().__init__(root, split, transform=transform, **kwargs)


class ImageNet128(ImageNet):
    def __init__(self, root: str = None, split: str = 'train', **kwargs: Any):
        """ Used as auxiliary data for ex-model CL with COrE50 (128x128 images). """
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        if root is None:
            root = default_dataset_location('imagenet')

        if split == 'train':
            transform = transforms.Compose([
                transforms.RandomSizedCrop(224),
                transforms.Resize(128),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])
        else:
            transform = transforms.Compose([
                transforms.Scale(256),
                transforms.CenterCrop(224),
                transforms.Resize(128),
                transforms.ToTensor(),
                normalize,
            ])
        super().__init__(root, split, transform=transform, **kwargs)
