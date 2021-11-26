import torch
from torch import nn
import torch.nn.functional as F

from avalanche.benchmarks import SplitMNIST
from avalanche.models import DynamicModule, IncrementalClassifier


class MaskedClassifier(DynamicModule):
    def __init__(self, classifier, out_features):
        super().__init__()
        self.mask = torch.zeros(out_features, dtype=torch.bool)
        self.classifier = classifier

    def train_adaptation(self, dataset):
        curr_classes = set(dataset.targets)
        self.mask[list(curr_classes)] = 1

    def forward(self, x):
        x = self.classifier(x)
        x[..., ~self.mask] = -10 ** 6
        return x


class EntropyWeighting(DynamicModule):
    def __init__(self, stream):
        """ Uses the entropy to scale outputs. """
        super().__init__()
        self.masks = []
        self.init_from_stream(stream)

    def init_from_stream(self, stream):
        """ Initialize masks from a complete stream. """
        for exp in stream:
            self.masks.append(list(set(exp.dataset.targets)))

    def forward(self, x):
        outs = []
        es = []
        for mask in self.masks:
            out = x
            res = torch.zeros_like(out) - 10 ** 6
            res[:, mask] = out[:, mask]
            outs.append(res)

            # score
            px = F.softmax(out[:, mask], dim=-1)
            score = -(torch.log(px)).mean(dim=-1)
            es.append(score)

        outs = torch.stack(outs)
        entropy = torch.stack(es)
        return (outs * entropy.unsqueeze(-1)).sum(dim=0)


class CosineClassifier(IncrementalClassifier):
    def __init__(self, in_features, initial_out_features=2):
        """ Classifier based on cosine similarity.

        TODO: add equation

        First proposed in LUCIR: TODO: add ref

        the number of output features is optional since the module
        automatically adds new units whenever new classes are
        encountered.

        :param in_features: number of input features.
        :param initial_out_features: initial number of classes (can be
            dynamically expanded).
        """
        super().__init__(in_features, initial_out_features)
        self.scale = nn.Parameter(torch.tensor(1.))

    def forward(self, x, **kwargs):
        assert len(x.shape) == 2
        y = super().forward(x, **kwargs)
        y_norm = torch.linalg.norm(y, dim=1)
        return self.scale * (y / y_norm)


if __name__ == '__main__':
    model = MaskedClassifier(784, 10)
    bench = SplitMNIST(5)

    mbatch = torch.randn(1, 784)
    for exp in bench.train_stream:

        model.train_adaptation(exp.dataset)
        print(model.mask)
        print(model(mbatch))
