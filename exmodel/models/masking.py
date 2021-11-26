import torch

from avalanche.benchmarks.utils import AvalancheDataset
from avalanche.benchmarks.utils.dataset_utils import ConstantSequence
from avalanche.models import MultiTaskModule


class MultiTaskMasking(MultiTaskModule):
    def __init__(self, logits_input=True, num_inital_units=2):
        """ Output unit masking.

        :param logits_input:
        :param num_inital_units:
        """
        super().__init__()
        self.logits_input = logits_input
        self.num_initial_units = num_inital_units
        self.mask_used = {'0': torch.zeros(num_inital_units, dtype=torch.bool)}
        self.max_classes = 2

    def to(self, device):
        for k in self.mask_used.keys():
            self.mask_used[k] = self.mask_used[k].to(device)

    def maybe_grow_mask(self, task_label, num_units):
        tkey = str(task_label)
        if not tkey in self.mask_used:
            self.mask_used[tkey] = torch.zeros(
                self.num_initial_units, dtype=torch.bool,
                device=self.mask_used['0'].device)

        curr_mask = self.mask_used[tkey]
        if num_units > curr_mask.shape[0]:
            new_mask = torch.zeros(num_units, dtype=torch.bool, device=curr_mask.device)
            new_mask[:curr_mask.shape[0]] = curr_mask
            self.mask_used[tkey] = new_mask

    def adaptation(self, dataset: AvalancheDataset = None):
        """ Called by Avalanche training loops. """
        task_label = dataset.targets_task_labels
        if isinstance(task_label, ConstantSequence):
            task_label = task_label[0]
        else:
            task_label = set(task_label)
            assert len(task_label) == 1
            task_label = list(task_label)[0]

        curr_classes = list(set(dataset.targets))
        self.maybe_grow_mask(task_label, max(curr_classes) + 1)
        self.mask_used[str(task_label)][curr_classes] = 1

    def forward_single_task(self, x: torch.Tensor, task_label: int) -> torch.Tensor:
        tkey = str(task_label)
        if self.logits_input:  # unseen units should be -inf
            x[:, ~self.mask_used[tkey]] = -1.e3
        else:  # probabilities input
            x[:, ~self.mask_used[tkey]] = 0
        return x
