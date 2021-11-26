"""
    Train separate models on each experience to prepare ex-model scenario.
"""
import sys
sys.path.append('/home/carta/avalanche')

try:
    import setGPU
except:
    pass

import copy

import os

from exmodel.utils import to_json
from experiments.utils import load_scenario, load_model

import argparse

import torch
from avalanche.training.strategies import Naive
from avalanche.evaluation.metrics import loss_metrics, ExperienceAccuracy, accuracy_metrics
from avalanche.logging import InteractiveLogger
from avalanche.training.plugins import EvaluationPlugin
import json


def main(args):
    log_dir_base = os.path.join(args.logdir, args.model, args.scenario + args.version)
    os.makedirs(log_dir_base, exist_ok=args.debug)

    with open(log_dir_base + '/sysargs.txt', 'w')as f:
        f.write(' '.join(sys.argv[1:]))

    for run_id in range(args.runs):
        if len(args.version) > 0:
            args.version = '_' + args.version
        log_dir = os.path.join(log_dir_base, f'run{run_id}')
        if args.debug:
            log_dir = os.path.join(args.logdir, 'debug')
        os.makedirs(log_dir, exist_ok=args.debug)

        device = 'cuda'
        print(f'Using device: {device}')

        # create scenario
        scenario = load_scenario(args, run_id)
        plugs = []

        # train on the selected scenario with the chosen strategy
        print('Starting experiment...')
        results = []
        orig_model = load_model(args).to(device)
        model_fname = os.path.join(log_dir, f'orig_model.pt')
        torch.save(orig_model, model_fname)

        for i, train_exp in enumerate(scenario.train_stream):
            # choose some metrics and evaluation method
            main_metric = ExperienceAccuracy()
            interactive_logger = InteractiveLogger()
            eval_plugin = EvaluationPlugin(
                main_metric,
                accuracy_metrics(epoch=True, stream=True),
                loss_metrics(epoch=True, experience=True, stream=True),
                loggers=[interactive_logger])

            if 'core50_nc' in args.scenario:
                assert len(scenario.test_stream) == 1
                evals = [[train_exp]]
            elif args.scenario in ['nic_core50', 'ni_core50']:
                assert len(scenario.test_stream) == 1
                evals = [[train_exp], [scenario.test_stream[0]]]
            else:
                evals = [[train_exp], [scenario.test_stream[i]]]

            model = copy.deepcopy(orig_model)
            optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
            criterion = torch.nn.CrossEntropyLoss()
            strategy = Naive(model, optimizer, criterion, train_epochs=args.epochs,
                             device=device, train_mb_size=args.batch_size, plugins=plugs,
                             evaluator=eval_plugin, eval_mb_size=1000, eval_every=1)

            strategy.train(train_exp, eval_streams=evals,
                           num_workers=16, pin_memory=True)
            model_fname = os.path.join(log_dir, f'model_e{train_exp.current_experience}.pt')
            torch.save(model, model_fname)
            results.append(main_metric.result())

            with open(os.path.join(log_dir, f'm{i}.json'), 'w') as f:
                f.write(to_json(strategy.evaluator.get_all_metrics()))
        with open(os.path.join(log_dir, 'accs.json'), 'w') as f:
            json.dump({'models_accuracy': results}, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', type=bool, default=False)
    parser.add_argument('--runs', type=int, default=5)
    parser.add_argument('--logdir', type=str, default='/raid/carta/ex_model_cl/logs/pret_models', help='Directory for logging.')
    parser.add_argument('--version', type=str, default='', help='Versioning for same scenario.')
    parser.add_argument('--scenario', type=str, default='joint_mnist')
    parser.add_argument('--lr', type=float, default=0.1, help='Learning rate.')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--hs', type=int, default=256, help='MLP hidden size.')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs.')
    parser.add_argument('--permutations', type=int, default=20, help='Number of experiences in Permuted MNIST.')
    parser.add_argument('--model', type=str, default='lenet')
    args = parser.parse_args()

    main(args)
