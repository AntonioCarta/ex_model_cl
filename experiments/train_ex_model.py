"""
    Train separate models on each experience to prepare ex-model scenario.
"""
import os

from torch.utils.data import random_split

from exmodel.evaluation.exmodel_samples import ExModelSamplePlugin
from exmodel.training.strategy_exml_distillation import AuxDataED, ReplayED, ModelInversionED, DataImpressionED, BufferPrecomputedED, GaussianNoiseED
from exmodel.utils import to_json

import torch
from avalanche.evaluation.metrics import accuracy_metrics, loss_metrics, \
    timing_metrics, forgetting_metrics
from avalanche.logging import TextLogger, TensorboardLogger
from avalanche.training.plugins import EvaluationPlugin
from exmodel.benchmarks import ExModelScenario, ExModelNIC50Cifar100
from experiments.utils import load_scenario, load_model, load_buffer_transform, \
    load_exmodels, train_loop, get_aux_data, get_classifier_weights_from_args, SEED_BENCHMARK_RUNS
from exmodel.evaluation.rich_text_logging import RichTextLogger


def train_ed_single_exp(args):
    """ Used to synthetic data generation parameters.

    :param args:
    :return:
    """
    log_dir = args.logdir
    if args.debug:
        args.buffer_size = 100
        args.buffer_iter = 100
    if args.no_ray:
        import setGPU  # ray selects

    # check if selected GPU is available or use CPU
    device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")
    print(f'Using device: {device}')
    # create scenario
    scenario = load_scenario(args, args.run_id)
    models = load_exmodels(args, scenario)
    scenario = ExModelScenario(scenario, models)
    buffer_transform = load_buffer_transform(args)

    # choose some metrics and evaluation method
    text_logger = TextLogger(open(os.path.join(log_dir, 'log.txt'), 'w'))
    tb_logger = TensorboardLogger(os.path.join(log_dir, 'tb'))
    rtl = RichTextLogger(log_dir, interactive=args.debug)
    eval_plugin = EvaluationPlugin(
        accuracy_metrics(epoch=True, experience=True, stream=True),
        loss_metrics(epoch=True, experience=True, stream=True),
        timing_metrics(epoch=True),
        forgetting_metrics(experience=True, stream=True),
        ExModelSamplePlugin(mode='train', n_rows=10, n_cols=10, group=True),
        loggers=[text_logger, tb_logger, rtl])

    model = load_model(args)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    criterion = torch.nn.CrossEntropyLoss()
    kwargs = {
        'model': model,
        'optimizer': optimizer,
        'criterion': criterion,
        'device': device,
        'train_mb_size': args.train_mb_size,
        'eval_mb_size': 512,
        'evaluator': eval_plugin,
        'eval_every': 1,
        'bias_normalization': args.bias_normalization,
        'reset_model': args.reset_model,
        'loss_type': args.loss_type,
        'ce_loss': args.ce_loss
    }

    if args.strategy == 'minversion_ed':
        if args.debug:
            args.num_iter_per_exp = 10
        strat_args = {
            'buffer_size': args.buffer_size,
            'num_iter_per_exp': args.num_iter_per_exp,
            'buffer_iter': args.buffer_iter,
            'buffer_lr': args.buffer_lr,
            'buffer_wd': args.buffer_wd,
            'transform': buffer_transform,
            'lambda_blur': args.buffer_blur,
            'lambda_bns': args.buffer_bns,
            'buffer_mb_size': args.buffer_mb_size,
            'temperature': args.buffer_temperature
        }
        strategy = ModelInversionED(**strat_args, **kwargs)
    elif args.strategy == 'dimpression_ed':
        if args.debug:
            args.num_iter_per_exp = 10
        strat_args = {
            'buffer_size': args.buffer_size,
            'num_iter_per_exp': args.num_iter_per_exp,
            'buffer_iter': args.buffer_iter,
            'buffer_lr': args.buffer_lr,
            'buffer_wd': args.buffer_wd,
            'transform': buffer_transform,
            'buffer_beta': args.buffer_beta,
            'get_classifier': get_classifier_weights_from_args(args),
            'lambda_blur': args.buffer_blur,
            'lambda_bns': args.buffer_bns,
            'buffer_mb_size': args.buffer_mb_size,
            'temperature': args.buffer_temperature
        }
        strategy = DataImpressionED(**strat_args, **kwargs)
    else:
        raise ValueError(f"unrecognized strategy {args.strategy}")

    # train on the selected scenario with the chosen strategy
    print('Starting experiment...')
    for i, experience in enumerate(scenario.train_stream):
        strategy.train(
            experiences=experience,
            eval_streams=[[experience], scenario.test_stream[0]],
            expert_models=scenario.trained_models[i],
            pin_memory=True, num_workers=8)
        model_fname = os.path.join(log_dir, f'model_e{experience.current_experience}.pt')
        torch.save(strategy.model, model_fname)
        break
    strategy.eval(scenario.train_stream[0])
    strategy.eval(scenario.test_stream[0])
    with open(os.path.join(log_dir, 'metrics.json'), 'w') as f:
        f.write(to_json(strategy.evaluator.get_all_metrics(), ignore_errors=True))


def train_ed(args):
    log_dir = args.logdir

    if 'mt' in args.scenario:
        # can't work because experts don't use a multihead
        assert args.init_from_expert == False
        # not needed since heads are independent
        assert args.bias_normalization == False

    # check if selected GPU is available or use CPU
    device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")
    print(f'Using device: {device}')
    # create scenario
    if args.scenario == 'nic_cifar100':
        experts_dir = os.path.join('/raid/carta/ex_model_cl/logs/pret_models', args.model, 'split_cifar100')
        scenario = ExModelNIC50Cifar100(experts_dir, SEED_BENCHMARK_RUNS[args.run_id])
    else:
        scenario = load_scenario(args, args.run_id)
        models = load_exmodels(args, scenario)
        scenario = ExModelScenario(scenario, models)
    buffer_transform = load_buffer_transform(args)

    # if False:
    #     if not args.debug:
    # for i, model in enumerate(scenario.trained_models):
    #     model.to('cuda')
    #     acc = Accuracy()
    #     if ('core50' in args.scenario) or (args.scenario == 'nic_cifar100'):
    #         data = scenario.test_stream[0].dataset
    #     else:
    #         data = scenario.test_stream[i].dataset
    #
    #     for x, y, t in DataLoader(data, batch_size=args.train_mb_size,
    #                               pin_memory=True, num_workers=8):
    #         x, y, t = x.to('cuda'), y.to('cuda'), t.to('cuda')
    #         y_pred = model(x)
    #         acc.update(y_pred, y, t)
    #     print(f"(i={i}) Original model accuracy: {acc.result()}")
    #     model.to('cpu')

    # choose some metrics and evaluation method
    text_logger = TextLogger(open(os.path.join(log_dir, 'log.txt'), 'w'))
    tb_logger = TensorboardLogger(os.path.join(log_dir, 'tb'))
    rtl = RichTextLogger(log_dir, interactive=args.debug)
    eval_plugin = EvaluationPlugin(
        accuracy_metrics(epoch=True, experience=True, stream=True, minibatch=True),
        loss_metrics(epoch=True, experience=True, stream=True),
        timing_metrics(epoch=True),
        forgetting_metrics(experience=True, stream=True),
        ExModelSamplePlugin(mode='train', n_rows=10, n_cols=10, group=True),
        loggers=[text_logger, tb_logger, rtl])

    model = load_model(args)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    criterion = torch.nn.CrossEntropyLoss()
    kwargs = {
        'model': model,
        'optimizer': optimizer,
        'criterion': criterion,
        'device': device,
        'train_mb_size': args.train_mb_size,
        'eval_mb_size': 512,
        'evaluator': eval_plugin,
        'eval_every': 1,
        'bias_normalization': args.bias_normalization,
        'reset_model': args.reset_model,
        'loss_type': args.loss_type,
        'ce_loss': args.ce_loss,
        'plugins': [],  # [ProfilerPlugin()],
        'num_iter_per_exp': args.num_iter_per_exp,
        'init_from_expert': args.init_from_expert
    }
    if args.strategy == 'aux_data_ed':
        aux_data = get_aux_data(args)
        if args.debug:
            removed_els = len(aux_data) - 1000
            aux_data, _ = random_split(aux_data, [1000, removed_els])
        strat_args = {
            'aux_data': aux_data
        }
        strategy = AuxDataED(**strat_args, **kwargs)
    elif args.strategy == 'replay_ed':
        if args.debug:
            args.num_iter_per_exp = 10
        strat_args = {
            'buffer_size': args.buffer_size,
        }
        strategy = ReplayED(**strat_args, **kwargs)
    elif args.strategy == 'gn_ed':
        if args.debug:
            args.num_iter_per_exp = 10
        strat_args = {
            'img_shape': scenario.train_stream[0].dataset[0][0].shape
        }
        strategy = GaussianNoiseED(**strat_args, **kwargs)
    elif args.strategy == 'minversion_ed':
        if args.debug:
            args.num_iter_per_exp = 10
        strat_args = {
            'buffer_size': args.buffer_size,
            'buffer_iter': args.buffer_iter,
            'buffer_lr': args.buffer_lr,
            'buffer_wd': args.buffer_wd,
            'transform': buffer_transform,
            'lambda_blur': args.buffer_blur,
            'lambda_bns': args.buffer_bns,
            'buffer_mb_size': args.buffer_mb_size,
            'temperature': args.buffer_temperature
        }
        strategy = ModelInversionED(**strat_args, **kwargs)
    elif args.strategy == 'dimpression_ed':
        if args.debug:
            args.num_iter_per_exp = 10
        strat_args = {
            'buffer_size': args.buffer_size,
            'buffer_iter': args.buffer_iter,
            'buffer_lr': args.buffer_lr,
            'buffer_wd': args.buffer_wd,
            'transform': buffer_transform,
            'buffer_beta': args.buffer_beta,
            'get_classifier': get_classifier_weights_from_args(args),
            'lambda_blur': args.buffer_blur,
            'lambda_bns': args.buffer_bns,
            'buffer_mb_size': args.buffer_mb_size,
            'temperature': args.buffer_temperature
        }
        strategy = DataImpressionED(**strat_args, **kwargs)
    elif args.strategy == 'from_path':
        if args.debug:
            args.num_iter_per_exp = 10
        full_buffer_path = os.path.join(
            '/raid/carta/ex_model_cl/logs/pret_models', args.model, args.scenario, f'run{args.run_id}', args.buffer_path)
        strat_args = {
            'logdir': full_buffer_path,
            'transform': buffer_transform
        }
        strategy = BufferPrecomputedED(**strat_args, **kwargs)
    else:
        raise ValueError(f"unrecognized strategy {args.strategy}")
    # train on the selected scenario with the chosen strategy
    print('Starting experiment...')
    peval_on_test = ('nic' not in args.scenario)
    train_loop(log_dir, scenario, strategy, peval_on_test=peval_on_test)


def main(args):
    if args.experiment_type == 'train_ed':
        train_ed(args)
    elif args.experiment_type == 'train_ed_single_exp':
        train_ed_single_exp(args)
    else:
        assert False, "Unknown experiment type."
