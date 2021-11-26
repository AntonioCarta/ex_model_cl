import argparse
import copy

import torch
from time import sleep

import ray
import sys

import os

from exmodel.evaluation import YAMLConfig
from exmodel.utils import write_config_file, create_grid, \
    get_best_config, set_gpus


class Tee(object):
    def __init__(self, *files):
        self.files = files
        self.fileno = files[0].fileno

    def write(self, obj):
        for f in self.files:
            if not f.closed:
                f.write(obj)

    def flush(self):
        for f in self.files:
            if not f.closed:
                f.flush()

    @property
    def closed(self):
        return False


class RayExperiment:
    def __init__(self, results_fname=None, resume_ok=False, num_cpus=0, num_gpus=0):
        """ Generic experiment manager.

        Each experiment produces an artifact (file).
        If the artifact is already available the experiment does not run.
        This allows to start/stop/resume experiment freely.

        Think of it like a Makefile but in python.

        You only need to define the method `run`.

        :param results_fname: results filename.
        :param resume_ok: whether the experiment can be resumed or not.
        """
        self.results_fname = results_fname
        self.resume_ok = resume_ok
        self.num_cpus = num_cpus
        self.num_gpus = num_gpus

        self.f = None
        self._ray_obj_id = None
        self._done = None

    def __call__(self, config):
        assert not self._done  # can call only once.
        ray_init(config)
        logdir = config.logdir

        if self.results_fname is not None:
            # check for conditional execution
            # continue only if the final artifact does not exists already.
            res_file = os.path.join(logdir, self.results_fname)
            if os.path.exists(res_file):
                return None

        config_fname = os.path.join(logdir, 'config_file.yaml')
        if os.path.exists(config_fname) and not config.debug:
            if not self.resume_ok:
                print("Resume experiment not allowed.")
                sys.exit(-1)

            # configs should match to allow resume.
            prev_config_file = os.path.join(logdir, 'config_file.yaml')
            prev_config = YAMLConfig(prev_config_file)
            assert config == prev_config
        else:
            os.makedirs(logdir, exist_ok=True)
            write_config_file(config, config.logdir)

        @ray.remote(num_cpus=self.num_cpus,
                    num_gpus=self.num_gpus,
                    max_calls=1)
        def run_exp(argum):
            self.f = open(os.path.join(logdir, 'out.txt'), 'w', buffering=1)
            sys.stdout = Tee(sys.stdout, self.f)
            sys.stderr = sys.stdout
            self.run(argum)

        if config.no_ray:
            self.f = open(os.path.join(logdir, 'out.txt'), 'w', buffering=1)
            sys.stdout = Tee(sys.stdout, self.f)
            sys.stderr = sys.stdout
            self._ray_obj_id = self.run(config)
        else:
            self._ray_obj_id = run_exp.remote(config)

    def wait(self):
        if self._ray_obj_id is not None:
            ray.get(self._ray_obj_id)
            if self.f is not None:
                self.f.close()
        self._done = True

    def run(self, config):
        assert NotImplementedError()

    def result(self):
        assert NotImplementedError()


def run_configs_and_wait(base_exp, configs, stagger=None):
    rem_ids = []
    for config in configs:
        exp = copy.deepcopy(base_exp)
        exp(config)
        ids = exp._ray_obj_id

        if ids is not None:
            rem_ids.append(ids)
            if stagger is not None:
                sleep(stagger)
    n_jobs = len(rem_ids)
    print(f"Scheduled jobs: {n_jobs}")

    while rem_ids:
        done_ids, rem_ids = ray.wait(rem_ids, num_returns=1)
        for result_id in done_ids:
            ray.get(result_id)
            n_jobs -= 1
            print(f'Job {result_id} terminated. Jobs left: {n_jobs}')


class SingleRunExp(RayExperiment):
    def __init__(self, main, num_cpus=0, num_gpus=0):
        super().__init__('metrics.json', resume_ok=True,
                         num_cpus=num_cpus, num_gpus=num_gpus)
        self.main = main

    def run(self, config):
        if config.cuda and config.no_ray:
            print(f'Using GPUs {os.environ["CUDA_VISIBLE_DEVICES"]}')
        elif config.cuda:
            print(f'Using GPUs {ray.get_gpu_ids()}')
        else:
            print('Using CPUs')
        return self.main(config)


class GridSearchExp(RayExperiment):
    def __init__(self, main):
        super().__init__(None, resume_ok=True)
        self.main = main

    def run(self, config):
        config.run_id = 0
        orig_args = copy.deepcopy(config)
        grid_args = create_grid(config)

        # Model selection
        if len(grid_args) > 1:
            print(f"Grid search on {len(grid_args)} configurations")
            for grid_id, single_config in enumerate(grid_args):
                # create jobs
                curr_folder = os.path.join(orig_args.logdir, f'VAL{grid_id}')
                print(curr_folder)
                single_config.logdir = curr_folder
            exp = SingleRunExp(self.main, num_cpus=orig_args.cpus_per_job, num_gpus=orig_args.gpus_per_job)
            run_configs_and_wait(exp, grid_args)
            best_args = get_best_config(orig_args.logdir, metric_name=orig_args.metric_name)
        else:
            print(f"Single configuration. Skipping grid search.")
            best_args = copy.deepcopy(orig_args)

        # Assessment
        assess_args = []
        for i in range(best_args.assess_runs):
            single_config = copy.deepcopy(best_args)
            single_config.run_id = i
            single_config.logdir = os.path.join(orig_args.logdir, f'ASSESS{i}')
            assess_args.append(single_config)
        assess_exp = SingleRunExp(self.main, num_cpus=orig_args.cpus_per_job,
                                  num_gpus=orig_args.gpus_per_job)
        run_configs_and_wait(assess_exp, assess_args)


def ray_init(args):
    if ray.is_initialized() or args.no_ray:
        return

    if args.cuda:
        ray.init(num_cpus=args.max_cpus, num_gpus=args.max_gpus)
    elif os.environ.get('ip_head') is not None:
        assert os.environ.get('redis_password') is not None, "Missing redis password"
        ray.init(address=os.environ.get('ip_head'),
                 _redis_password=os.environ.get('redis_password'))
        print("Connected to Ray cluster.")
        print(f"Available nodes: {ray.nodes()}")
        args.gpus_per_job = 0
    else:
        ray.init(num_cpus=args.max_cpus)
        args.gpus_per_job = 0
        print(f"Started local ray instance.")

    assert ray.is_initialized(), "Error in initializing ray."


def shutdown_experiment():
    print('Shutting down ray. Please wait...')
    ray.shutdown()
    print('Ray closed.')


def init_experiment_args():
    global args, args
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', type=str, default='',
                        help='path to yaml configuration file')
    cmd_args = parser.parse_args()
    if cmd_args.config_file == '':
        raise ValueError('You must provide a config file.')
    args = YAMLConfig('./CONFIGS/default.yaml')
    args.update(cmd_args.config_file)
    # expand logdir name
    if args.debug:
        args.logdir = '/raid/carta/debug'
    args.logdir = os.path.expanduser(args.logdir)
    config_name = args.config_files[-1].split('/')[-1].split('.')[0]
    args.logdir = os.path.join(args.logdir, config_name)
    os.makedirs(args.logdir, exist_ok=True)
    torch.set_num_threads(args.max_cpus)
    if args.cuda:
        set_gpus(args.max_gpus)
    return args


__all__ = [
    'Tee',
    'RayExperiment',
    'run_configs_and_wait',
    'SingleRunExp',
    'GridSearchExp'
]