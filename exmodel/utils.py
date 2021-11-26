import json
from json import JSONDecodeError

import numpy
import torch
import copy
import os
import re
from shutil import copyfile
from pandas import read_csv
import numpy as np
import yaml
from sklearn.model_selection import ParameterGrid

from exmodel.evaluation import YAMLConfig

INDENT = 3
SPACE = " "
NEWLINE = "\n"


# Changed basestring to str, and dict uses items() instead of iteritems().
def to_json(o, level=0, ignore_errors=False):
    """ pretty-print json.
    source: https://stackoverflow.com/questions/10097477/python-json-array-newlines
    :param o:
    :param level:
    :return:
    """
    # TODO: fix NaN encoding from 'nan' to 'NaN'. 'nan' is non-standard and crashes during loading.
    ret = ""
    if isinstance(o, dict):
        ret += "{" + NEWLINE
        comma = ""
        for k, v in o.items():
            ret += comma
            comma = ",\n"
            ret += SPACE * INDENT * (level + 1)
            ret += '"' + str(k) + '":' + SPACE
            ret += to_json(v, level + 1, ignore_errors)

        ret += NEWLINE + SPACE * INDENT * level + "}"
    elif isinstance(o, str):
        ret += '"' + o + '"'
    elif isinstance(o, list):
        ret += "[" + ",".join([to_json(e, level + 1, ignore_errors) for e in o]) + "]"
    # Tuples are interpreted as lists
    elif isinstance(o, tuple):
        ret += "[" + ",".join(to_json(e, level + 1, ignore_errors) for e in o) + "]"
    elif isinstance(o, bool):
        ret += "true" if o else "false"
    elif isinstance(o, int):
        ret += str(o)
    elif isinstance(o, float):
        ret += '%.7g' % o
    elif isinstance(o, numpy.ndarray) and numpy.issubdtype(o.dtype, numpy.integer):
        ret += "[" + ','.join(map(str, o.flatten().tolist())) + "]"
    elif isinstance(o, numpy.ndarray) and numpy.issubdtype(o.dtype, numpy.inexact):
        ret += "[" + ','.join(map(lambda x: '%.7g' % x, o.flatten().tolist())) + "]"
    elif o is None:
        ret += 'null'
    elif ignore_errors:
        # we do not recognize the type but we don't want to raise an error.
        ret = '"<not serializable>"'
    else:
        # Unknown type. Raise error.
        raise TypeError("Unknown type '%s' for json serialization" % str(type(o)))
    return ret


def load_ex_models(log_dir, num_experiences):
    try:
        models = []
        for i in range(num_experiences):
            model_fname = os.path.join(log_dir, f'model_e{i}.pt')
            models.append(torch.load(model_fname))
        return models
    except FileNotFoundError:
        print("please train separate models before launching ex-model experiment.")
        raise


def set_gpus(num_gpus):
    try:
        import gpustat
    except ImportError:
        print("gpustat module is not installed. No GPU allocated.")

    try:
        selected = []

        stats = gpustat.GPUStatCollection.new_query()

        for i in range(num_gpus):

            ids_mem = [res for res in map(lambda gpu: (int(gpu.entry['index']),
                                          float(gpu.entry['memory.used']) /\
                                          float(gpu.entry['memory.total'])),
                                      stats) if str(res[0]) not in selected]

            if len(ids_mem) == 0:
                # No more gpus available
                break

            best = min(ids_mem, key=lambda x: x[1])
            bestGPU, bestMem = best[0], best[1]
            # print(f"{i}-th best GPU is {bestGPU} with mem {bestMem}")
            selected.append(str(bestGPU))

        print("Setting GPUs to: {}".format(",".join(selected)))
        os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
        os.environ['CUDA_VISIBLE_DEVICES'] = ",".join(selected)
    except BaseException as e:
        print("GPU not available: " + str(e))


def write_config_file(args, result_folder):
    """
    Write yaml configuration file inside result folder
    """
    with open(os.path.join(result_folder, 'config_file.yaml'), 'w') as f:
        yaml.dump(dict(vars(args)), f)


def create_result_folder(result_folder):
    '''
    Set plot folder by creating it if it does not exist.
    '''
    result_folder = os.path.expanduser(result_folder)
    return result_folder


def get_device(cuda):
    '''
    Choose device: cpu or cuda
    '''

    mode = 'cpu'
    if cuda and torch.cuda.is_available():
        mode = 'cuda'
    device = torch.device(mode)

    return device


def save_model(model, modelname, base_folder, path_save_models='saved_models', version=''):
    '''
    :param version: specify version of the model.
    Usually used to represent the model when trained after step 'version'
    '''

    torch.save(model.state_dict(), os.path.join(
        os.path.expanduser(base_folder),
        path_save_models, modelname+version+'.pt'))


def load_model(model, modelname, device, base_folder, path_save_models='saved_models', version=''):
    check = torch.load(os.path.join(
        os.path.expanduser(base_folder),
        path_save_models, modelname+version+'.pt'), map_location=device)

    model.load_state_dict(check)

    model.eval()

    return model


def create_grid(args, grid_arg='grid'):
    """
    Create grid search by returning a list of args.
    :parameter args: argument parser result
    :parameter grid_arg: field of `args` which contains
        a dictionary of
        'parameter name': list of possible values
    :return: list of configurations, one for each
        element in the grid search
    """

    try:
        grid = ParameterGrid(getattr(args, grid_arg))
    except AttributeError:
        print("Running without grid search")
        return [args]

    final_grid = []
    for el in grid:
        conf = copy.deepcopy(args)
        for k, v in el.items():
            conf.__dict__[k] = v
        final_grid.append(conf)

    return final_grid


def compute_average_eval_accuracy(folder, eval_result_name='metrics.json',
                                  metric_name='Top1_Acc_Stream/eval_phase/test_stream/Task000'):
    """
    Return average and std accuracy over all experiences
    after training on all experiences.
    """

    cur_file = os.path.join(folder, eval_result_name)
    with open(cur_file) as f:
        # fix NaN encoding for bugged JSON config
        j = f.read()
        j = re.sub(r'\bnan\b', 'NaN', j)

        try:
            data = json.loads(j)
        except JSONDecodeError as e:
            print(f"Error decoding json file: {cur_file}")
            print(j)
            raise e

    m_eval = data[metric_name][1][-1]
    return m_eval, 0


def compute_average_training_accuracy(folder,
                                      training_result_name='training_results.csv'):
    """
    Return average and std accuracy over all experiences
    after the last training epoch.
    """

    cur_file = os.path.join(folder, training_result_name)
    data = read_csv(cur_file)
    # select last epoch
    data = data[data['epoch'] == data['epoch'].max()]
    data = data['val_accuracy'].values

    # both are array of 2 elements (loss, acc)
    acc = np.average(data, axis=0)
    acc_std = np.std(data, axis=0)

    return acc, acc_std


def get_best_config(result_folder,
                    val_folder_name='VAL',
                    config_filename='config_file.yaml',
                    metric_name='Top1_Acc_Stream/eval_phase/test_stream/Task000'):
    """
    Choose best config from a specific result folder containing
    model selection results. It produces a `best_config.yaml`
    file in the project root folder.
    :return: parsed args from the best configuration
    """

    best_config_filename = 'best_config.yaml'

    # find folders with format 'VAL{number}'
    ids = [str(el) for el in range(10)]
    dirs = [el for el in os.listdir(result_folder)
            if os.path.isdir(os.path.join(result_folder, el))
            and el.startswith(val_folder_name)
            and el[-1] in ids]

    best_dir = None
    best_acc = 0
    for dir_path in dirs:
        acc, _ = compute_average_eval_accuracy(os.path.join(result_folder, dir_path), metric_name=metric_name)
        if acc > best_acc:
            best_dir = dir_path
            best_acc = acc
    assert best_dir is not None, "Error in retrieving best accuracy"

    copyfile(os.path.join(result_folder, best_dir, config_filename),
             os.path.join(result_folder, best_config_filename))

    best_config = YAMLConfig(os.path.join(result_folder, best_config_filename))

    return best_config


__all__ = [
    'set_gpus',
    'get_device',
    'write_config_file',
    'create_result_folder',
    'save_model',
    'load_model',
    'create_grid',
    'compute_average_eval_accuracy',
    'compute_average_training_accuracy',
    'get_best_config',
    'get_strategy'
]