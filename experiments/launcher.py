from experiments.experiment_management import GridSearchExp, \
    shutdown_experiment, init_experiment_args
from experiments.train_ex_model import main


if __name__ == '__main__':
    args = init_experiment_args()
    gs = GridSearchExp(main)
    gs(args)
    gs.wait()
    shutdown_experiment()
