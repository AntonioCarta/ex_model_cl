# Ex-model CL
Ex-model continual learning is a setting where a stream of experts (i.e. model's parameters) is available and a CL model learns from them without access to the original data.

![ExML scenario](https://github.com/AntonioCarta/ex_model_cl/blob/main/exml.PNG)

**NOTE: This repository is a work in progress. It's a heavy refactoring of the code from our internal repository to make it easier to understand and reuse by other researchers. At some point we plan to integrate the strategies and benchmarks directly in Avalanche.**

The module `exmodel` follows Avalanche's structure:
- **benchmarks**: `ExModelScenario` adds an attribute `trained_models` to the benchmarks. The original `train_stream` and `test_stream` are available for evaluation purposes (they are assumed private by the scenario).
- **models**: custom `nn.Module`s and baseline architectures used in the experiments.
- **evaluation**: loggers and metrics.
- **training**: training algorithms. The ex-model distillation strategy is here. 

The folder `experiments` contains the code to run the experiments. The main is in `launcher.py`, while the training function is in `train_ex_model.py`.

## Install Dependencies
```
conda env create -f environment.yml
```
avalanche must be installed separately.
This repository used Avalanche pre-release, aligned with the master branch. You can use commit `a299bd4`.

## Run Experiments
to launch an experiment run:
```
python experiments/launcher.py --config CONFIGS/debug.yaml
```
The directory `CONFIGS` contains the configuration already setup for you.
To run the experiments you may need to change the logs and data folders in the `CONFIGS` yaml files.

## Known issues
The table printed by the rich-based logger sometimes misalign metric values (when there are missing values). The textual logger and json files are all correct.
