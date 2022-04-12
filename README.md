# Ex-model CL
Ex-model continual learning is a setting where a stream of experts (i.e. model's parameters) is available and a continual learning model learns from them without access to the original data.

![ExML scenario](https://github.com/AntonioCarta/ex_model_cl/blob/main/exml.PNG)

This is based on our [recent work](https://arxiv.org/abs/2112.06511) on continual learning from pretrained models. This repository provides a snapshot of the codebase at the time of publication. If you want to test your own strategies, the benchmarks and pretrained models are available directly in [Avalanche](https://github.com/ContinualAI/avalanche), and you should use them instead of this repository.

We plan to add also the Ex-Model Distillation inside Avalanche in the future.

**NOTE: This repository is a heavy refactoring of the original codebase which was used to run the experiment. The refactoring was necessary to make it easier to understand and reuse by other researchers. However, due to the high variance of the experiements, there may be slight differences in the results compared to the paper.**

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
This repository used Avalanche pre-release, commit `a299bd4`.

## Run Experiments
to launch an experiment run:
```
python experiments/launcher.py --config CONFIGS/debug.yaml
```
The directory `CONFIGS` contains the configuration already setup for you.
To run the experiments you may need to change the logs and data folders in the `CONFIGS` yaml files.

## Known issues
The table printed by the rich-based logger sometimes misalign metric values (when there are missing values). The textual logger and json files are all correct.

## Citation
If you find this useful consider citing:
```
Carta, A., Cossu, A., Lomonaco, V., & Bacciu, D. (2021). Ex-Model: Continual Learning from a Stream of Trained Models. arXiv preprint arXiv:2112.06511.
```
bibtex:
```
@article{carta2021ex,
  title={Ex-Model: Continual Learning from a Stream of Trained Models},
  author={Carta, Antonio and Cossu, Andrea and Lomonaco, Vincenzo and Bacciu, Davide},
  journal={arXiv preprint arXiv:2112.06511},
  year={2021}
}
```