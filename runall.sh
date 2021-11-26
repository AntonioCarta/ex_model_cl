######################
# SimpleMLP          #
######################
# splitMNIST, ensembling
python experiments/train_ex_model.py --model simple_mlp --scenario smnist --strategy model_average
python experiments/train_ex_model.py --model simple_mlp --scenario smnist --strategy model_ensemble
python experiments/train_ex_model.py --model simple_mlp --scenario smnist --strategy entropy_ensemble
# permutedMNIST, ensembling
python experiments/train_ex_model.py --model simple_mlp --scenario pmnist --strategy model_average
python experiments/train_ex_model.py --model simple_mlp --scenario pmnist --strategy model_ensemble
python experiments/train_ex_model.py --model simple_mlp --scenario pmnist --strategy entropy_ensemble


######################
# LeNet              #
######################
# SplitMNIST, ensembling
python experiments/train_ex_model.py --model lenet --scenario smnist --strategy model_average
python experiments/train_ex_model.py --model lenet --scenario smnist --strategy model_ensemble
python experiments/train_ex_model.py --model lenet --scenario smnist --strategy entropy_ensemble

# SplitMNIST - aux FMNIST
python experiments/train_ex_model.py --model lenet --scenario smnist --strategy aux_data --lr 0.1 --epochs 10

# SplitMNIST, sampling-based naive
python experiments/train_ex_model.py --model lenet --scenario smnist --strategy buffer_naive --buffer_size 500 --lr 0.1 --epochs 1000
python experiments/train_ex_model.py --model lenet --scenario smnist --strategy minversion_naive --buffer_size 5000 --buffer_iter 10000 --buffer_tau 20 --buffer_wd 0.01 --lr 0.1
python experiments/train_ex_model.py --model lenet --scenario smnist --strategy dimpression_naive --buffer_size 5000 --buffer_iter 10000 --buffer_tau 20 --buffer_wd 0.01 --lr 0.1

# SplitMNIST, sampling-based cumulative
python experiments/train_ex_model.py --model lenet --scenario smnist --strategy buffer_cumulative --buffer_size 500 --lr 0.1 --epochs 1000
python experiments/train_ex_model.py --model lenet --scenario smnist --strategy minversion_cumulative --buffer_size 5000 --buffer_iter 10000 --buffer_tau 20 --buffer_wd 0.01 --lr 0.1
python experiments/train_ex_model.py --model lenet --scenario smnist --strategy dimpression_cumulative --buffer_size 5000 --buffer_iter 10000 --buffer_tau 20 --buffer_wd 0.01 --lr 0.1

# SplitMNIST, sampling-based replay
python experiments/train_ex_model.py --model lenet --scenario smnist --strategy buffer_replay --buffer_size 500 --lr 0.1 --mem_size 5000 --epochs 1000
python experiments/train_ex_model.py --model lenet --scenario smnist --strategy minversion_replay --buffer_size 5000 --buffer_iter 10000 --buffer_tau 20 --buffer_wd 0.01 --lr 0.1 --mem_size 50000
python experiments/train_ex_model.py --model lenet --scenario smnist --strategy dimpression_replay --buffer_size 5000 --buffer_iter 10000 --buffer_tau 20 --buffer_wd 0.01 --lr 0.1 --mem_size 50000

# JointMNIST, sampling-based naive
python experiments/train_ex_model.py --model lenet --scenario joint_mnist --strategy buffer_naive --buffer_size 500 --lr 0.1 --epochs 10000
python experiments/train_ex_model.py --model lenet --scenario joint_mnist --strategy dimpression_naive --buffer_size 50000 --buffer_iter 10000 --buffer_tau 20 --buffer_wd 0.01 --lr 0.1
python experiments/train_ex_model.py --model lenet --scenario joint_mnist --strategy minversion_naive --buffer_size 50000 --buffer_tau 20 --buffer_wd 0.01 --lr 0.1 --buffer_iter 10000

# JointMNIST - aux FMNIST
python experiments/train_ex_model.py --model lenet --scenario joint_mnist --strategy aux_data --lr 0.1 --epochs 100


######################
# ResNet             #
######################
# CIFAR10, sampling-based cumulative
python experiments/train_ex_model.py --model resnet --scenario joint_cifar10 --strategy buffer_cumulative --buffer_size 500 --lr 0.1 --epochs 1000
python experiments/train_ex_model.py --model resnet --scenario joint_cifar10 --strategy minversion_cumulative --buffer_size 5000 --buffer_iter 10000 --buffer_tau 20 --buffer_wd 0.01 --lr 0.1
python experiments/train_ex_model.py --model resnet --scenario joint_cifar10 --strategy dimpression_cumulative --buffer_size 5000 --buffer_iter 10000 --buffer_tau 20 --buffer_wd 0.01 --lr 0.1

# CIFAR10, sampling-based naive
python experiments/train_ex_model.py --model resnet --scenario joint_cifar10 --strategy buffer_naive --buffer_size 500 --lr 0.1 --epochs 10000
python experiments/train_ex_model.py --model resnet --scenario joint_cifar10 --strategy dimpression_naive --buffer_size 5000 --buffer_iter 10000 --buffer_tau 20 --epochs 1000 --buffer_wd 0.01 --lr 0.1
python experiments/train_ex_model.py --model resnet --scenario joint_cifar10 --strategy minversion_naive --buffer_size 5000 --buffer_tau 20 --buffer_wd 0.01 --lr 0.1 --epochs 1000 --buffer_iter 10000

# SplitCIFAR100, sampling-based cumulative
python experiments/train_ex_model.py --model resnet --scenario split_cifar100 --strategy buffer_cumulative --buffer_size 500 --lr 0.1 --epochs 1000
python experiments/train_ex_model.py --model resnet --scenario split_cifar100 --strategy minversion_cumulative --buffer_size 500 --epochs 1000 --buffer_iter 10000 --buffer_tau 2 --epochs 1000 --buffer_wd 0.001 --lr 0.1
python experiments/train_ex_model.py --model resnet --scenario split_cifar100 --strategy dimpression_cumulative --buffer_size 500 --epochs 100 --buffer_iter 10000 --buffer_tau 20 --buffer_wd 0.01 --lr 0.1

python experiments/train_ex_model.py --model resnet --scenario split_cifar100 --strategy buffer_cumulative --buffer_size 500 --lr 0.1 --epochs 1000 --version class_mask
python experiments/train_ex_model.py --model resnet --scenario split_cifar100 --strategy minversion_cumulative --buffer_size 500 --epochs 1000 --buffer_iter 10000 --buffer_tau 2 --epochs 1000 --buffer_wd 0.001 --lr 0.1
python experiments/train_ex_model.py --model resnet --scenario split_cifar100 --strategy dimpression_cumulative --buffer_size 500 --epochs 1000 --buffer_iter 10000 --buffer_tau 2 --buffer_wd 0.001 --lr 0.1 --version v2
