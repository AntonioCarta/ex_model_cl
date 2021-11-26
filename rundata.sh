# SimpleMLP - permutedMNIST
python experiments/prepare_pretrained_models.py --model simple_mlp --scenario pmnist

# LeNet - JointMNIST, SplitMNIST
python experiments/prepare_pretrained_models.py --model lenet --scenario smnist --epochs 20 --lr 0.01
python experiments/prepare_pretrained_models.py --model lenet --scenario joint_mnist --epochs 100 --lr 0.01

# ResNet - Cifar10
python experiments/prepare_pretrained_models.py --model resnet --scenario joint_cifar10 --epochs 100 --lr 0.01
python experiments/prepare_pretrained_models.py --model resnet --scenario split_cifar10 --epochs 100 --lr 0.01
python experiments/prepare_pretrained_models.py --model lenet --scenario split_cifar10 --epochs 100 --lr 0.01

# ResNet pretrained - Cifar10
python experiments/prepare_pretrained_models.py --model resnet_pretrained --scenario joint_cifar10 --epochs 100 --lr 0.01

# ResNet pretrained - Cifar100
# hparams set in JointCIFAR100HParams
python experiments/prepare_pretrained_models.py --model resnet_pretrained --scenario joint_cifar100
python experiments/prepare_pretrained_models.py --model resnet_pretrained --scenario split_cifar100 --epochs 100 --lr 0.01
python experiments/prepare_pretrained_models.py --model resnet --scenario joint_cifar100
python experiments/prepare_pretrained_models.py --model resnet --scenario split_cifar100

# Core50-NC
python experiments/prepare_pretrained_models.py --scenario joint_core50 --model mobilenet_pretrained --epochs 40 --lr 0.001 --batch_size 128
python experiments/prepare_pretrained_models.py --scenario core50_nc --model mobilenet_pretrained --epochs 4 --lr 0.001 --batch_size 128
python experiments/prepare_pretrained_models.py --scenario nic_core50 --model mobilenet_pretrained --epochs 4 --lr 0.001 --batch_size 128
python experiments/prepare_pretrained_models.py --scenario ni_core50 --model mobilenet_pretrained --epochs 10 --lr 0.001 --batch_size 128

