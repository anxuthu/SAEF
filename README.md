# SAEF
Step-ahead local error feedback.

## Software Requirements

Python 3 and PyTorch 1.4

## Experiments

### CIFAR-100

+ SGDM: `python main.py --a resnet56 --dataset cifar100 --method va`
+ EF-SGDM: `python main.py --a resnet56 --dataset cifar100 --method ef-Sign -cb -sec`
+ SAEF-SGDM: `python main.py --a resnet56 --dataset cifar100 --method r-Sign -cb -sec`

### IMAGENET

+ SGDM: `NCCL_IB_DISABLE=1 python main.py --dataset imagenet --data /export/Data/ILSVRC2012/ -p 5005 -a resnet50 -wd 1e-4 --epochs 90 -ds 30 60 -b 256 --method va --spar 0.1 -cb -sec`
+ EF-SGDM: `NCCL_IB_DISABLE=1 python main.py --dataset imagenet --data /export/Data/ILSVRC2012/ -p 5005 -a resnet50 -wd 1e-4 --epochs 90 -ds 30 60 -b 256 --method ef-TopK --spar 0.1 -cb -sec`
+ SAEF-SGDM: `NCCL_IB_DISABLE=1 python main.py --dataset imagenet --data /export/Data/ILSVRC2012/ -p 5005 -a resnet50 -wd 1e-4 --epochs 90 -ds 30 60 -b 256 --method r-TopK --spar 0.1 -cb -sec`
