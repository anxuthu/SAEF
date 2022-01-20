# SAEF
The synthetic codes for paper ["Step-Ahead Error Feedback for Distributed Training with Compressed Gradient"](https://ojs.aaai.org/index.php/AAAI/article/view/17254). [[arxiv](https://arxiv.org/abs/2008.05823)]

## Citation
```
@inproceedings{xu2021step,
  title={Step-Ahead Error Feedback for Distributed Training with Compressed Gradient},
  author={Xu, An and Huo, Zhouyuan and Huang, Heng},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={35},
  number={12},
  pages={10478--10486},
  year={2021}
}
```

## Tested Environment

Python 3 and PyTorch 1.4

## Experiments

### CIFAR-100

+ SGDM: `python main.py --a resnet56 --dataset cifar100 --method va`
+ EF-SGDM: `python main.py --a resnet56 --dataset cifar100 --method ef-Sign -cb -sec`
+ SAEF-SGDM: `python main.py --a resnet56 --dataset cifar100 --method r-Sign -cb -sec`

### IMAGENET

+ SGDM: `NCCL_IB_DISABLE=1 python main.py --dataset imagenet --data /export/Data/ILSVRC2012/ -p 5005 -a resnet50 -wd 1e-4 --epochs 90 -ds 30 60 -b 256 --method va --spar 0.1 -cb -sec`
+ EF-SGDM: `NCCL_IB_DISABLE=1 python main.py --dataset imagenet --data /export/Data/ILSVRC2012/ -p 5005 -a resnet50 -wd 1e-4 --epochs 90 -ds 30 60 -b 256 --method ef-TopK --spar 0.1 -cb -sec`
+ SAEF-SGDM: `NCCL_IB_DISABLE=1 python main.py --dataset imagenet --data /export/Data/ILSVRC2012/ -p 5005 -a resnet50 -wd 1e-4 --epochs 90 -ds 30 60 -b 256 --method r-TopK --spar 0.05 -ap 20 -cb -sec`
