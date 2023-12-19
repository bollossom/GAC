# Gated Attention Coding for Training High-performance and Efficient Spiking Neural Networks ([AAAI24](https://arxiv.org/abs/2308.06582))

[Xuerui Qiu](https://scholar.google.com/citations?user=bMwW4e8AAAAJ&hl=zh-CN), [Rui-Jie Zhu](https://scholar.google.com/citations?user=08ITzJsAAAAJ&hl=zh-CN), [Yuhong Chou](),[Zhaorui Wang](), [Liang-Jian Deng](https://scholar.google.com/citations?user=TZs9NxkAAAAJ&hl=zh-CN), [Guoqi Li](https://scholar.google.com/citations?user=qCfE--MAAAAJ&)

Institute of Automation, Chinese Academy of Sciences
University of Electronic Science and Technology of China

:rocket:  :rocket:  :rocket: **News**:

- **Dec. 19, 2023**: Release the code for training and testing.
- **Dec. 17, 2023**: Accepted as poster in AAAI2024.

## Abstract
Spiking neural networks (SNNs) are emerging as an energy-efficient alternative to traditional artificial neural networks (ANNs) due to their unique spike-based event-driven nature. Coding is crucial in SNNs as it converts external input stimuli into spatio-temporal feature sequences. However, most existing deep SNNs rely on direct coding that generates powerless spike representation and lacks the temporal dynamics inherent in human vision. Hence, we introduce Gated Attention Coding (GAC), a plug-and-play module that leverages the multi-dimensional gated attention unit to efficiently encode inputs into powerful representations before feeding them into the SNN architecture. GAC functions as a preprocessing layer that does not disrupt the spike-driven nature of the SNN, making it amenable to efficient neuromorphic hardware implementation with minimal modifications. Through an observer model theoretical analysis, we demonstrate GAC's attention mechanism improves temporal dynamics and coding efficiency. Experiments on CIFAR10/100 and ImageNet datasets demonstrate that GAC achieves state-of-the-art accuracy with remarkable efficiency. Notably, we improve top-1 accuracy by 3.10\% on CIFAR100 with only 6-time steps and 1.07\% on ImageNet while reducing energy usage to 66.9\% of the previous works. To our best knowledge, it is the first time to explore the attention-based dynamic coding scheme in deep SNNs, with exceptional effectiveness and efficiency on large-scale datasets.

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/gated-attention-coding-for-training-high/image-classification-on-cifar-10)](https://paperswithcode.com/sota/image-classification-on-cifar-10?p=gated-attention-coding-for-training-high)


[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/gated-attention-coding-for-training-high/image-classification-on-cifar-100)](https://paperswithcode.com/sota/image-classification-on-cifar-100?p=gated-attention-coding-for-training-high)


[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/gated-attention-coding-for-training-high/image-classification-on-imagenet)](https://paperswithcode.com/sota/image-classification-on-imagenet?p=gated-attention-coding-for-training-high)
## Prerequisites
The Following Setup is tested and it is working:
 * Python 3.7
 * Pytorch 1.8.0
 * Cuda 10.2

## Description
 * use a triangle-like surrogate gradient `ZIF` in `models/layer.py` for step function forward and backward.

 * The 0-th and 1-th dimension of snn layer's input and output are batch-dimension and time-dimension. 

 * The most straightforward way of training higher quality models is by increasing their size. In this work, we would like to see that deepening network structures could get rid of the degradation problem and always be a trustworthy way to achieve satisfying accuracy for the direct training of SNNs.

 * This repository contains the source code for the training of our MS-ResNet on ImageNet. The models are defined in `models/MS_ResNet.py` .

1. Change the data paths `vardir,traindir` to the image folders of ImageNet/CIFAR dataset.
2. For CIFAR dataset, to train the model, please run  `run.sh`.
3. For ImageNet dataset, to train the model, please run  `run.sh` or `CUDA_VISIBLE_DEVICES=GPU_IDs python -m torch.distributed.launch --master_port=1234 --nproc_per_node=NUM_GPU_USED train_amp.py -net resnet34 -b 256 -lr 0.1` .
`-net` option supports `resnet18/34` .

## Weight file
* Due to the size limit of the uploaded file, we currently open source the experimental weight of T=6 on the CIFAR100 dataset CIFAR100_T=6.pth.
* The test code can be run according to the following requirements
1. Change the data paths `vardir,traindir`  the image folders of CIFAR100 dataset.
2. Then run `python test.py` in CIFAR file.
* Other weight files will be open source soon.
## link
Our experimental weight of T=6 on the CIFAR100 dataset  can be found in 
~~~
link：https://pan.baidu.com/s/1jue9S9hAKFeYCs2iGWXUxg 
code：4567
~~~

## Data Prepare

- use `PyTorch` to load the CIFAR10 and CIFAR100 dataset.
Tree in `./data/`.

```shell
.
├── cifar-100-python
├── cifar-10-batches-py

```

ImageNet with the following folder structure, you can extract imagenet by this [script](https://gist.github.com/BIGBALLON/8a71d225eff18d88e469e6ea9b39cef4).

```shell
│imagenet/
├──train/
│  ├── n01440764
│  │   ├── n01440764_10026.JPEG
│  │   ├── n01440764_10027.JPEG
│  │   ├── ......
│  ├── ......
├──val/
│  ├── n01440764
│  │   ├── ILSVRC2012_val_00000293.JPEG
│  │   ├── ILSVRC2012_val_00002138.JPEG
│  │   ├── ......
│  ├── ......
```

## Contact Information

```
@article{qiu2023gated,
  title={Gated Attention Coding for Training High-performance and Efficient Spiking Neural Networks},
  author={Qiu, Xuerui and Zhu, Rui-Jie and Chou, Yuhong and Wang, Zhaorui and Deng, Liang-jian and Li, Guoqi},
  journal={arXiv preprint arXiv:2308.06582},
  year={2023}
}
```

For help or issues using this git, please submit a GitHub issue.

For other communications related to this git, please contact `qiuxuerui2024@ia.ac.cn`.
