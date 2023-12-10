# Gated Attention Coding for Training High-performance and Efficient Spiking Neural Networks
 Code for Gated Attention Coding

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
