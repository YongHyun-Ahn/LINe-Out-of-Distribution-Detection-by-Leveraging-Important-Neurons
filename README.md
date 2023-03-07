# LINe: Out-of-Distribution Detection by Leveraging Important Neurons

This is the source code for [LINe: Out-of-Distribution Detection by Leveraging Important Neurons]

## Usage

### 1. Dataset Preparation for Large-scale Experiment 

#### In-distribution dataset

Please download [ImageNet-1k](http://www.image-net.org/challenges/LSVRC/2012/index) and place the training data and validation data in
`./datasets/ILSVRC-2012/train` and  `./datasets/ILSVRC-2012/val`, respectively.

#### Out-of-distribution dataset

We have curated 4 OOD datasets from 
[iNaturalist](https://arxiv.org/pdf/1707.06642.pdf), 
[SUN](https://vision.princeton.edu/projects/2010/SUN/paper.pdf), 
[Places](http://places2.csail.mit.edu/PAMI_places.pdf), 
and [Textures](https://arxiv.org/pdf/1311.3618.pdf), 
and de-duplicated concepts overlapped with ImageNet-1k.

For iNaturalist, SUN, and Places, we have sampled 10,000 images from the selected concepts for each dataset,
which can be download via the following links:
```bash
wget http://pages.cs.wisc.edu/~huangrui/imagenet_ood_dataset/iNaturalist.tar.gz
wget http://pages.cs.wisc.edu/~huangrui/imagenet_ood_dataset/SUN.tar.gz
wget http://pages.cs.wisc.edu/~huangrui/imagenet_ood_dataset/Places.tar.gz
```

For Textures, we use the entire dataset, which can be downloaded from their
[original website](https://www.robots.ox.ac.uk/~vgg/data/dtd/).

Please put all downloaded OOD datasets into `./datasets/`.

#### Pre-trained model

Please download [Bit-S-R101x1](https://drive.google.com/file/d/1hrM3h6HL-BE6PtXJmmc1H6RWe7oakNZo/view?usp=sharing) and place in the `checkpoints` folder.

### 2. Dataset Preparation for CIFAR Experiment 

#### In-distribution dataset

The downloading process will start immediately upon running. 

#### Out-of-distribution dataset
We provide links and instructions to download each dataset:

* [SVHN](http://ufldl.stanford.edu/housenumbers/test_32x32.mat): download it and place it in the folder of `./datasets/ood_datasets/svhn`. Then run `python select_svhn_data.py` to generate test subset.
* [Textures](https://www.robots.ox.ac.uk/~vgg/data/dtd/download/dtd-r1.0.1.tar.gz): download it and place it in the folder of `./datasets/ood_datasets/dtd`.
* [Places365](http://data.csail.mit.edu/places/places365/test_256.tar): download it and place it in the folder of `./datasets/ood_datasets/places365/test_subset`. We randomly sample 10,000 images from the original test dataset. 
* [LSUN-C](https://www.dropbox.com/s/fhtsw1m3qxlwj6h/LSUN.tar.gz): download it and place it in the folder of `./datasets/ood_datasets/LSUN`.
* [LSUN-R](https://www.dropbox.com/s/moqh2wh8696c3yl/LSUN_resize.tar.gz): download it and place it in the folder of `./datasets/ood_datasets/LSUN_resize`.
* [iSUN](https://www.dropbox.com/s/ssz7qxfqae0cca5/iSUN.tar.gz): download it and place it in the folder of `./datasets/ood_datasets/iSUN`.

For example, run the following commands in the **root** directory to download **LSUN-C**:
```
cd datasets/ood_datasets
wget https://www.dropbox.com/s/fhtsw1m3qxlwj6h/LSUN.tar.gz
tar -xvzf LSUN.tar.gz
```
### 3. Pre-trained Model Preparation

For CIFAR, the model we used in the paper is already in the checkpoints folder. 

For ImageNet, the model we used in the paper is the pre-trained ResNet-50 provided by Pytorch. The download process
will start upon running.

## Preliminaries
It is tested under Ubuntu Linux 20.04 and Python 3.8 environment, and requries some packages to be installed:
* [PyTorch](https://pytorch.org/)
* [numpy](http://www.numpy.org/)

## Precompute
LINe need precomputing for calculate Shapley value approximation.

Run `./precompute.py`.

## Demo
### 1. Demo code for Large-scale Experiment 

Run `./demo-imagenet.sh`.

### 2. Demo code for CIFAR Experiment 

Run `./demo-cifar.sh`.

This codebase is from DICE (Sun et al. ECCV2022) 
https://github.com/deeplearning-wisc/dice
