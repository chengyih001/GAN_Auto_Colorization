# Auto Colorization with cGANs
Author: Yih CHENG, Yao ZHANG

Project Detail and Summary : https://github.com/chengyih001/GAN_Auto_Colorization/blob/master/paper/Auto%20Colorization%20with%20cGANs.pdf

## Results

<p float="left">
  <img src="./images/cGAN_output.png"/>
  <img src="./images/cGAN_sketch_output.png"/>
</p>


## Overview
Image translation has been one of the most popular topics in computer vision, and approaches aiming to study the mapping function between input images and output images have been one of the main focuses toward this problem. In this project, the target is to colorize grayscale flower images with large, deep conditional GAN (cGAN) architecture. 

## Problem Description
Given an input grayscale image _`xi`_ `∈ [0, 255]` m\*n\*3 in domain X, the corresponding colorized image output is _`yi`_ `∈ [0, 255]` m\*n\*3 in domain Y, with m being the image height, n being the image width, and 3 representing the three `RGB` channels. The goal for this project is to implement a model that can learn the mapping function `GXY : X → Y` , which is capable of generating output images `GXY (X)` that are indistinguishable from the ground-truth images Y.

## Implementation

![](https://github.com/chengyih001/GAN_Auto_Colorization/blob/master/images/cGAN_structure.png?raw=true)

A cGAN consists of one generator and one discriminator. Working in `CIELAB` color space, the generator generates the `ab` channels of a `L` channel input image. The concatenation of `L` and `ab` channels is then the colorized version of the input image. The discriminator is used to check whether the generated colorful image is real (ground-truth) or fake (generated). To find a more efficient and high-quality method, two versions of the generator were implemented. The first one uses a U-net structure that includes 8 U-net blocks for the encoder and 8 U-net blocks for the decoder. Each encoder U-net block includes a convolutional layer followed by a batch normalization layer and an activation layer. Dropout layer is also implemented to reduce overfitting and improve generalization errors. For decoder U-net blocks, each includes a deconvolutional layer followed by a batch normalization layer and a 'ReLu' activation layer. In addition, the 'ReLu' activation layer for the last U-net block in the decoder is replaced by a 'Tanh' activation layer. In the second generator implementation, Dynamic U-Net from fastai library, a structured 'U-Net' with 'ResNet18' as backbone from torchvision library was utilized. For the discriminator, it is composed of three Conv-BatchNorm-LeakyReLU blocks. No normalization is used in the first block, while neither normalization nor activation function is implemented in the last block.

