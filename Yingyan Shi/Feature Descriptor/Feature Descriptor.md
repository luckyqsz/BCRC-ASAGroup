Yingyan Shi

shiyingyan12@qq.com

October 1st, 2018

Brain-Chip Research Center, Fudan University

---

[TOC]

# Traditional Feature Descriptor

## What is a Feature Descriptor?

A feature descriptor is a representation of an image or an image patch that simplifies the image by extracting useful information and throwing away extraneous information. Typically, a feature descriptor converts an image of size width x height x 3 (channels ) to a feature vector / array of length n. 

## Histogram of Oriented Gradients

 ### Definition

Locally normalized histogram of gradient orientation in dense overlapped grids.

### Essence

Histogram of Oriented Gradients provide a dense overlapping description of image regions.

The idea behind the HOG descriptors is that the object appearance and the **shape** within an image can be described by the histogram of **edge** directions.

## Scale Invariant Feature Transform





## Speeded-Up Robust Features



## Limitations of Traditional Hand-Engineered Features

Feature engineering is difficult, time-consuming, and requires expert knowledge on the problem domain. Traditional hand-engineered features are too sparse in terms of information that they are able to capture from an image. This is because the first-order image derivatives are not sufficient features for the purpose of most computer vision tasks such as image classification and object detection.

# Two extremes of Visual Learning

## Extrapolation problem

Generalization

Diagnostic features

## Interpolation problem

Correspondence

Finding the differences

# The Evolution of Vision Databases

Object-centered datasets

Scene-centered datasets

