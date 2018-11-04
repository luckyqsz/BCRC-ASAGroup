# Course Notes

## Generative Adversarial Network (GAN), 2018 by Hung-yi Lee

Yingyan Shi

shiyingyan12@qq.com

Brain Chip Research Center, Fudan University

***

[TOC]

### Video link

https://www.bilibili.com/video/av24011528?from=search&seid=2423494053064598440

https://www.youtube.com/playlist?list=PLJV_el3uVTsMq6JEFPW35BCiOQTsoqwNw

### Code

Zoo for GAN and its derivations, implemented by PyTorch

https://github.com/corenel/GAN-Zoo

### Derivation

https://mp.weixin.qq.com/mp/profile_ext?action=home&__biz=MzA5NDI4MTcwMg==&scene=124#wechat_redirect

------

### 1. Introduction

#### 1.1 Basic Idea of GAN

同时训练两个模型

1. 一个能捕获数据分布的生成模型 G 

2. 一个能估计数据来源于真实样本概率的判别模型 D

Minimax Game 极大极小博弈

1. 对于 D 而言要尽量使公式最大化（识别能力强）
2. 对于 G 又想使之最小（生成的数据接近实际数据）



##### 1.1.1 Illustration

![](images/1.png)

![](images/2.png)

##### 1.1.2 Formulation

![](images/3.png)

#### 1.2 GAN as Structured Learning

##### 1.2.1 Structured Learning

Structured Learning/Prediction: output a sequence, a matrix, a graph, a tree ......

Output is composed of components with dependency and they should be considered globally.

The relation between the components are critical.

##### 1.2.2 One-Shot/Zero-Shot Learning

![](images/4.png)

##### 1.2.3 Structured Learning Method

![](images/5.png)

It is easier to catch the relation between the components by top-down evaluation.

The CNN filter is good enough.

Generators imitate the appearance, and hard to learn the correlation between components.

Discriminators consider the big picture, but generation is not always feasible, especially when your model is deep. 

How to do negative sampling?  ------ Using Generators to get negative examples which is efficient.

![](images/8.png)

#### 1.3 Can Generator Learn By Itself?

Encoder in auto-encoder provides the code.

##### 1.3.1 Auto-encoder

![](images/6.png)

Variationaly Auto-encoder performs better.

#### 1.4 Can Discriminator Generate?

Discriminator training needs some negative examples.

Generate negative examples by discriminator D. 

![](images/7.png)

#### 1.5 A Little Bit Theory

![](images/9.png)

![](images/10.png)

![](images/11.png)

![](images/12.png)

![](images/13.png)

### 2. Conditional Generation

Generator will learn to generate realistic images but completely ignore the input conditions.

Discriminator will judge: 

1. x(object) is realistic or not +
2. c(condition) and x are matched or not

#### 2.1 cGAN Framework

Conditional GAN Framework

![](images/14.png)

#### 2.2 Stack GAN

To get better performance, stack GANs.

![](images/15.png)

#### 2.3 Image to Image

![](images/16.png)

![](images/17.png)

![](images/18.png)

#### 2.4 Speech Enhancement

![](images/19.png)

#### 2.5 Video Generation

![](images/20.png)

###  3. Unsupervised Conditional Generation

Transform an object from one domain to another without paired data(e.g. style transfer).

![](images/21.png)

Approach 1 only deploys slightly changes.

Approach 2 is suitable for large change, only keeping the semantics.

#### 3.1 Direct Transformation

To avoid that Generator will output images with no relevance to images fed into, two methods:

![](images/22.png)

![](images/23.png)

##### 3.1.1 CycleGAN

![](images/24.png)

##### 3.1.2 StarGAN

![](images/25.png)

![](images/26.png)

#### 3.2 Projection to Common Space

![](images/27.png)

![](images/28.png)

![](images/29.png)

![](images/30.png)

![](images/31.png)

![](images/32.png)

![](images/33.png)

Application: Voice conversion

### 4. Basic Theory

#### 4.1 Concept

##### 4.1.1 Divergence

In statistics and information geometry, **divergence** or a **contrast function** is a function which establishes the "distance" of one probability distribution to the other on a statistical manifold. The divergence is a weaker notion than that of the distances, in particular the divergence need not to be symmetric (that is, in general the divergence from p to q is not equal to the divergence from q to p), and need not satisfy the triangle inequality.

### 5. General Framework



### 6. WGAN, EBGAN



### 7. InfoGAN, VAE-GAN, BiGAN

Modifying a specific dimension, no clear meaning.

#### 7.1 InfoGAN

If Discriminator is removed, Generator will contribute to Classifier's performance by directly copy the input c to x. So Discriminator is necessary.

z is composed of two factors: c (code) and z' (random numbers, which is not known to us).

We expect that every dimension of c can manipulate the output images clearly and directly.

![](images/36.png)

#### 7.2 VAE-GAN

![](images/35.png)

![](images/37.png)

#### 7.3 BiGAN

![](images/38.png)

![](images/39.png)

#### 7.4 Domain-Adversarial Training





### 8. Photo Editing

Each dimension of input vector represents some characteristics.

Understand the meaning of each dimension to control the output.

Target: Connecting code and attribute.

Code means attribute representation.

![](images/40.png)

Image Super Resolution

Image Completion

### 9. Sequence Generation



### 10. Evaluation

Estimate the distribution of P~G~(x) from sampling.

We don't want memory GAN which means this GAN only generates images of a specific category.

#### 10.1 Likelihood v.s. Quality

![](images/41.png)

![](images/42.png)