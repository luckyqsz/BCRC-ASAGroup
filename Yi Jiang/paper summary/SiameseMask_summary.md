## paper summary：

论文题目：Fast Online Object Tracking and Segmentation: A Unifying Approach

论文地址：https://arxiv.org/abs/1812.05050 

### 该论文拟解决的问题

- 视频目标分割（VOS）的速度慢

- 目标跟踪的精度不高（如：SiamFC、SiamRPN）

本论文引入Mask分支，bounding box采用了旋转框，网络结构采用 Siamese Network提提高了速度。

### 方法

采用的Siamese网络，继承了了Siamese网络实时性好的优点，引入了Mask的分支和旋转框，提高了bounding box的精度。

**SiamFC Network：**

![5-1](F:\新建文件夹\jiangyi\桌面\GitHub&论文\论文\8SiameseMask\5-1.jpg)

![8-3](F:\新建文件夹\jiangyi\桌面\GitHub&论文\论文\8SiameseMask\8-3.png)

**SiamRPN：**

![6-1](F:\新建文件夹\jiangyi\桌面\GitHub&论文\论文\8SiameseMask\6-1.png)

引入了RPN网络，有两个支路，一个是分类，一个是bounding box回归

**SiamMask Network：**

![8-1](F:\新建文件夹\jiangyi\桌面\GitHub&论文\论文\8SiameseMask\8-1.png)

![8-2](F:\新建文件夹\jiangyi\桌面\GitHub&论文\论文\8SiameseMask\8-2.png)

**Masks representation：**

![8-4](F:\新建文件夹\jiangyi\桌面\GitHub&论文\论文\8SiameseMask\8-4.png)

通过两层神经网络hϕ 和学习参数ϕ来预测w×h的二元masks(每个RoW都有一个)。用mn表示预测的mask，对应着第第n个RoW

**Mask 分支的Loss Function**

![8-5](F:\新建文件夹\jiangyi\桌面\GitHub&论文\论文\8SiameseMask\8-5.png)

i，j表示像素点

yn为{-1，1}，IoU大于0.6视为正样本，即值为1，loss function即只考虑正样本情况

Cn为{-1，1}，表示在第n个RoW的像素点相对应的标签值

**整个网络的Loss Function(2个分支VS3个分支)**

![8-6](F:\新建文件夹\jiangyi\桌面\GitHub&论文\论文\8SiameseMask\8-6.png)

λ1，λ2，λ3为超参数，训练时设置λ1=32，λ2=1，λ3=1

#### **实验**

**实验效果**

![8-7](F:\新建文件夹\jiangyi\桌面\GitHub&论文\论文\8SiameseMask\8-7.png)

作者尝试了三种不同的方法。 Min-max（红色）：包含对象的与轴对齐的矩形； MBR（绿色，作者的方法）：最小边界矩形； opt（蓝色）：通过VOT-2016中提出的优化策略获得的矩形。

**实验结果**

- 在VOT-2018上多种模型的对比

![8-8](F:\新建文件夹\jiangyi\桌面\GitHub&论文\论文\8SiameseMask\8-8.png)

  在VOT-2016和VOT-2018上，三种不同方法的对比

![8-9](F:\新建文件夹\jiangyi\桌面\GitHub&论文\论文\8SiameseMask\8-9.png)

### 思考

对于实时性这个问题，两个分支和三个分支的速度分别为：平均每秒55帧和60帧。按理说三个分支的参数以及计算量会比两个分支的大，这边也没有细讲，这边的网络是否有所区别。