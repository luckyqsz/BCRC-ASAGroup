# siamFC

## 解决的问题

​	在追踪问题上，传统的方法不好，一般的深度学习方法受学习时数据限制。

​	传统的目标跟踪都是仅仅提取目标的外观特征实现跟踪，是一种在线学习，视频本身是唯一的训练数据来源，只能学习到很简单的模型。考虑引入深度学习，但是由于我们想要跟踪任意物体，事先是不知道待跟踪目标的，即没有很大的训练集，那么如果使用SGD必然也是在线的，就会大大限制了跟踪的时效性。

​    对于传统的目标跟踪算法，都是专门在线学习对象的外观的模型来解决的，视频本身是唯一的训练数据来源。学习到的东西比较少。在试图引入深度学习时，发现因为预先不知道目标的位置，没办法使用SGD。siamFC将siamese网络应用与目标跟踪领域（simaese network是一种用于衡量连个输入之间的相似度的网络），那么siamFC就是输入候选区与待跟踪目标，导入siamese network进行比对，相似度最高的即为目标。

​	换言之，siamFC做的就是一个匹配搜寻的工作：给出第一帧的目标，用网络去寻找后续帧中与目标匹配的对象。

## 实现方法

​	![siamFC](./image/siamFC.png)

​	如图，z表示第一帧的目标，x表示第二帧的图像，一般x尺度都比z大。

### 实现思路

​	 Siamese networks apply an identical transformation 'phi' to both inputs and then combine their representations using another function g according to f(z; x) = g(’(z); ’(x)). When the function g is a simple distance or similarity metric, the function 'phi' can be considered an embedding. *表示卷积，换句话说可以将phi（z）的feature map看做卷积核，在phi（x）上进行滑动，得到的feature map称为得分图，得分最高的为目标所在。将上一帧目标所在位置放在得分图中间，乘上得分最高区域的滑动步长，就可以得到当前目标的位置。$\phi$其实也是卷积函数。

### 训练方法（loss function）	

$$
l = log(1+exp(-yv))，
$$

​		where v is the real-valued score of a single exemplar-candidate pair and y $\in$ \{-1,1\}  is its ground-truth label.

那么y怎么得到呢，

<img src="./image/siam2.png" alt="siam2" style="zoom:60%;" />

即y是根据与中心之间的距离得到的，当距离不超过R时，为+1，否则为-1。因为两帧之间目标的移动很小，所以移动很大的肯定不是目标，同时应该注意到，±1的取值使得不同位置对于loss的贡献不同，即若要使loss最小，那么必须要使距中心很远的目标得分越低越好，距离中心很近的目标，得分越高越好。

上式中的\l只是单个候选图像的loss，用于SGD的是整副图像的loss，即将整副得分图的l求平均。最终求得$\phi$的卷积参数。

## 实现效果

## 一些思考

	1. 本文提出的siamFC之应用于单目标，能否应用到多目标，在siam中z充当了卷积核的作用，能否将同时输入多张目标，通过增加卷积核数量的方式实现多目标的跟踪。
