# SORT

——《simple online and realtime tracking》

## 总结论文工作

​	这篇paper提出了tracking-by-detection的MOT框架，即：目标检测器-状态预测-数据关联-track管理。具体来说，目标由detection检测为bbox（因此该框架下，跟踪的效果很依赖检测的准确度），然后MOT问题就可以看做data association问题（所谓的data association就是将检测到的bbox对象与跟踪目标配对的过程）。本文解决的就是如何建立帧与帧之间目标的关联。

## 网络细节与一些思考

​	首先SORT网络分为四个部分，往后的多目标预测基本也是按照这个框架来。即detection——estimation model——data association——creation and deletion of track identities。

#### detection

​	没啥说的，直接上最NB的检测网络，作者使用的faster rcnn。paper指出，检测效果对跟踪的效果非常重要。

#### estimation model

​	this component used to 预测目标在下一帧的状态。作者用bbox的中心坐标，bbox的长宽这几个量来作为目标的状态。怎么预测呢？We approximate the inter-frame displacements of each object with a linear constant velocity model which is independent of other objects and camera motion.速度怎么解决呢，作者用的是卡尔曼滤波器。

​	为什么需要这个component呢？因为目标是运动（变化）的，我们需要预测出下一帧目标的位置（状态），并将此当做其真实的状态，用它对检测到的目标做关联。

#### data association

​	建立检测器检测到的目标与已经存在的目标之间的对应关系。注：这里已经存在的目标的状态是由预测网络预测得到的。如何建立关联呢？首先计算each detection与predicted bbox的IOU，然后用Hungarian（匈牙利算法）对其进行配对。

​	值得注意的是IOU这种衡量方式可以一定程度上解决遮挡问题，因为在本文中作者对于对象的状态使用的基本都是位置信息，一些遮挡只是遮挡对象内容，不会对目标的坐标造成太大的干扰。（这也是一个缺点？不管内容只管位置）

#### creation and deletion of track identities

​	新目标的产生与旧目标的消失是多目标跟踪相较单目标跟踪的一大难点。本文中作者认为 For creating trackers, we consider any detection with an overlap less than IOUmin to signify the existence of an untracked object。Tracks are terminated if they are not detected for TLost frames。作者将TLoss设置为1.

#### 两个重要的评价指标与影响因素分析

FP（lower is better）：number of false detections（错检）

FN（lower is better）：number of missed detections.（漏检）

##### 分析：

​	按照本文的方法，这二者主要是在IOU计算时产生问题。也就是目标状态的预测以及检测。那么什么时候会出现错检与漏检呢？

​	错检：与另外一个目标的IOU最大——>预测的位置误差太大？跳帧？

​	漏检：没检测到目标或者与所有的选框的IOU值都很小，判断为该目标丢失——>检测器问题比较大？