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

​	值得注意的是IOU这种衡量方式可以一定程度上解决遮挡问题，因为在本文中作者对于对象的状态使用的基本都是位置信息，一些遮挡只是遮挡对象内容，不会对目标的坐标造成太大的干扰。
（这也是一个缺点？不管内容只管位置也导致跟踪结果的IDsw很高，因为一旦estimate不准那么就可能匹配到其他位置的ID）

#### creation and deletion of track identities

​	新目标的产生与旧目标的消失是多目标跟踪相较单目标跟踪的一大难点。本文中作者认为 For creating trackers, we consider any detection with an overlap less than IOUmin to signify the existence of an untracked object。Tracks are terminated if they are not detected for TLost frames。作者将TLoss设置为1.

#### 两个重要的评价指标与影响因素分析

FP（lower is better）：number of false detections（错检）

FN（lower is better）：number of missed detections.（漏检）

##### 分析：

​	按照本文的方法，这二者主要是在IOU计算时产生问题。也就是目标状态的预测以及检测。那么什么时候会出现错检与漏检呢？

​	错检：与另外一个目标的IOU最大——>预测的位置误差太大？跳帧？

​	漏检：没检测到目标或者与所有的选框的IOU值都很小，判断为该目标丢失——>检测器问题比较大？



# deepsort 

——《simple online and realtime tracking with a deep association metric》

## 总结论文工作

​	SORT方法只利用了图像的选框位置信息，IDs很高，为了提高sort的性能，estimate环节在sort的基础上融合了外观信息，即原来导入匈牙利算法的只是IOU(位置信息)得分，现在导入的是结合了位置信息与图像内容信息的得分。那么怎么结合呢？首先对于位置信息，作者对sort中的状态信息施以马氏距离，记为d_1，对于位置信息，先用神经网络提取图像特征，然后采用余弦距离衡量二者在内容上的相似程度，记为d_2（余弦距离就是两个向量的cos值，相比与欧式距离，余弦距离更加注重向量方向上的差异）。到此我们就得到了位置与图像内容上的两种距离，然后将将二者结合起来得到匹配时的loss值c：
$$
c_i,_j = \lambda d^1_i,_j + (1-\lambda)d^2_i,_j
$$
同时分别对马氏距离与余弦距离设置两个阈值，当他们小于各自的阈值时记1分，否则记0分，然后把得分相加，得到目标对于某track的得分b_i,j。

为什么要计算c与b呢？相当于一个成功的匹配要满足两个条件：首先其得分要大于0，其次匹配的是loss最小的track。即在小于阈值的前提下尽可能匹配最接近的。

| MOT16 Test Results |          |      |      |       |       |      |      |        |        |         |
| ------------------ | -------- | ---- | ---- | ----- | ----- | ---- | ---- | ------ | ------ | ------- |
| detection          | tracker  | MOTA | MOTP | FP    | FN    | IDs  | FM   | MT     | ML     | runtime |
| faster R-CNN       | IOU      | 45.4 | 77.5 | 7639  | 89535 | 2284 | 2310 | 113    | 265    |         |
|                    | sort     | 59.8 | 79.6 | 8698  | 63245 | 1423 | 1835 | 25.40% | 22.70% | 60Hz    |
|                    | deepsort | 61.4 | 79.1 | 12852 | 56668 | 781  | 2008 | 32.80% | 18.20% | 40Hz    |

## 一些思考

​	由上面的对比可以看到，加入图像内容的信息后IDs确实大幅度下降，漏检数量也有所下降，不过错检也随之上升，相当于加入了内容信息后更加容易匹配，但也越容易收到干扰。
