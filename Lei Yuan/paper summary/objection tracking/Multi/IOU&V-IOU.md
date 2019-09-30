# IOU tracker

​	——《High-Speed Tracking-by-Detection Without Using Image Information》

## 总结论文工作

​	作者认为目标检测技术现在性能比较好，不需要任何复杂的跟踪网络，提出了一个完全依赖检测的结果的跟踪方法。具体怎么做呢，检测会得到一个个目标的选框，计算当前帧与上一帧的哪个目标IOU更高，认为他们在一个track内。

![](https://github.com/luckyqsz/BCRC-ASAGroup/blob/master/Lei%20Yuan/image/iou1.png)

​		因为这种方法完全依赖于检测，所以对检测有两个要求：1. 每一帧中的目标都要能检测出来，即中间不会断帧，目标是连续的。2. 对于同一个目标，在连续的帧中要有很高的IOU。（也就是说这个检测器要有很高的recalling，precision；且目标的移动速度，变化要很低）。

![](https://github.com/luckyqsz/BCRC-ASAGroup/blob/master/Lei%20Yuan/image/iou2.png)	

​	首先设定检测阈值1, 对于所有检测结果，只保留检测分大于阈值1的，然后对于每一个检测对象，计算它与上一帧对象的IOU，判断最高iou是否大于阈值2，如果大于，则将该目标归于该track；否则判断上一个目标与上上一个匹配对象的iou是否大于阈值3，如果大于，且该track长度也大于track的最小长度，则认为该track跟踪的目标已经移出镜头，该track结束。对于那些匹配不成功的则认为他们是新目标，新开一条track进行跟踪。

## 评价

	1. 显然该方法完全依赖与检测，然而检测不是理想的，检测器无论是检错或者漏检，都要新开一条路径。匹配完全依赖于目标的位置信息，与SORT有一样的问题，跟踪时很容易跟错，IDs很高，跟踪的准确度很低，精度还行。
 	2. 但是该网络没有用到任何复杂的跟踪网络，所以速度非常之快。作者说可以达到10k fps。另外因为没有estimate与association环节，所以FP也不高。



# V-IOU Tracker

​	——《Extending IOU Based Multi-Object Tracking by Visual Information》

## 论文工作总结

​	这篇论文是IOU tracker 的升级版，仿照deepsort，在进行IOU匹配时加入图像内容信息，使得IDs下降。

IDs为什么很高的分析，见iou的评价部分，主要是漏检FN造成的。那么怎么做呢？当目标找不到与上一帧的匹配时，启用一个visual tracking extension（一个单目标跟踪器），从图像内容上进行跟踪。实际上可以把visual tracking extension当成一个连接器，对track的断点进行连接。具体来说分两点：1. 向后连接，目标与上一帧匹配不上时，在上一帧的同样位置启动visual tracking extension，连续使用t帧，在这t帧中如果目标满足iou中的匹配阈值，则visual tracking extension，连接成功，继续iou；如果直到t帧结束都没有满足匹配阈值，那么认为这条track是真正结束了。2. 向前连接，当一个目标需要开辟一条新track时，先对此目标向前跟踪t帧，如果满足iou匹配阈值，则合并track。

​	一句话概括就是通过增加一单目标跟踪器合并track。

## 性能的改进

![](https://github.com/luckyqsz/BCRC-ASAGroup/blob/master/Lei%20Yuan/image/iou3.png)

​	可以看到v-iou相比与iou，IDs确实大幅度下降，不过FP轻微上升，这是合并过度导致的，速度也降低了两个数量级。

## 评价

1. 综合deepsort与v-iou的行文原因来看，只进行位置信息的匹配是不行的，IDs太高，因此在跟踪时必须加入内容信息。同时还有一点值得注意的是，内容要比位置容易匹配得多，即加入内容可以降低匹配的阈值。
2. 对比deepsort与v-iou的解决方式，deepsort是将位置与内容信息综合到一起，速度损失不算大，而v-iou则是完全开启一个单独的单目标跟踪网络，导致速度降了两个数量级，因此网络嵌套慎用。FN主要是由检测器漏检造成的，换个recalling更高的检测器是不是会有很大的提升。
