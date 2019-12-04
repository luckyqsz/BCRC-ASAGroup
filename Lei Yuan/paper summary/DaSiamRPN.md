## Distractor-aware Siamese Networks for Visual Object Tracking

#### 													—— ECCV 2018, CASIA Zheng Zhu, Bo Li, Qiang Wang

1. ### work summary

   ​	本文分析了siamRPN的3个缺陷，并提出相应的改进方法。

   ​	**siamRPN缺陷分析：**

   - **在训练阶段，存在样本不均衡问题。即大部分样本都是没有语义的背景（注：背景不是非target的物体，而是指那些既不属于target，也不属于干扰物，没有语义的图像块，例如大片白色。）这就导致了网络学到的仅仅是对于背景与前景的区分，即在图像中提取出物体，但我们希望网络还能够识别target与非target物体。作者从数据增强的角度解决此问题。**
   - **模型判别性不足，缺乏模板的更新，即在heatmap中很多非target的物体得分也很高；甚至当图像中没有目标时，依然会有很高的得分存在。作者认为与target相似的物体对于增强模型的判别性有帮助，因此提出了distractor-aware module对干扰（得分很高的框）进行学习，更新模板，从而提高siam判别性。**
   - **由于第二个缺陷，原始的siam以及siamRPN使用了余弦窗，因此当目标丢失，然后从另一个位置出现，此时siam不能重新识别target, (siamfc/ siamRPN的搜索是否是全局搜索？)，该缺陷导致siam无法适应长时间跟踪的问题。对此作者提出了local-to-global的搜索策略，其实就是随着目标消失帧数的增加，搜索区域逐渐扩大。**

2. ### 数据增强

   ​	作者通过数据增强解决训练时样本不均衡问题。

   - Diverse categories of positive pairs can promote the generalization ability. 

     ​	作者发现siamRPN对于未训练过的类别的目标，跟踪的效果会变差。因此增加训练类别对结果还有改善。其实就是增大训练集类别，除了直接增加训练集外，可以人为的增强，例如将target抠出来，换个背景，以此来增加正样本的数量。

   - Semantic negative pairs can improve the discriminative ability.

     ​	增加来自不同类别的负样本可以抑制网络随意匹配前景的问题，例如在目标被遮挡以及消失时预测点乱飘。

     ​	增加来自同一类别的负样本，可以增强网络的对target的细粒度区分。

   - Customizing effective data augmentation for visual tracking.

     ​	在常规的数据增强方法外，作者发现增加图像的运动模糊也有效。

   ![](../iamge/dasiam2.png)

3. ### Distractor-aware

   ​	作者对heatmap得到目标先NMS去除一些冗余框，然后将相似度靠前的框（干扰物）拿出来，让网络对其学习，拉大target_embedding与这些干扰物之间的相似度。**实质上是对exemplar的更新修正。注意此处可以看到exemplar采用的不是第一帧，而是上一帧的目标，这就会带来模板污染问题，但是DaSiam却可以胜任长时间跟踪问题，可见该跟新策略的抗污染性不错。可以分析，污染问题的结果就是target及其相似物体的得分随着迭代次数的增加，得分排名变化，但是该更新策略不是暴力的直接替换，而是综合了排名靠前的框，因此可以有效的抑制模板污染问题。从后面的实验结果也可以看到加入了更新策略的DaSiam比不更新的SiamRPN无论是在A还是R上都要高。证明该更新策略有效。**

   

   ![](../iamge/dasiam1.png)

4. ### DaSiamRPN for Long-term Tracking

   ​	siamRPN的搜索区域好像只是上一帧目标的附近，因此当目标跟踪失败后，当目标重新出现时无法再次有效识别目标，因为目标很可能已经离开了搜索区域。

   ​	所以DaSiamRPN的loacl-to-global策略就是当目标跟踪失败后，搜索区域不断增大。

5. ### experiments

   ![](../iamgep/dasiam3.png)

   ​		根据消融实验，数据增强，distractor-aware以及数据增强带来的EAO收益均在0.02左右。

   ![](../iamge/Dasiam4.png)
