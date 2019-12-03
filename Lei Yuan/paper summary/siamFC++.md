## SiamFC++: Towards Robust and Accurate Visual Tracking
with Target Estimation Guidelines

####                                                             —— AAAI 2020, Zhejiang university, Gang Yu

## 1. Target

​		针对siam网络分析了之前的工作不合理的地方，提出了4条guidelines，并就这4条guidelines对siamfc进行了改进。

#### 	1.1 guidelines：

- 跟踪网络分为两个子任务，一个是分类，一个是位置的准确估计。即网络需要有分类与位置估计两个分支。缺少分类分支，模型的判别性会降低，表现到VOT评价指标上就是R不高；缺少位置估计分支，目标的位置准确度会降低，表现到VOT评价指标上就是A不高。
- 分类与位置估计使用的feature map要分开。即不能分类的feature map上直接得到位置的估计，否则会降低A。
- siam匹配的要是原始的exemplar，不能是与预设定的anchors匹配，否则模型的判别性会降低，siamFC++的A值略低于siamRPN++，但是R值在测试过的数据集上都比siamRPN++高，作者认为就是anchors的原因，在论文的实验部分，作者进行了实验发现siamRPN系列都是与anchors进行匹配而不是exemplar本身，但是anchors与exemplar之间存在一些差异，导致siamRPN的鲁棒性不高。
- 不能加入数据分布的先验知识，例如原始siamFC的三种尺度变换，anchors等实际上都是对目标尺度的一种先验，否则会印象模型的通用性。

2. ## method

   #### 2.1 network

   ![](/home/lei/Desktop/siamfc++1.png)

   ​		quality assessment是啥？？？

   ​		值得注意的是在分类时，作者认为落在bbox内部的点都算positive，在计算loss时，只考虑positive的点，在对位置回归时，作者实验发现PSS loss比IOU loss高0.2个点，所以位置回归分支使用PSS loss，分类分支使用focal loss，quality分支使用BCE。如果考虑背景会不会对模型的判别性有进一步的提高。

   ![](/home/lei/Desktop/siamfc++2.png)

3. ## Experiments

   #### 3.1 Ablation study

   ![](/home/lei/Desktop/siamfc++3.png)

   ​	由该图可以看到siamFC++改进的各部分对于最后结果的提升（与原始的siamFC比较），位置回归对EAO提升最大，the regression branch (0.094), data source diversity (0.063/0.010), stronger backbone (0.026), and better head structure (0.020).

   #### 3.2 model result

   ![](/home/lei/Desktop/siamfc++4.png)

   #### 3.3 siamRPN系列anchors问题的实验分析

   ![](/home/lei/Desktop/siamfc++5.png)

4. ### 改进方向

   ​	在训练时，只使用了positive的点，该模型对背景的区分度是可能不够，如果把背景也加上会不会进一步提升模型的判别性。跟踪不仅仅是单纯的根据exemplar图像本身特征寻找，目标周围的环境对跟踪也有帮助，对loss进行改进。