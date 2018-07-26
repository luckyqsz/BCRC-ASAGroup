
# 20180720-20180726
## 周待做列表
* 将clock代码封装成一个class，给冯新宇提供接口；
* 改步态识别的代码，搭建系统；
* 看自然语言处理的一些东西，了解什么叫词向量等。
## clock class封装
>参考：https://blog.csdn.net/dfdfdsfdfdfdf/article/details/52439651
>>vector作为参数的三种传参方式：https://www.cnblogs.com/xiaoxi666/p/6843211.html
## 改步态识别的代码
>要拍摄视频<br>
买三脚架<br>
得到步宽信息<br>
尝试正面视频的处理<br>

### 面临问题
* 步态识别所使用模型openpose不够精确，侧面识别和正面识别的精确度都不高
  + 可能因为视频的像素影响
  + 可能因为视频背景对轮廓识别的影响
* 深度摄像头得到的深度视频没有灰度-深度之间的对应关系
* 深度视频的精确度不高（像素点抖动等）
* 搭建系统的空间
### 需要做的工作
* 首先要忽略识别的精确度去做深度视频的深度提取、深度精确度考察
>对步宽的定义：人物在视频中部，且人物迈开步子时，两脚踝深度之差<br>
1.导出全视频过程中两脚踝深度信息<br>
由于没有灰度和深度之间的函数对应关系，所以暂时先导出两脚踝的灰度值进行对比。
但是经过对比，两脚踝的灰度并没有很大差别，所以利用深度摄像头侧面信息获得步宽信息的可行性有待考量。
![脚踝深度之差理论](work_record/gait_depth_theory.png)
![脚踝深度之差实际](work_record/gait_depth.png)
可能原因:<br>
a) 摄像头精度问题（实验排除）<br>
b) gray to bgr和bgr to gray的可逆性问题<br>
c) 16UINT-8UINT的精度损失<br>
d）视频中人走路太快<br>
摄像头存疑问题:<br>
a) 黑色对识别有影响<br>

* 更多更稳定的视频的拍摄
* 尝试正面识别
* 精度评估
### 深度视频的应用

## 自然语言处理
### 基于的想法
传统：文字->向量
创新：拼音->向量
### 词嵌入、词向量（word embedding、word vector）
参考：[词嵌入](https://blog.csdn.net/ch1209498273/article/details/78323478)<br>
词嵌入是将词汇映射到实数向量的方法总称。将文本X{x1,x2,x3,x4,x5……xn}映射到多维向量空间Y{y1,y2,y3,y4,y5……yn }，这个映射的过程就叫做词嵌入。  
#### 词嵌入的三种方法：
>**1.Embedding Layer**<br>
Embedding Layer是与特定自然语言处理上的神经网络模型联合学习的单词嵌入。该嵌入方法将清理好的文本中的单词进行one hot编码（热编码），向量空间的大小或维度被指定为模型的一部分，例如50、100或300维。向量以小的随机数进行初始化。Embedding Layer用于神经网络的前端，并采用反向传播算法进行监督。<br>
>**2.Word2Vec/Doc2Vec**<br>
其核心思想就是基于上下文，先用向量代表各个词，然后通过一个预测目标函数学习这些向量的参数。Word2Vec 的网络主体是一种单隐层前馈神经网络，网络的输入和输出均为词向量。该算法给出了两种训练模型，CBOW (Continuous Bag-of-Words Model) 和 Skip-gram (Continuous Skip-gram Model)。<br>
>* CBOW将一个词所在的上下文中的词作为输入，而那个词本身作为输出，也就是说，看到一个上下文，希望大概能猜出这个词和它的意思。<br>
>* Skip-gram它的做法是，将一个词所在的上下文中的词作为输出，而那个词本身作为输入，也就是说，给出一个词，希望预测可能出现的上下文的词。
>
>Word2Vec只是简单地将一个单词转换为一个向量，而Doc2Vec不仅可以做到这一点，还可以将一个句子或是一个段落中的所有单词汇成一个向量。<br>
>**3.GloVe（Global Vectors for Word Representation）**<br>
GloVe是Pennington等人开发的用于有效学习词向量的算法，结合了LSA矩阵分解技术的全局统计与word2vec中的基于局部语境学习。<br>
LSA全称Latent semantic analysis，中文意思是隐含语义分析。<br>
#### 神经网络语言模型
>**1.Neural Network Language Model ，NNLM**<br>
>**2.Log-Bilinear Language Model， LBL**<br>
>**3.Recurrent Neural Network based Language Model，RNNLM**<br>
>**4.Collobert 和 Weston 在2008 年提出的 C&W 模型**<br>
>**5.Mikolov 等人提出了 CBOW（ Continuous Bagof-Words）和 Skip-gram 模型**<br>
#### 限制
单词嵌入主要限制之一是单词的可能含义被混合成单个表示（语义空间中的单个向量）。Sense embeddings 是这个问题的解决方案：单词的个体含义在空间中表示为不同的向量。
### 语音识别
参考：[语音识别](https://baike.baidu.com/item/%E8%AF%AD%E9%9F%B3%E8%AF%86%E5%88%AB/10927133?fr=aladdin)
#### 数据库
>MIT Media lab Speech Dataset（麻省理工学院媒体实验室语音数据集）<br>Pitch and Voicing Estimates for Aurora 2(Aurora2语音库的基因周期和声调估计）<br>Congressional speech data（国会语音数据）<br>Mandarin Speech Frame Data（普通话语音帧数据）<br>用于测试盲源分离算法的语音数据<br>
#### 隐式马尔科夫模型（HMM）
#### 语音信号处理
（1）时域分析：短时能量、短时平均幅度、短时过零率进行语音端点检测<br>
（2）端点检测：双门限起点检测算法 
#### CNN 在语音识别中的应用
[CNN 在语音识别中的应用](https://www.cnblogs.com/qcloud1001/p/7941158.html?utm_source=debugrun&utm_medium=referral)
（1）CLDNN：（CONVOLUTIONAL, LONG SHORT-TERM MEMORY,FULLY CONNECTED DEEP NEURAL NETWORKS）<br>
特征向量用的是40维的log梅尔特征。<br>
[梅尔特征](https://blog.csdn.net/xmdxcsj/article/details/51228791)
>MFCC(Mel-Frequency Cepstral Coefficients)：转化到梅尔频率，然后进行倒谱分析。梅尔刻度的滤波器组在**低频部分的分辨率高**，跟人耳的听觉特性是相符的，这也是梅尔刻度的物理意义所在。倒谱的含义是：对时域信号做傅里叶变换，然后取log，然后再进行反傅里叶变换。可以分为复倒谱、实倒谱和功率倒谱，我们用的是功率倒谱。频谱的峰值即为共振峰，它决定了信号频域的包络，是辨别声音的重要信息，所以进行倒谱分析目的就是获得频谱的**包络信息**。
（2）deep CNN:
GMM-HMM——>DNN-HMM
### 论言语发音与感知的互动机制
#### 发音器官和听音器官频率声能互补
#### 低频敏感区与元音格局
#### 音类扩散分布与感知区别增强
#### 说者协同发音与听者感知补偿
#### 音节中音类分布于演变的不均衡性于相关神经机制

