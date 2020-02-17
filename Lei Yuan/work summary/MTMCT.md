## 2020-2-15

1. 2020 AI City Chanllenge paper submission.

   AI City Chanllenge是CVPR2020的一个workshop，workshop独立于会议审稿，且入选难度比会议小。可以改进下，转投workshop，投稿细节尚未出。

2. 《Multi-Camera Tracking of Vehicles based on Deep Features Re-ID and Trajectory-Based Camera Link Models》——2019 AI City Chanllenge Oral   (未开源)

   与TrackletNet同作者，猜测trackletNet未中转投该workshop。

   ​	这篇文章认为MCT(mutil-camera tracking)问题，可以拆分为三个子问题，第一个子问题是单摄像头多目标跟踪问题(SCT), 第二子问题是外观特征得re-ID问题，第三个子问题是多个摄像头的连接问题ICT(inter-camera tracking)。re-ID也是为了关联不同摄像头拍到的物体。

   ![](C:\Users\91190\Desktop\1.PNG)

   ​	1）trackletNet

   ​	2）re-ID模块

   ​			(a) 使用mask-RCNN提取特征。

   ​			(b) 将提取到的特征导入一个Temporal Attention Model。

   ​			(c) loss function——triplet loss, 该loss用于人脸识别，适用于判别相似的物体。

   ​	3）Trajectory-Based Camera Link Models Because

   ​			![](C:\Users\91190\Desktop\2.PNG)

   ​			将一幅图像分为6块，以标志汽车运动方向，如蓝色箭头可记为 (5, 2)

   ![](C:\Users\91190\Desktop\3.PNG)

   ​			根据运动方向关联多个摄像头，同时由于汽车运动轨迹受到交通规则限制，有些方向路径是不存在的，因此可以根据运动方向避免一些 ID Switch。

   ​		4）experiment

   ![](C:\Users\91190\Desktop\4.PNG)

   ​		5）summary

   ​			最后相机关联的方法值得借鉴。

   3. 后续计划

      1）争取每天看一篇论文，积累关联多摄像头的方法，这方面的论文其实很少，且很多未开源。

      2）以siammot为基础，添加摄像头关联模块测试下结果。

      3）改进siammot，深度受限 ——>小目标跟踪不到，该数据集是否存在这个问题。
      
      ------
      
      
   
   ## 2020-2-16
   
   《Multi-Camera Vehicle Tracking with Powerful Visual Features and Spatial-Temporal Cue》
   
     					                                                                         ——2019 CVPR workshop oral
   
      启发：相机关联分为外观特征提取以及时空关联，外观特征部分是否可以使用siam网络，以每个相机中的每个ID为单位，进行跨摄像头之间的ID匹配。
   
   ​	问题：如何将一个ID单独的从一幅图像中提取出来，Mask？？？
   
   后续计划：跑UW代码
   
   ------
   
   