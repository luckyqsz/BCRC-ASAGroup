# about cpp
## 1.c++ vector
https://blog.csdn.net/duan19920101/article/details/50617190/
## 2.图像处理
### 2.1.图像灰度化
用python、matlab、c++实现:https://blog.csdn.net/what_lei/article/details/48681903
### 2.2.图像的缩放
利用resize函数<br>
https://blog.csdn.net/u012005313/article/details/51943442

## 3.数值类型转换
### 3.1.string和char*之间的转换
https://www.cnblogs.com/Pillar/p/4206452.html
### 3.2.OpenCV中将MAT类型的对象作为InputArray类型的对像传递给函数
https://blog.csdn.net/NNNNNNNNNNNNY/article/details/50203223
## 4.about visual studio
### 4.1.无法解析的外部符号
#### 4.1.1.错误显示
![vs_error1](work_record_pic/vs_error1.png)
#### 4.1.2解决方法
可以看到是因为opencv的库出现了问题，可以参考一下步骤解决：<br>1.下载opencv3.4.0。<br>2.在“项目名”——属性——VC++目录——包含目录和库目录中分别加入opencv安装包中的include目录和lib目录<br>3.在“项目名”——属性——链接器——输入——附加依赖项添加lib文件名称（opencv_world340.lib;opencv_world340d.lib;）<br>注意：2和3中release和debug要分别配置。<br>4.另外记得把opencv的dll加入环境变量。
## 5.clock项目进行
### 5.1.处理方法
1.图片放入文件夹，按文件夹将所有图片的导入，批量处理（方便测评accuracy）（已解决）
2.需要注意的是，如果文件夹内的文件超过了100个，不仅在main中需要修改max_pic_num，还要在getAllLabels的参数Labels中修改维度。
### 5.2.resize时需要考虑的问题
1.resize到160*160<br>（已解决）
2.图像在图片中占比问题<br>
3.resize后图像变形问题（按照短边剪裁为方形然后resize）<br>（已解决）
4.目标钟表是否在图片中部问题
### 5.3.gray时需要考虑的问题
1.如果本来就是gray图像的处理
### 5.4.二值化处理
1.运用原来的二值化函数。（已解决）
2.将二值化图片导出。（已解决）
### 5.5.处理中遇到的问题
#### 5.5.1.光线问题
经处理发现，光线问题对图片二值化之后的结果影响较大，有两种解决办法：<br>
1.修改二值化算法<br>
2.对拍照光线作相应要求<br>
个人认为第二种方法比较靠谱。<br>
### 5.6.实验结果
1.如图，对示例的88个图片进行处理得到下面结果，可以看到contour的准确度是0.966,hand是0.955，number是0.943，结果还不错。<br>
![clock_result_ep](work_record_pic/clock_result_ep.png)

### 5.7.存在的一些原理性问题
1.画钟实验是给患者说一句话，让其画钟，然后得到轮廓，指针，数字完备度三方面的打分。但是事实上存在很多问题，比如轮廓为什么不可以是方的，指针要不要画秒针，数字的完备度是否能体现其操作能力等等。<br>
2.如果通过综合性的方法，如7.12组会提出的给出一个标准去训练的方式，可能在具体实践中也需要去给出一个标准，然后去衡量患者画钟的结果，否则这种训练会让标准固化，得分僵化。<br><br><br>
# clock7
## 1.关于项目的备注

```
conda info -e  看所有conda安装的环境<br>
source activate clock   激活clock环境<br>
python predict.py   运行代码<br>
python clock1.py   运行训练代码<br>
```

# 量表的实现
## 1.关于class
class了解：https://blog.csdn.net/dfdfdsfdfdfdf/article/details/52439651


