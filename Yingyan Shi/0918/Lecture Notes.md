Yingyan Shi

shiyingyan12@qq.com

17th September, 2018

Brain Chip Research Center, Fudan University

***

**Topic:**

**Human Activity Recognition:**

**Past Progress and Emerging Directions**

**Speaker:**

Greg Mori

Program Chair for CVPR 2020

Professor,  School of Computing Science, Simon Fraser University, Canada

Research Director, Borealis AI Vancouver

**Time:**

2018-09-17, 9:00a.m.-11:30a.m.

**Location:**

Shanghai Jiao Tong University Minhang Campus

**Video:**

http://aitutorial.org

To be uploaded later

**Slides:**

To be uploaded later 

while figures below are referred from his slides on **CVPR 2018 Tutorial on Human Activity Recognition**

(http://www2.cs.sfu.ca/~mori/courses/cvpr18/) 

![](images/1.png)

****



Outline:

1. Overview
2. Action Detection
3. Activity Recognition
4. Conclusion

[TOC]

****

## 1. Overview:

**Group Activity Recognition:**

- What does activity recognition involve?
  - To extract semantic knowledge or descriptions about people in the image, such as what are the people doing in the image, where they are, and so on.

![](images/2.png)

![](images/3.png)

![](images/4.png)

Group Activity Recognition:

What is the overall situation？

![](images/5.png)

Levels of  human:

**gestures -> actions -> human-object interactions -> interaction -> human activity** 

AI-complete: full semantic understanding necessary for success.

![](images/6.jpg)

![](images/7.jpg)

![](images/8.jpg)



****

## 2. Action Detection

Activity detection —temporal-level**

![](images/16.jpg)

**Activity detection: spatio-temporal**

- Bounding boxes
- Temporal extent
- Spatio-temporal cuboid containing an action

![](images/17.jpg)



****



**Video classification with CNNs**

![](images/18.jpg)

![](images/19.jpg)

![](images/20.jpg)

![](images/21.jpg)



****



**Spatio-Temporal Action Localization**

- Detailed understanding of **who** did which **actions**, **when** and **where**.
- Application:
  - Video search
  - Surveillance
  - Robotics
  - ...

![](images/9.png)

**Challenges in Action Localization**

- Simultaneous actions
- Start/end point unclear
- Search space is larger O((WH)^T)
- less labeled data
- Technical details:
  - Need more GPU memory, end-to-end training can be prohibitive

![](images/26.jpg)

![](images/11.png)

![](images/12.png)

![](images/13.png)

![](images/14.png)



***

## 3. Activity Recognition

**Group Activity Recognition**

![](images/23.png)

![](images/24.png)

![](images/25.png)



![](images/22.png)



***

## 4. Conclusion

![](images/16.png)

![](images/15.png)

