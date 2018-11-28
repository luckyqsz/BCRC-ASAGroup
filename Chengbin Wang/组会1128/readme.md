# 1. Writing Application Materials

Spent approx. 2 days. 

Strongly recommend reading some English articles (news, technical reports, novels, etc.) daily if you want to write in English. It's hard to express my ideas in English properly, writing is far more difficult than reading. 

# 2. Collecting data in Donghai

Spent approx. 2 days. 

Several problems found. 

1. Poor environment for collecting videos. 

   We need 1) enough space and 2) noiseless backgrounds, but only sufficient space was provided. The camera was facing a large piece of glass which revealed so many reflections of people passing by, providing uncertainty for finding keypoints. 

   The model is not that vulnerable. We can easily omit the incomplete reflections. However, we need to take "controlling unstable backgrounds" into consideration. 

2. Poor quality of clock::pics.

   Our current clock::cameras can not stand stably on the table (when bent). So doctors can only hold A4 papers in front of the "straight (unbent)" camera to take photo in order to accelerate their work. These photos seem to be useless. 

Further consideration:

1. Unify background.

   Can we use some sort of cloth/plastic bags to provide a better, unified background?

   Considering the existence of depth camera, we cannot use ordinary cloth as background material (even torn black plastic bags works). 

2. Improve or change current clock::cameras.

   - Scanner?
   - A camera with well-designed case?
   - Or we design the case of clock::camera?

# 3. Reconstructing clock::code

Spent approx. 0.5 day. Not finished. 

- adding ports for region proposal estimation (digits).
- put methods achieving similar functions into same classes. 
- Rename some methods and variables. 

# 4. Extracting & Reorganizing Datasets

## 4.1 Low quality of original clock::pics

Clocks drawn by patients are saved on 1) A4 papers (which maintains the best resolution) and 2) FTP server (doctors took pictures of those A4 papers **without adjucting light, focus of camera and the relative position**). 

I can not use the clocks saved on FTP server because of 1) low quality , 2) unpredictable light condition, 3) messy background and 4) unpredictable angles **unless I crop these pics manually**. If I choose to crop these pics manually, why not cropping the scanned digital copies of A4 papers? 

## 4.2 Cropping clocks and digits (not completed)

I need to crop 1) the whole clock (for testing the whole evaluation procedure) and 2) separate digits with correct labels (for domain transfer using GAN). That will be a heavy workload. 

The hardest part is: how to label those cropped digits easily? Perhaps I need to write a script. **I need help...**

# 5. Reading

## 6.1 Papers

Uncertainty in Deep Learning (Yarin Gao, Ph.D Dissertation)

Spent approx. 1 day (not finished). 

- I want to spend some time on learning "variational inference".
- I know little about Bayes' Law.  

## 6.2 Books

《深度学习入门：基于Python的理论与实现》

- Not using ANY FRAMEWORKS (tensorflow, caffe, pytorch, ...). 
- Good visualisation.
- Why not code a CNN without using frameworks? Backprop is the most difficult part...
- Quite short, 2 hrs reading, 4 hrs coding. 

《机器学习vs复杂系统》

- I came to know the author in Wechat Official Account (ID: chaoscruiser). His view is different. He has a prior knowledge of physics and neuroscience.  
- It's just an introduction. Treat this book as a "novel". 
- Quite short, 4 hrs of reading. 
- Heuristic.

