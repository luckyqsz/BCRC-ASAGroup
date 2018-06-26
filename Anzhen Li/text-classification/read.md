
利用sklearn实现文本分类
=====================
## 1.开发环境
python3.6 + Mac +sklearn
语料库是爬取的新闻排行榜，一共有47872个新闻文档，分为13类。

## 2.代码实现
预处理
采用HanLP.segment分词，去除了停用词<br>
#### 构造词袋
sklearn.feature_extraction.text.CountVectorizer<br>
#### 特征选择（可选）
sklearn.feature_selection.SelectPercentile(chi2, percentile=)<br>
#### 特征提取（可选）
sklearn.feature_extraction.text.TfidfTransformer<br>
#### 采用了十折交叉验证
sklearn.model_selection.StratifiedKFold<br>
#### 分类器
sklearn.naive_bayes.MultinomialNB()<br>

## 3.代码效果
### chi2 + TFIDF + MultinomialNB:<br>
测试精确率: 0.8853990894379399<br>
召回率: 0.8440497398407629<br>
F1值: 0.859943175692477<br>

### TFIDF + MultinomialNB:<br>
测试精确率: 0.9260021820543615<br>
召回率: 0.8447783964310519<br>
F1值: 0.8674466594383018<br>

### chi2  + MultinomialNB:<br>
