利用sklearn实现文本分类
=====================
## 1.开发环境
python3.6 + Mac +sklearn
语料库是爬取的新闻排行榜，一共有47872个新闻文档，分为13类。

## 2.代码实现
预处理：采用HanLP.segment分词，去除了停用词
构造词袋：sklearn.feature_extraction.text.CountVectorizer
特征选择（可选）：sklearn.feature_selection.SelectPercentile(chi2, percentile=)
特征提取（可选）：sklearn.feature_extraction.text.TfidfTransformer
采用了十折交叉验证：sklearn.model_selection.StratifiedKFold
分类器：sklearn.naive_bayes.MultinomialNB()

## 3.代码效果
chi2 + TFIDF + MultinomialNB:
测试精确率: 0.8853990894379399
召回率: 0.8440497398407629
F1值: 0.859943175692477

TFIDF + MultinomialNB:
测试精确率: 0.9260021820543615
召回率: 0.8447783964310519
F1值: 0.8674466594383018

chi2  + MultinomialNB:
