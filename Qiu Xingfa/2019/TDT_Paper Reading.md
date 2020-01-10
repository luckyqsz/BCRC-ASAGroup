# TDT_Paper Reading

`topic detection and tracking` `single pass clustering` `topic` `event` `story` `topic tracking` `topic detection` `story detection` `event detection` `aspect tracking` `storyline extraction` `news story` `Online Detection` `data stream`

---

## [ 基于增量型聚类的自动话题检测研究](http://search.cnki.net/down/default.aspx?filename=RJXB201206018&dbcode=CJFD&year=2012&dflag=pdfdown)（软件学报 2012）

* 话题检测与跟踪(TDT)的目标就是检测相关信息并跟踪事件的发展变化

* 回顾式的话题检测即检测已有新闻库中尚未发现的话题,在线的话题检测即在线的检测当前到达的新闻所属的话题.

* 定义话题为讨论一致的话题或概念的新闻集,两篇讨论不同特定问题的新闻文档可以属于同一个话题

* 有些文章利用话题的名词实体来提高话题辨别能力,但是这些方法对话题检测性能的提高是有限的

* 话题之间的相似性由两个话题中最相似的两个**子话题**的相似值决定,在聚类的过程中不断提炼基于话题的高辨别性特征词向量,利用该向量表示话题,通过计算BIC(Bayesian information criterion)来判断两个类能否被合并

* 特征的话题辨别能力.它表示一个特征对于区分话题的能力,当一个特征在一个话题出现比较频繁而在其他话题出现比较稀少的情况下,该特征具有较高的话题辨别能力（类似tf-idf）

* 新闻事件都有一个持续时间的特性

* TDT4话题持续性大部分少于4个月，通过某个长度的时间窗口将数据集划分，然后进行预聚类，最后将预聚类好之后的类再进行聚类

* 挑选k-means，CMU（与本文类似，将文档划分再不同的桶里）作为对比

  <br>

* 模型选择方法，计算复杂度太高，不适合海量数据

* 类似于从下到上的分层聚类方法

* 文章使用了分块预聚类的方法

* 使用类似tf-idf的方法挑选具有话题辨别能力的特征，保留大于阈值T的特征，减少了其他系统中基于类的中心向量方法对话题检测性能的影响,减少了话题检测的误差,提高了话题检测的准确率.

* ![](https://github.com/qiuxingfa/picture_/blob/master/2019/f30338a9f0dc998571a124d5ea02621.png)

* 通过BIC值估计话题个数，即通过BIC来判断是否合并两个话题

* 预聚类的方法使得本文的方法已经不是在线聚类了，因为需要提前处理所有数据

---

## 话题检测与跟踪算法改进研究_侯晓冲（华科 硕士论文 2013）

### 相关技术

* ![](https://github.com/qiuxingfa/picture_/blob/master/2019/30bc159392b456620e9005d01bd7518.png)

* 事件（event）：发生在某个特定时间地点的一件事。报道（story）：与话题密切相关的包含了多个独立陈述某个事件的新闻片段
* 五个主要任务
  * 报道切分（story segmentation）
  * 关联检测（story link detection）：判断两个随机产生的报道是否在谈论同一个话题
  * 话题跟踪（topic tracking）：将后面输入的报道与之给定的话题关联起来
  * 新事件检测（new event detection）
  * 话题检测（topic detection）：检测出系统中尚未发现的新话题
* 使用ICTCLAS进行分词和命名实体识别
* 常用的表示模型：布尔模型，向量空间模型，概率模型，语言模型
* 特征选取：
  * 文档频率
  * 信息增益
  * 互信息
  * 卡方统计量
* 文本聚类技术，文本分类技术

### 基于层次聚类的话题检测

* 流程：开始 -> 数据预处理 -> 特征选择 -> 构建向量模型 -> 相似度计算 -> 层次聚类算法 -> 输出新闻话题簇 -> 结果评测簇 -> 结束

### 基于改进的KNN算法的话题跟踪

* 每个类别的分布是无法确定的
* KNN和SVM混合方法

### 系统实验测评

* 评价指标：召回率、准确率、F1值、漏检率、误报率、归一化系统开销
* 使用网络爬虫下载的新闻报道以及使用研究机构标准数据集进行测试，一部分爬虫抓取560篇文章，8个话题，一部分是来源于复旦大学国家数据库中心自然语言处理小组的数据集，其中文档2816篇，10个类别

<br>

* 使用层次聚类的方法可以不设定类别个数
* k近邻算法偏向规模大的话题
* 层次聚类时间复杂度较大
* 话题中错误信息增加就会导致话题偏移
* 文本聚类的主要依据，同类别的文本相似度较大
* CURE使用多个点代表一个簇，并在处理大数据量时随机采样

---

## [DiaQueTT: A Diachronic and Queryable Topic-Tracking Model](https://ears2019.github.io/Nakamura-EARS2019.pdf) （SIGIR2019）

* 词语的意思会随着时间变化，如“云”，结合词语语义的变化检测长时间的话题的变化，diachronic and
  queryable topic tracking (DiaQueTT)
* 将词语语义的变化和TDT结合起来，使用带时间特征的词向量，(Linux cluster, 1995) and
  (cloud, 2015) are close to each other, but the vector (cloud, 1995) is far from them.
* 实验数据为学术文章（Aminer Citation Dataset），15720902篇文章，1990-2014
* ![](https://github.com/qiuxingfa/picture_/blob/master/2019/e4eaa393252b18423ffa2d85ca62298.png)

* 分别对每年的数据训练w2v词向量，不同年份之间的词向量是不可比的，使用了转换矩阵，
* 正则化的词向量*TD-IDF之和表示文档，比Doc2vec和平均的方法鲁棒性要好一些
* 把单词和时间作为共同输入寻找最接近的文档
* 用一个vMF分布表示一个话题，用贝叶斯信息准则决定话题数量，找到话题后，按年分开，表示话题随时间变化的情况
* 话题一致性评估
  <br>
* 大部分语义随时间变化不大
* 将不同年份的词向量映射到同一语义向量空间，用vMF进行文章话题聚类，每个话题内按年表示话题随时间的变化，对结果进行话题一致性评估

---

## [Automatic Evaluation of Topic Coherence](https://www.researchgate.net/profile/David_Newman6/publication/220817098_Automatic_Evaluation_of_Topic_Coherence/links/0deec51cd651fd8e71000000/Automatic-Evaluation-of-Topic-Coherence.pdf?_sg%5B0%5D=djzA9EBIX9oWCf7UxZDFiUx7nX6fQgYvXQsqZovKAOiVbaB4l8i9hjc4pkzHeHxykCIOCLs09Mg8OsE2Kt8Vmw.Pls630zK4lpOfXQrB_HoIlX06T1zA6A6TgNIEP8dMJ9BuBVfkGtMG789eAIZdwbIxvhr5VdlsO9IMOgeccZJbg&_sg%5B1%5D=uX9Y576Ytq2oYHiHy2bBLqZenM_By9UL24IGqLDoxf_dtp5_Q9npc5oCTECEoK-60uUkvEtShJjgNaS9Da7ryrvKS38fXfNS8BSVHL4TvyXJ.Pls630zK4lpOfXQrB_HoIlX06T1zA6A6TgNIEP8dMJ9BuBVfkGtMG789eAIZdwbIxvhr5VdlsO9IMOgeccZJbg&_iepl=)（NAACL 2010）

* 主题一致性的内在语义评估，提出一种全自动评估方法，达到接近人类的准确度
* 用LDA产生话题，top-n
* 人工评估
  <br>
* 主题模型主要是从文本中提取关键词
* 数据包括55000新闻文章和12000书籍，新闻话题数T=200，书籍T=400，随机挑选273个话题，9个人进行打分

---

## [Temporal Event Detection Using Supervised Machine Learning Based Algorithm](https://link.springer.com/content/pdf/10.1007%2F978-3-030-16681-6.pdf)（IBICA 2018）

* 提取事件信息和时间信息的关系，使用有监督的机器学习技术和基于规则的方法
* 事件提取：从文本中提取事件元素（ACTION，STATE，OCCURRENCE...）
* 事件信息和时间信息分开抽取，然后再判断事件和时间的关系，使用神经网络的方法进行训练
* 新闻数据集按 3：1比例训练和测试

---

## [Research on Topic Detection and Tracking for Online News Texts](https://kopernio.com/viewer?doi=10.1109/access.2019.2914097&route=6)（IEEE Access 2019）

* 用LDA提取话题，用吉布斯采样计算参数，用单遍遍历追踪话题，用JS (Jensen-Shannon) divergence表示话题相似性，增加时间衰减函数，JS(P,Q) = 1/2KL(P,(P+Q)/2) + 1/2KL(Q,(P+Q)/2)
* 主题模型LDA也被频繁用在TDT任务中
* variable online LDA
* Single-Pass clustering algorithm with sliding time window
* 按照最低复杂度来选择话题的个数
* 使用TDT2003的评价标准
* LDA效果比K-means效果更好
* 使用了3000条十类搜狗新闻数据，随机选5类做聚类个数为5的聚类，比较话题提取的效果
* 使用七天的新闻数据进行话题跟踪实验，根据某一天的实验数据得出一天数据的话题数为150，平均一天5000条数据，用滑动窗口的方法，从不同时间提取出的主题词体现事件的变化趋势，选择“花莲地震”作为特征事件进行评估，还进行了热度分析
  <br>

* 中文数据
* 很多新闻的寿命不超过一天
* 话题追踪并没有很好地进行实验评估
* 依然指定了一天的话题数
* 用LDA进行话题提取，用KL散度表示相似度进行聚类然后进行话题跟踪实验

---

## [An Adaptive Topic Tracking Approach Based on Single-Pass Clustering with Sliding Time Window ](https://kopernio.com/viewer?doi=10.1109/ICCSNT.2011.6182201&route=6)(ICCSNT 2011)

* 通常在不同时间点同一话题的分布是独立的，传统的关键词搜索方法只是因为包含特定的关键词就返回信息，会导致有很多冗余信息
* 预处理过程包括分词，特征选择（名词，动词和命名实体），权重计算（TF-IDF）
* 计算余弦相似度
* 计算故事和话题之间的相似度，通过阈值控制聚类的粒度并发现新的类别，单遍聚类的缺点是聚类结果受输入顺序的影响
* 设置两个阈值0.7和0.45，以向量平均值表示类别，处于两个阈值之间的故事将会在k个时间窗口（2天）内多次计算进行话题分配
* TDT-4 数据集
  <br>

* 现实中的新闻数据就是有时间顺序的，我们也不可能一次性把所有数据输入

---

## [Automatic Online News Issue Construction in Web Environment](https://kopernio.com/viewer?doi=10.1145/1367497.1367560&route=6)(WWW 2008)

* 没有考虑话题重叠
* 对wf进行动态更新
* 故事平均值表示话题
* 分层聚类的方法，合并相似度最高的两个类
* 350篇文章87话题，953篇文章108个话题，24872篇新闻文章和1339篇博客论坛文章
* 使用P，R和F1作为评估
* 在数据集2中使用不同的聚类方法实验
* 去除冗余句子提升效果，考虑标题提升效果，只考虑短文本标题效果最好，
* 实时结果和日/周/月排行（彩票出现在月排行），对结果进行定性评估

---

## [An Online Topic Modeling Framework with Topics Automatically Labeled](https://www.aclweb.org/anthology/W19-3624) (ACL2019)

* 手动标注了507篇报道6个类别，总的7076篇股票交易报道
* 之前的话题追踪方法基本上基于LDA
* 投票数，浏览数和文章长度表示文章质量
* 用报道的话题分布作为输入训练SVM分类器
* 话题一致性评估

---

## [Tracking Aspects in News Documents](https://kopernio.com/viewer?doi=10.1109/ICDMW.2018.00165&route=6)（ICDMW 2018）

* 使用HMM跟踪话题，考虑故事之间的转换状态，维特比算法选择最佳路径，HMM根据序列数据产生MM，基于两个假设，每个话题基于统一的马尔可夫模型结构，按年代排列的文章作为一个方面在概率上是可区分的
* 没有通用的框架去对形势追踪建模，对于突发新闻容易追踪错误
* 使用带标签的新闻数据，手动挑选特定几个话题的文章打乱，按时间排序，提取了文章中的专有名词和动词，KL散度表示距离，
* 选取特定话题的数据进行实验

---

## [话题检测与跟踪的评测及研究综述](http://search.cnki.net/down/default.aspx?filename=MESS200706014&dbcode=CJFD&year=2007&dflag=pdfdown)（中文信息学报 2007）

* 语料：
  * TDT-Pilot：收集了1994年7月1日到1995年6月30日之间约16000篇新闻报道，标注人员从所有语料中人工识别涉及各种领域的25个事件作为检测与跟踪对象。
  * TDT2收集了1998年前六个月的中英文两种语言形式的新闻报道。其中，LDC人工标注了200个英文话题和20个中文话题
  * TDT3收集了1998年10月到12月中文、英文和阿拉伯文三种语言的新闻报道。其中，LDC对120个中文和英文话题进行了人工标注
  * TDT4收集了2000年10月到2001年1月英文、中文和阿拉伯文三种语言的新闻报道。其中，LDC分别采用三种语言对80个话题进行人工标注。
  * TDT5收集了2003年4月到9月的英文、中文和阿拉伯文三种语言的新闻报道。LDC对250个话题进行了人工标注
  * 论文中有评测工具，指南即语料的获取方式，语料需购买
* 话题**定义**：一个话题由一个种子事件或活动以及与其直接相关的事件或活动组成。论文举例：关于“联邦航空局通过调查飞机坠毁的原因修改航空条例”的报道与飞机坠毁的话题并不相关。
* 话题检测实质上为跟踪系统提供了先验的话题模型，而话题跟踪则辅助检测系统完善对话题整体轮廓的描述。
* 话题跟踪
  * **传统**话题跟踪，基于分类策略的话题跟踪研究，kNN和决策树，二元分类（精确度高，但是依赖于训练语料和分类器，先验报道的稀疏性一定程度上影响了二元分类方法的召回率），还有相似度匹配算法，由于构造话题模型的初始信息相对稀疏，很多方法都无法有效跟踪一段时期以后话题的发展。
  * **自适应**话题跟踪，初始训练得到的话题模型不够充分和准确，一种具备自学习能力的无指导自适应话题跟踪(Adaptive Topic Tracking，简写为ATT)逐渐成为TT领域新的研究趋势。有通过摘要代表文章。自适应本质上是一种伪反馈，导致话题漂移。有通过二次阈值截取的方法，设置更高的阈值进行更新。
* 话题检测
  * 在线话题检测，单路径聚类算法
  * 新事件检测，统计模型的最大缺陷在于无法有效区分同一话题下的不同时间，以名实体为主的特征集对于不同事件的区分贡献更高
  * 事件回顾检测，回顾过去所有的新闻报道，从中检测出未被识别的新闻事件，有些话题跳跃式地出现在不同时间，时间跨度更长
  * 层次话题检测，凝聚层次聚类方法（时间和空间复杂度过高），改进为增量式层次聚类方法

---

## [Topic Detection and Tracking Evaluation Overview](https://kopernio.com/viewer?doi=10.1007/978-1-4615-0933-2_2&route=1)

* **story**:a topically cohesive segment of news that includes two or more declarative independent clauses about a single event
* **event**: meaning something that happens at some specific time and place along with all necessary preconditions and unavoidable consequences
* **topic**: a seminal event or activity, along with all directly related events and activities
* **Topic Tracking** – detect stories that discuss a target topic,
* **Link Detection** – detect whether a pair of stories discuss the same topic,
* **Topic Detection** – detect clusters of stories that discuss the same topic,
* **First Story Detection** – detect the first story that discusses a topic, 
* **Story Segmentation** – detect story boundaries

---

## [Hierarchical clustering based on single-pass for breaking topic detection and tracking](http://search.cnki.net/down/default.aspx?filename=GJST201804005&dbcode=CJFD&year=2018&dflag=pdfdown)（HIGH TECHNOLOGY LETTEＲS 2018）

* 单遍聚类算法在面对海量数据时优势明显，简单有效但是计算复杂度高，提前进行数据分块（分类假设，假设在共同目录下或有相似特征的文章比较接近，假设时间接近的文章更有可能讨论同一话题），话题可能随着时间发生变化
* topic model适合稳定的话题，而不适合变化的
* 文章由单词和词性组成，命名实体以及对应的TF-IDF权重表示文章向量，词袋模型表示性能不好，有些句子相似度很高但属于不同事件，分为关键特征向量（标题）和命名实体特征向量，关键词包含事件，名词，动词给予较高权重，一般的单词和短语给予较低的权重。改进的TF-IDF表示，给予不同的词性，是否位于标题权重
* 多层级多粒度策略适合海量数据
* 核心报道的选择（代表话题）：与话题相似度更高的，有用信息更多的（命名实体数），最新的；选择k篇文章代表话题，融合成一篇，并不断更新，长文本使用余弦距离，短文本则用基于HowNet的杰卡德相似度
* 淘汰过时话题，话题**能量**由文章和话题相似度以及时间总体作用
* 利用网站分类信息或用朴素贝叶斯分类器
* 三层聚类：
  * 基于单遍遍历和时间窗口（一天）的局部聚类
  * 分别在不同类别中基于局部聚类结果的凝聚聚类
  * 在总体不同类别基于KNN的凝聚聚类，K=1
* 聚类性能由新浪新闻数据420篇10个话题数据，时间复杂度评估使用283638篇数据，从2015年10月到12月

---

## [Analyzing the Changes in Online Community based on Topic Model and Self-Organizing Map](https://www.researchgate.net/publication/281537461_Analyzing_the_Changes_in_Online_Community_based_on_Topic_Model_and_Self-Organizing_Map/fulltext/5641a11708aec448fa611d6d/Analyzing-the-Changes-in-Online-Community-based-on-Topic-Model-and-Self-Organizing-Map.pdf?_sg%5B0%5D=nbYwnOBM7HJKQCi9pi_YK39-EW_6_TZVg9h-WKMOAWhbAlrcAFwKwSYO8kERs0F7roVckxIzSYckc3U7Vb6_Pg.hnCvgeDfvniNr-_cZVBmqvom-Irjpn0Sut5kxg4siDfjwWHpheXZFN-ujZgvQPqWVEnWMfwbPvS8_TX3LLb-Sg&_sg%5B1%5D=TbdqCCLyB0E7Ks6EWK9bRUMDKHPs4YMt8IhlMxJqQESUgSstupfaUrpEPEbnmgJn4k30EdC132SYxlGvhtmUevW8MypNzpo7MiOcTmkuwAFe.hnCvgeDfvniNr-_cZVBmqvom-Irjpn0Sut5kxg4siDfjwWHpheXZFN-ujZgvQPqWVEnWMfwbPvS8_TX3LLb-Sg&_iepl=)（IJACSA 2015）

* a **topic** is a mixture component defining a distribution of words
* 分析话题和用户随时间在社区中的变化
* Temporal – Author – Recipient – Topic model (TART)

---

## [A single pass algorithm for clustering evolving datastreams based on swarm intelligence](https://kopernio.com/viewer?doi=10.1007/s10618-011-0242-x&route=6)(Data Min Knowl Disc 2011)

* 分散的自底向上的自自组织策略
* 指定聚类数目，对于过时和最近数据使用相同权重，没有捕捉到数据的变化
* 分为online和offline两个部分，
* CluStream有两个缺点，一是不能发现任意形状的类别，二是类别数需提前指定，基于密度的聚类方法可以克服以上缺点（DenStream，D-Stream，MR-Stream），
* 权重随着时间的变化指数衰减，对每一个新数据，判断属于哪一种微聚类，core-micro-cluster，potential c-micro-cluster，outlier micro-cluster，DenStream不能处理大量的数据因为它对于每一个新数据都需要寻找最近的微聚类
* FlockStream,用并行的随机局部多主体搜索取代了全局搜索最近邻的方法，每个中介只与可见距离内的中介进行比较，数据：DS1，DS2，DS3

---

## [Story Disambiguation: Tracking Evolving News Stories across News and Social Streams](https://arxiv.org/pdf/1808.05906.pdf)(arXiv 2018)

* 基于分类的方法需要大量的正负标注样本，对于随时间变化的故事并不敏感，只能用于一种文本（如新闻文章）
* 之前的大部分方法集中于一种文本，本文同时考虑了新闻文章和社交媒体文本
* 实际上，一个故事由几个关键的命名实体决定，所以实体图是一种有效表示故事的方法
* 使用了一种半监督的方法自动更新故事的实体图和特征向量，以反映故事随时间的变化，分类器不变而是对特征进行更新
* 监督的TDT方法基本上假定于一个**静态**的环境
* 将故事追踪任务当成L2R（learning-to-rank）任务
* 提供少量的文本，目标故事由用户进行初始化，用**半监督**的方法对故事进行更新，对于新加入的数据，并没有对分类器进行重新训练而只是对事件特征进行更新
* 只考虑**最近的文本**可能会产生偏移，
* 选择维基百科作为外部知识，用NLTK进行命名实体识别，并用词性标注识别名词短语以扩展命名实体列表，用TAGME进行**实体消歧**，将短时间内（如1小时）带有相同标签的推特组合起来作为一篇文章，以实体id，实体位置和消歧置信分数组成文本表示，**可变的**时间窗口增加前面出现的实体的权重，以时间窗口中的共现表示实体间的关系，权重为共现次数
* 用NetworkX构建故事实体图，以及计算biased PageRank权重
* 事件为已知的，事件作为query但是会变化，对于L2R模型，有三种特征，query，document，association between them，也可以分为图特征和文本特征
* ![](https://github.com/qiuxingfa/picture_/blob/master/2019/2a6422dd5e4cf9ca34a8bd4f46d6d2f.png)
* 使用**Random Forest**作为分类器，选择“2016爱尔兰大选”作为训练集，选择其他八个故事作为测试集，选择900篇相关文章，1800篇相关推特，以及27000篇无关文本，模拟数据流，产生了15个不同的故事表示，训练29.7k×15个故事-文本对，每50篇文章加入故事更新一篇，每500篇文章重新分类一次，挑选置信度最高的一篇更新
* **删除了标题的关键词**（类似标签）**和推特的标签**，十倍的相同时间窗口内的负样本
* SOTA方法：1）binary classification; 2) story clustering / topic model; 3) learning-to-rank
* 越复杂的事件越难追踪

---

## [Hashtagger+: Efficient High-CoverageSocial Tagging of Streaming News](https://kopernio.com/viewer?doi=10.1109/tkde.2017.2754253&route=7)（IEEE T Knowl Data En 2018）

* 一个故事可以有多个标签，在故事的不同阶段可能会有不同的标签
* 问题定义：将文章映射实时地到推特标签上来，L2R模型分为 pointwise，pairwise，listwise
* 以分类的思想（数据稀疏性以及噪音），以主题模型的思想（新标签），深度神经网络多分类（静态）
* 使用L2R的方法将排序问题转为分类问题
* MCC方法：使用有标签的推特作为训练集，对于概念的变化需要重新训练，每五分钟使用过去4小时的数据重新训练
* 标签的噪音很多，因为很多用户的文本和标签不一定相关，不能算很好的标注数据

---

## [STREAMCUBE: Hierarchical Spatio-temporal HashtagClustering for Event Exploration over the TwitterStream](https://kopernio.com/viewer?doi=10.1109/ICDE.2015.7113425&route=6)（ICDE 2015）

* 提出问题：
  与一般的文档不同，推特可能提供地理信息和话题信息，从而可以从不同的空间粒度和时间粒度发现事件
* 贡献
  * 对于推特数据流的一个分层的时空粒度的话题聚类
  * 从不同的空间粒度和时间粒度发现事件
  * 使用高效的单遍遍历算法，提供可理解的聚类标签
  * 在一个特定的时空条件下进行有效的事件排序，发现爆炸事件或局部的事件
  * 与传统方法不同，本文从推特流中实时地处理数据
  * 递增地将新数据与旧数据融合，
* 方法
  * 由三个部分组成，1）时空聚类；2）单遍遍历的话题标签聚类；3）事件排序
  * 事件就是标签的集合，而标签是随时间变化的，基于标签的聚类具有可解释性
  * 考虑时间和空间粒度，将数据划分到一个时空之内进行小范围的聚类，
  * 在时间和空间上构建quad-tree，每次四等分，三层，只把最近6小时的数据当作流入的新数据（`类似三位编码`）
  * 标签的表示方法：1）所有使用当前标签的推特文本集合；2）多个标签的集合表示，55%的文本有多个标签，多标签是用户提供的标签聚类
  * 用标签的word vector 和 hashtag vector 的加权余弦相似度来表示相似度，事件的表示和相似度计算类似。
  * 标签和事件都是随时间变化的
  * 对于静态的标签（假设标签不变）聚类，为了减少计算量，只把和当前标签有共同部分的聚类作为候选（`对于单标签的数据不友好，会漏掉很多不共现的标签`），如果当前标签和最近邻事件的相似度小于当前事件和其他事件的最小相似度，那就应该成立一个新事件（`初始相似度阈值如何定义，标签和事件的相似度和事件之间的相似度是否有可比性`）
  * 对于变化的标签，新标签出现时，先当作不活跃的新事件，超过30条推特标签变为活跃状态，每三十次更新检查聚类是否需分裂或融合（`计算量较大`）
  * 按照 热度，突发性和局部性 对事件排序，通过有监督的训练，得到权重值
* 数据
  * 获取900万条推特，有200万标签，时间为从2013年12月到2014年1月，没有地理位置标签的使用用户位置来代表或随机采样（`不合适`）
  * 对比不同方法选出的top5事件（`对于一个事件，也就是标签的集合，如何选出一个代表的标签`）
  * 融合不同方法选出的top50标签作为候选，让十个人投票选出top10
* 存在问题
  * 对于无标签和单标签的数据不友好
  * 计算聚类质量的时候，ground truth的产生方法存疑，250个标注事件是如何标注的，对于变化的标签以及变化的事件是如何评估的，如何体现动态聚类的结果
  * 流式数据是连续的，强行切分会导致切分点附近的数据不连续，如果只做一次聚类，那一个标签可能会在几个部分中存在，那是如何分配的呢
  * 标签是很重要的信息，如何利用标签信息，如何处理无标签，单标签，多标签的关系，标签的粒度也是不一致的，如何体现标签的变化（取最近一段时间的文本表示）

---

## [Mining Text and Social Streams: A Review](https://kopernio.com/viewer?doi=10.1145/2641190.2641194&route=6)（KDD 2014）

* 初始化k个类，后来的数据和已有的类做相似度计算，超过阈值的归入当前聚类，否则作为异常值，取代最不活跃的类
* 突发特征表示，使用一些突发性特征优化聚类结果，
* 计算当前文档和之前k个文档的相似度

---

## [LEARNING GENERAL PURPOSE DISTRIBUTED SENTENCE REPRESENTATIONS VIA LARGE SCALE MULTITASK LEARNING](https://arxiv.org/pdf/1804.00079.pdf)（ICLR 2018）

* 将不同的句子表示方式融合进一个多任务学习的框架

---

## [BB_twtr at SemEval-2017 Task 4: Twitter Sentiment Analysis with CNNs and LSTMs](https://www.aclweb.org/anthology/S17-2094)(2017)

* 使用100万推特预训练词向量
* CNN模型选择推特词长度为80
* 预训练的词向量包含的情感极性信息很少，正负极性的词距离可能很近（“好”和“坏”），再用远程监督方法进行微调，

---

## [Embeddings for Word Sense Disambiguation: An Evaluation Study](https://www.aclweb.org/anthology/P16-1085)（ACL 2016）

---

## [On clustering network-valued data](https://papers.nips.cc/paper/7282-on-clustering-network-valued-data.pdf)（NIPS 2017）

---

## [Short Text Topic Modeling Techniques,Applications, and Performance A Survey](https://arxiv.org/pdf/1904.07695.pdf)（arXiv 2019）

* 短文本的词共现信息十分有限，传统方法在短文本上出现性能退化
* 三种方法：Dirichlet multinomial mixture，global word co-occurrences，self-aggregation
* 传统的主题模型：PLSA和LDA
* 某些短文本也不一定单主题
* DMM假设每篇文章只属于一个潜在的话题
* 把迪利克雷分布作为先验分布

---

## [See What’s enBlogue – Real-time Emergent Topic Identification in Social Media](https://kopernio.com/viewer?doi=10.1145/2247596.2247636&route=6)（EDBT 2011）

* 提出问题：
  * 在大量的数据流中，新兴产生的话题，通常不是由一个标签表示，而是由很多不同的标签集合而成，通过检测单独的标签来检测新兴话题是不合适的
  * 对于文档的标签，可直接给出或者通过话题分类或命名实体识别给出，但是，这些标签通常不精确且只强调了一个方面，比如，“France”,“Cuba”, 或 “Vancouver”的话题
* 论文贡献
  * 数据为 (timestamp, document, tagset) 三元组
  * 提出enBlogue，用来检测emergent topics，使用滑动窗口来计算标签和标签对的统计数据，这些数据用来检测不同寻常的变化，
  * 对于文档的tag，可直接给出或者通过话题分类或命名实体识别给出
* 两个标签的关联性由标签的交集程度或杰卡德相似度定义
* 三个部分：种子标签的选择，关联追踪，变化检测

---

## [Event Detection and Retrieval on Social Media](https://arxiv.org/pdf/1807.03675.pdf)（arXiv 2018）

* 三个主要类别：1）新闻相关；2）娱乐事件；3）个人事务
* 大量的噪音和无意义的内容使得这项任务变得困难，有些事件会有大量的与之相关的文本，从中选出信息丰富的具有代表性的文本也是一项具有挑战性的工作
* 事件检测的三种方法：
  * 基于特征的
    * 特征可以从关键词，命名实体到社交
    * 基于特征的时间分布
    * SVM分类器
  * 基于文本的
    * 使用贝叶斯分类器判断是否新闻，TF-IDF表示tweet
    * 词汇的多样性使得同一事件出现不同的词语，在微博客上这种现象更加明显
  * 主题模型
    * 对于社交媒体的处理，不仅把词语当作可观测量，还要考虑用户，图片信息，发布时间和地址等信息，对于短文本来说，一个文本是许多话题的集合的假设可能有点问题
    * LECM（Latent Event and Category Model）：包含相同的命名实体，关键词，相近的时间和地点更可能是同一事件
* 简短，非正式，不合乎文法的文本使得社交媒体的摘要变得更加困难
* 评估
  * P，R，F1
  * NMI（Normalized Mutual Information）
  * 摘要：ROUGE
  * 定性分析
  * recall较难计算

---

## [Cluster Ensembles – A Knowledge Reuse Framework for Combining Multiple Partitions](https://www.researchgate.net/profile/Joydeep_Ghosh6/publication/220320167_Cluster_Ensembles_-_A_Knowledge_Reuse_Framework_for_Combining_Multiple_Partitions/links/00463521fe3b4643bd000000/Cluster-Ensembles-A-Knowledge-Reuse-Framework-for-Combining-Multiple-Partitions.pdf?_sg%5B0%5D=TqIQ6wGXEMXtwA_iADV5oXchPiZn5AyK3Qt_jZkHftEVM3EOGGLpCeUW9fG664gBHxNrqQOF617AMq5JS51WQw.NSr2fq4jPKQyUAaBQVDQs2icrf8-IcLf3JWDcIn_JL_Lq9MYFvRm5lNnTE9mid-M-rI7ebbnQNwQhX9LTEpnKg&_sg%5B1%5D=9Tb4aPkeCn7Tt8nXduAzEoiZ9ute1SvcxtZ5-NgUaEmvb9GWn7naF9x3tYXMgr6IA3AwE2-vpR3iYm36Sfi--E_fnWoLIlb4MSWrN8Hntxf7.NSr2fq4jPKQyUAaBQVDQs2icrf8-IcLf3JWDcIn_JL_Lq9MYFvRm5lNnTE9mid-M-rI7ebbnQNwQhX9LTEpnKg&_iepl=)（J Mach Learn Res 2002）

* 聚类融合
* 进行聚类融合的动机是增加聚类算法的健壮性

---

## [Event Detection in Social Streams](http://charuaggarwal.net/sdm2012-camera.pdf)（SDM 2012）

### Introduction

* study the two related problems of clustering and event detection in social streams
* study both the supervised and unsupervised case for the event detection problem
* illustrate the effectiveness of incorporating network structure in event discovery over purely content-based methods
* it is often necessary to process the data in a ``single pass`` and one cannot store all the data on disk for re-processing
* ``study the problems`` of clustering and event detection in social streams in which each text message is associated with at least a pair of actors in the ``social network``
* both network locality and structure need to be leveraged in a dynamic streaming scenario for the event detection process

### key challenges for event detection

* The ability to use ``both the content and the (graphical) structure`` of the interactions for event detection.
* The ability to use ``temporal information`` in the event detection process
* The ability to handle very ``large and massive`` volumes of text documents under the one-pass constraint of streaming scenarios

### The Model

* the object Si is represented by the tuple (``qi``(tweeting actor),``Ri``(recipient set), ``Ti``(twitter content))
* ``NOVEL EVENT:`` The arrival of a data point Si is said to be a novel event if it is placed as a single point within a newly created cluster Ci. ``An evolution event`` is defined with respect to specific time horizon and represents a change in the relative activity for that particular cluster. ``fractional presence`` is denoted by F(t1, t2, Ci)
* use ``sketch-based methods`` in order to compute the similarities more effectively
* ``cluster-summary`` ψi(Ci) = (Vi, ηi,Wi,Φi)
* In each iteration, we compute the similarity of the incoming social stream object Si to ``each cluster summary ψi(Ci)``.we maintain the mean μ and standard deviation σ of all closest similarity values of incoming stream objects to cluster summaries. The similarity value is said to be significantly below the threshold, if it is less than ``μ−3·σ``. If a new cluster is created ,the cluster ``replaces`` the most stale cluster from the current collection C1 . . . Ck.
* In order to compute the similarity, we need to compute both the structural ``SimS(Si, Cr))`` and the content components ``SimC(Si, Cr)`` (TF-IDF)of the similarity value

### Event Detection with Clustering

* it is always assigned to its closest cluster
* The assumption in the supervised case is that the training data about the social stream objects which are related to the event are available in the historical training data（有历史数据来表征事件特征）

### Experimental Results

* dataset
  * Twitter Social Stream: 1,628,779 tweets, 59,192,401 nodes, 
  * Enron Email Stream: a total of 349911 emails, a total of 29,083 individuals
* Evaluation Measures
  * For the case of the Twitter stream, these class labels were the ``hash tags`` which were associated with the tweet. the hash tags were only associated with a subset of the tweets. 
  * For the case of the Enron email stream, the class labels were defined by the ``most frequently occurring tokens`` in the subject line.
  * For the case of the unsupervised algorithm, we will present ``a case study``
* network information in the social stream provides much more powerful information for clustering as compared to the text-information

### 思考

* 只是以简单的TD-IDF计算相似度
* 没有利用时间特征
* 以时间窗口的方法体现的事件的变化
* 使用标准**正态分布**来挑选阈值，μ − 3 · σ
* 社交网络相似度的计算没太看懂
* 以hashtag作为ground truth并不合理，如果能直接识别出hashtag，聚类将没有意义，对于多标签的处理也不明确

---

## [Identifying Content for Planned Events Across Social Media Sites](http://delivery.acm.org/10.1145/2130000/2124360/p533-becker.pdf?ip=202.120.224.53&id=2124360&acc=ACTIVE%20SERVICE&key=BF85BBA5741FDC6E%2E88014DC677A1F2C3%2E4D4702B0C3E38B35%2E4D4702B0C3E38B35&__acm__=1570539867_62c67a2fe4e6077fb3bc6f0e28ed3534)（WSDM 2012）

### INTRODUCTION

* explore approaches for identifying diverse social media content for planned events
* Automatically identifying social media content associated with known events is a challenging problem due to the heterogenous and noisy nature of the data.
* propose a ``two-step`` query generation approach: 
  * the first step combines known event features into several queries aimed at retrieving high-precision results; 
  * the second step uses these high-precision results along with text processing techniques such as term extraction and frequency analysis to build additional queries, aimed at improving recall

### RELATEDWORK

* quality content extraction in social media
* event identification in textual news
* event identification in social media

### MOTIVATION AND APPROACH

* define an event as a real-world occurrence e with 
  * an associated time period Te and 
  * a time-ordered stream of social media documents De discussing the occurrence and published during time Te
* ![](https://github.com/qiuxingfa/picture_/blob/master/2019/18153cc6ad50eccf1d31b9374960bd1.png)
* In a study of trends on Twitter, Kwak et al.discovered that most trends last for one week once they become "active"
* We consider two important criteria for ordering the event queries: ``specificity``(rank long, detailed queries) and ``temporal profile``
* we can leverage event content from one social media site to help retrieve event documents from another social media site in different ways

### EXPERIMENTS

* Comparison of the automatically generated queries against human-produced queries for the events
* Evaluation by human judges of the automatically generated queries
* Evaluation of the quality of the documents retrieved by the automatically generated queries
* Planned Event ``Dataset``: We assembled a dataset of event records posted between May 13, 2011 and June 11, 2011 on four different event aggregation platforms
* comparison against human-produced queries, human evaluation of generated queries, and evaluation of document retrieval results. use the ``Jaccard coeficient`` to measure the similarity of the set of automatically generated queries G to the set of human-produced queries H for each event.
* asked two annotators to label 2,037 queries selected by our strategies for each event on a scale of 1-5, based on their relevance to the event.

### 思考

* 类似于从事件文本中提取代表文本、代表的关键词，提取有效信息，在选取事件表示时值得借鉴
* 事件是有时效性的
* 结合了基于准确率和基于召回率的方法，即考虑不同粒度
* 文章还使用了不同来源的文本，相互促进

---

## [Event Discovery in Social Media Feeds](http://aria42.com/pubs/events.pdf)（ACL 2011）

### Abstract

* We develop a **graphical model** that addresses these problems by learning a latent set of records and a record-message alignment simultaneously   

### Problem Formulation

* **Message**: a single posting to Twitter
* **Record**: A record is a representation of the canonical properties of an event (Throughout, we assume a known fixed number K of records)
* **Message Labels**: We assume that each message has a sequence labeling
* ![](https://github.com/qiuxingfa/picture_/blob/master/2019/d3bf4e8b313304f4403f856495045ab.png)

### Model

* ![](https://github.com/qiuxingfa/picture_/blob/master/2019/d949e4c11ec344a081afae931a9fb44.png)
* One challenge with Twitter is the so-called **echo chamber effect**: when a topic becomes popular, or “trends,” it quickly dominates the conversation online. As a result some events may have only a few referent messages while other more popular events may have thousands or more.

### 思考

* 转为序列标注问题，即事件信息提取，提取人物和地点
* 推特中的每一个词都有表示事件属性的标签

---

## [TwitterNews: Real time event detection from the Twitter data stream](https://www.researchgate.net/profile/Mahmud_Hasan8/publication/309426330_TwitterNews_Real_time_event_detection_from_the_Twitter_data_stream/links/581a12b008aeffb294130fd1/TwitterNews-Real-time-event-detection-from-the-Twitter-data-stream.pdf?_sg%5B0%5D=TnznRnCrOp6ZZclRyRwcqEB4IIRkbOvPhDGOkLY403iD2TAh87WNHmQ3YgnlhW8H_kxDJ4o4zW7AvOJIUF_s6Q.rvivN9t4tv9QsIZ-bRsBICXS9YUkxh3FnhWYdRYuE0aqJQP9dlN752umgbo-SgQC6Or5VYV4nH1rhlxVQ0vbrQ&_sg%5B1%5D=W7u992XqavOYSoexjseJ7Q97t4XvzkCJC-9DGowiktOVZ4moje8qyeerx2nuRO5dB9wPyEYIS7EGMEmtYDwb_bLE83W_ZeQ7AW1GznnRG27r.rvivN9t4tv9QsIZ-bRsBICXS9YUkxh3FnhWYdRYuE0aqJQP9dlN752umgbo-SgQC6Or5VYV4nH1rhlxVQ0vbrQ&_iepl=)（PeerJ Preprints 2016）

### Abstract and Introduction

* **TwitterNews** provides a novel approach, by combining random indexing based term vector model with locality sensitive hashing, that aids in performing incremental clustering of tweets related to various events within a fixed time.
* **definition of an event**: An occurrence causing change in the volume of text data that discusses the associated topic at a specific time. This occurrence is characterized by topic and time, and often associated with entities such as people and location
* The approaches based on **term interestingness** and **topic modeling** suffer from high computational cost among other things
* we propose an incremental clustering based end-to-end solution to detect newsworthy events from a stream of time ordered tweets
* ![](https://github.com/qiuxingfa/picture_/blob/master/2019/503e217a772d365266fb56b078e0e4b.png)
* **The operation in the first stage** of TwitterNews is implemented by combining Random Indexing (RI) based term vector model with the Locality Sensitive Hashing (LSH) scheme proposed by Petrovic et al. to determine whether a tweet is “not unique”. 
  * Using the **Search Module** we aim to detect a soft burst, that is, we intend to find at least one previous tweet that discusses the same topic as the input tweet
  * A **fixed number** of hash tables are maintained to increase the chance of finding the nearest neighbor of the input tweet.
* Subsequently **the second stage**, implemented using a novel approach by adapting the generic incremental clustering algorithm, deals with generating the candidate event clusters by incrementally clustering the tweets that were determined as bursty (“not unique”) during the first stage.The second stage also incorporates a **defragmentation strategy** to deal with the fragmented events that were generated when a particular event is detected multiple times as a new event.
* **Finally** a set of filters are applied after the second stage to report the newsworthy events from the candidate event clusters

### Related Work

* Term Interestingness Based Approaches(``often capture misleading term correlations, and measuring the term correlations can be computationally prohibitive in an online setting``)
* Topic Modeling Based Approaches (``incur too high of a computational cost to be effective in a streaming setting, and are not quite effective in handling events that are reported in parallel``)
* Incremental Clustering Based Approaches (``are prone to fragmentation, and usually are unable to distinguish between two similar events taking place around the same time``)
  * Becker et al . For each tweet, its similarity is computed against each of the existing cluster.Once the clusters are formulated, a **Support Vector Machine** based classifier is used to distinguish between the newsworthy events and the non-events
  * McMinn et al.  have utilized an **inverted index** for each named entity with its associated near neighbors to cluster the bursty named entities for event detection and tracking
  * Petrovic et al.  have adapted a variant of the **locality sensitive hashing** (LSH) technique to determine the novelty of a tweet by comparing it with a fixed number of previously encountered tweets

### Experiment Results and Evaluation

* **Corpus**： Events2012 corpus provided by McMinn. The corpus contains 120 million tweets from the 9th of October to the 7th of November, 2012. Along with this corpus 506 events, with relevant tweets for these events, are provided as a ground truth
* we have only reconfirmed the first three days (9th to 11th of October, 2012) of the ground truth events and manually selected a total of 41 events that belong to our selected time window
* We have used the LSH scheme proposed by Petrovic et al.  as our baseline
* TwitterNews achieved a recall of 0.87 by identifying 27 events out of the 31 ground truth events. The precision is calculated as a fraction of the 100 randomly chosen events that are related to realistic events.
        

### 思考

* 和很多其他论文一样，本文将事件提取分为两个阶段，第一阶段进行事件的发现，第二阶段进行事件聚类
* 文章说用ti-idf计算相似度开销大，使用RI向量加快计算速度，因为这个向量是定长的，词向量也有这个特征，可使用矩阵乘法进行相似度计算，不过矩阵的维护和更新也会影响计算速度，
* 每个事件只选取一条推特，可能不具有代表性
* 作者在第一阶段选择计算**历史推特**和当前推特的相似度，超过一定阈值才进入下一阶段，历史推特维持一个固定值，这个方法的计算开销也很大，如果把独立的推特当成独立的事件，直接计算推特和事件的相似度，合并两个阶段，计算量小的同时也能达到这样的效果，不过这种方法的优点是会减少事件的数量，不是所有推特都形成事件，只有在一定的时间窗口内有多条推特讨论时才考虑进行事件的聚类，这对于**讨论较少**的事件和**间隔较长**的事件不友好，这和数据窗口的长度（需指定）以及数据的完整有很大的关系
* 通过LSH（Locality Sensitive Hashing）方法优化较少计算量的方法值得进一步调研
* 作者也将事件设置了一个有效期（8-15分钟），时间有点短，但是时间一长容易导致事件数太多
* 作者的评估方法存疑，31个事件中识别出16个事件则召回率为0.52，认为识别出一个事件的标准是什么？和ground truth有多少重叠认为是同一事件？准确率也是根据随机挑选的100个事件来计算的，并不合理

---

## [Streaming First Story Detection with application to Twitter](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.170.9438&rep=rep1&type=pdf)(NAACL 2010)

### Introduction

* the goal of FSD is to identify the first story to discuss a particular event
* The traditional approach to FSD, where each new story is compared to all, or a constantly growing subset, of previously seen stories, does not scale to the Twitter streaming setting
* Allan et al. (2000) report that this distance（cosine） outperforms the KL divergence, weighted sum, and language models as distance functions on the first story detection task.
* ![](https://github.com/qiuxingfa/picture_/blob/master/2019/b0f3cacad9f3fb7003e3de2de0afd82.png)
* In the context of first story detection, this means we are not allowed to store all of the previous data in main memory nor compare the new document to all the documents returned by LSH.

### Experiments

* TDT5
* We compare the systems on the English part of the TDT5 dataset, consisting of 221, 306 documents from a time period spanning April 2003 to September 2003.
* Data was collected through Twitter’s **streaming API**. Our corpus consists of 163.5 million timestamped tweets, totalling over 2 billion tokens
* we employed two human experts to manually label all the tweets returned by our system as either Event, Neutral, or Spam.we only labeled the 1000 fastest growing threads from June 2009

### 思考

* FSD需比较当前文档和历史文档的相似度，当最小距离大于阈值时，则认为是新事件，其实这也是事件聚类的一部分，可以分开做也可以一起做。
* 数据不可能是无限的，所以论文在很多地方设置了上限，如历史推特的比较数，
* LSH的基本思想是：将原始数据空间中的两个相邻数据点通过相同的映射或投影变换（projection）后，这两个数据点在新的数据空间中仍然相邻的概率很大，而不相邻的数据点被映射到同一个桶的概率很小。

---

## [ELD: Event TimeLine Detection - A Participant-Based Approach to Tracking Events](http://delivery.acm.org/10.1145/3350000/3344921/p267-mamo.pdf?ip=65.49.38.140&id=3344921&acc=OPEN&key=4D4702B0C3E38B35%2E4D4702B0C3E38B35%2E4D4702B0C3E38B35%2E6D218144511F3437&__acm__=1571052520_5df392b95790ecb98b40e864b95da3ad)(HT 2019)

* ELD’s inputs do not only include a seed set of keywords that describe the event , but also the lengths of two time windows. In the **first time window**, ELD uses the user’s seed set to collect tweets that shape its understanding of the event. In the **second time window**, ELD uses this understanding to follow the event and build its timeline
* we use this dataset to build Reed et al.’s Term Frequency- Inverse Corpus Frequency (**TF-ICF**) instead
* FIRE adopts a traditional incremental clustering algorithm as the　**document-pivot** approach. Incoming tweets join the most similar　cluster if the similarity exceeds a threshold; otherwise, they form　a new cluster.

### 思考

* 在第一个时间窗口，ELD使用事件的种子集理解事件，在第二个时间窗口，ELD使用这个理解去追踪和建立事件的时间线
* Automatic Participant Detection 实际上类似于新事件发现
* 作者没有用TF-IDF，而是用的TF-ICF

---

## [BURSTY TOPIC DETECTION IN TWITTER](http://dspace.dtu.ac.in:8080/jspui/bitstream/repository/16327/1/SHIVANI%202K16SWE14%20Thesis%20report.pdf)（Master's Thesis 2018）

### INTRODUCTION

* How to proficiently keep up appropriate insights to trigger identification.
* How we can show bursty points without having the opportunity to inspect the whole arrangement of important tweets as the case in conventional point demonstrating.
* How to analyse those detected bursty topics for their polarity.

### LITERATURE REVIEW

* Clustering-Based vs. Topic-Modelling Based
* Retrospective vs. Online or Real-time

### Research Methodology

### Experimental Setup

---

## 《机器学习中的不平衡学习方法》

### 不平衡学习策略

* 重采样策略
* 代价敏感学习
* 单类别学习
* 集成学习

---

## [StoryMiner: An Automated and Scalable Framework for Story Analysis and Detection from Social Media](https://escholarship.org/content/qt9637m3j1/qt9637m3j1.pdf?t=pwzuo0)(Dissertations 2019)

### ABSTRACT OF THE DISSERTATION

* StoryMiner derives stories and narrative structures by automatically 
  * extracting and co-referencing the actants (entities such as people and objects) and their relationships from the text by proposing an Open **Information Extraction** system, 
  * assigning named-entity types and importance scores for entities and relationships using character-level neural language architectures and other traditional machine learning models, 
  * making use of context-dependent word embeddings to aggregate actant-relationships and form contextual story graphs in which the nodes are the actants and the edges are their relationships, and 
  * enriching the story graphs with additional layers of information such as sentiments or sequence orders of relationships.

### Introduction

* the main theme of this research is to introduce a narrative framework which is capable of identifying and representing narrative structures and contents from a large corpus of text

<br>

* StoryMiner 是一个将文本数据转为story graph的一个自动的事件检测框架，主要包括关系抽取（StoryMiner RelEx），主体抽取和故事图的构建，

---

## [Adaptive Multi-Attention Network Incorporating Answer Information for Duplicate Question Detection](http://delivery.acm.org/10.1145/3340000/3331228/p95-liang.pdf?ip=202.120.224.53&id=3331228&acc=ACTIVE%20SERVICE&key=BF85BBA5741FDC6E%2E88014DC677A1F2C3%2E4D4702B0C3E38B35%2E4D4702B0C3E38B35&__acm__=1571709823_661dc7afae46acce82bc80b96e3b56ab#URLTOKEN#)（SIGIR 2019）

### ABSTRACT

* we propose an answer information-enhanced adaptive multiattention network to detect duplicate questions
* To obtain multi-level textual features, we use the concatenation of word embedding, character embedding, and syntactical features as the representation
* we utilize three heterogeneous attention mechanisms: **self-attention**, which facilitates modeling of the temporal interaction in a long sentence; **cross attention**, which captures the relevance between questions and the relevance between answers; and **adaptive co-attention** which extracts valuable knowledge from the answers.

### EXPERIMENT

* datasets：In addition to CQADupStack, we expand the Quora Question Pairs (QQP) dataset with the paired answers and named it the answerenhanced QQP (AeQQP).

### 思考

* 论文主要解决的是问句的语义匹配问题，加入了答案的信息，为降低负面影响，引入了几种注意力机制，作为一个二分类问题
* 去除注意力机制之后效果下降，表示注意力机制的有效性
* 微博正文也可考虑评论或转发信息，但对于及时性运算速度，影响很大

---

## [Pairwise Word Interaction Modeling with Deep Neural Networks for Semantic Similarity Measurement](https://www.aclweb.org/anthology/N16-1108.pdf)（NAACL 2016）

* Most previous work use sentence modeling with a “Siamese” structure

### Model Overview

* Bidirectional Long Short-Term Memory Networks (**Bi-LSTMs**) are used for context modeling of input sentences, which serves as the basis for all following components.
* A novel **pairwise word interaction modeling** technique encourages direct comparisons between word contexts across sentences.
* A novel **similarity focus layer** helps the model identify important pairwise word interactions across sentences.
* A 19-layer deep convolutional neural network (**ConvNet**) converts the similarity measurement problem into a pattern recognition problem for final classification.
* this is the first neural network model, a novel hybrid architecture **combining** Bi-LSTMs and a deep ConvNet, that uses a **similarity focus mechanism** with selective attention to important pairwise word interactions for the STS problem.

---

## [Semi-supervised Question Retrieval with Gated Convolutions](https://arxiv.org/pdf/1512.05726.pdf)（NAACL 2016）

### Abstract

* The task is difficult since 1) key pieces of information are often buried in extraneous details in the question body and 2) available annotations on similar questions are scarce and fragmented.
* Several factors make the problem difficult. 
  * First, submitted questions are often long and contain extraneous information irrelevant to the main question being asked.
  * The second challenge arises from the noisy annotations.

### 思考

* 作者在问句匹配任务中使用一正一负的句子对作为训练数据

---

## [Feature Driven Learning Framework for Cybersecurity Event Detection](https://asonamdata.com/ASONAM2019_Proceedings/pdf/papers/029_0196_083.pdf)（ASONAM 2019）

* 对于社交媒体的网络安全事件检测，之前的方法基本上是集中于无监督和弱监督的方法，这些方法在现实中效果不佳，存在特征稀疏、处理弱信号（在大量数据中的少量数据）能力不强，模型泛化能力不强等缺点，这篇论文提出了一种多任务学习的有监督模型
* 将不同种类的目标机构视为不同的任务
* 收集2014年1月至2016年12月的893个网络安全事件，将大量的推特数据分为训练集和测试集，用命名实体所属机构进行标注

* 标注方法存疑

---

## [Jointly Detecting and Extracting Social Events From Twitter Using Gated BiLSTM-CRF](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8866717&tag=1)（IEEE Access 2019）

### 简介

* 使用pipeline的方法会造成错误传播，许多方法经常基于手工特征和现成的NLP工具（不适用于短文本）
* 提出在用深度学习的方法在推特中联合检测和提取事件的方法，将事件发现任务认为是二分类任务，而元素提取任务当成序列标注问题，进行联合学习，使用BiLSTM-CRF的结构

### 相关工作

* 社交媒体的事件发现：可分为开放领域和特定领域，开放领域经常使用无监督聚类的方法，而特定领域经常使用有监督分类的方法，依赖于特征工程和现有的NLP工具
* 社交媒体的事件提取：用pipeline的方法进行事件的发现和提取
* 联合模型：Zhang联合发现、聚类和摘要

### 问题定义

* 判断一条给定的推特是否和感兴趣的事件(二分类)，相关然后提取推特中的事件元素
* 事件定义为在特定的时间和地点发生的包含一个及以上参与者的事情，由四元组（触发词，时间，地点，参与者）组成，其中触发词是必须的

### 方法

* 将事件检测定义为二分类问题，判断是否与社会事件相关
* ![](https://github.com/qiuxingfa/picture_/blob/master/2019/3830b4b6feddd7cd6d116d7945793dc.png)

### 实验

* 选择“社会动乱”事件做二分类，
* 在五个月的中文推特里根据某些关键词选出33094条推特，通过SimHash删除重复推特，去除明显与事件无关的推特，最后人工标注，剩余3000正例和3000负例
* 使用5w条新闻和收集的推特进行词向量训练

### 思考

* 依然是pipeline，共享了一部分参数
* 将事件发现简化为二分类是否相关的问题
* 标注的方式很大程度上决定了解决的问题，文中根据关键词的方法进行事件检测，解决的就是针对特定领域事件的推特进行判断，正负均衡，泛化性不高

---

## 《基于社会化标注的个性化推荐算法研究》

### 绪论

* 用户在社会化标注中往往倾向于对相似的资源添加类似的标签，因此，通过这些标签就可以找到相关联的资源，这在一定意义上形成了信息资源的分类法
* 社会化标签存在以下问题：1）标签的同义和多义问题；2）标签缺乏层次性；3）合成标签问题；4）标签没有标准的结构；5）基准的波动，即对上下位词的混合使用

### 相关研究进展

* 基于内容过滤的推荐：通过比较资源与资源之间、资源与用户之间兴趣之间的相似度来推荐信息
* 基于协同过滤的推荐：基于评分相似的最近邻居评分数据向目标用户进行推荐

---

## 《实时数据流的算法处理及其应用》

* 实时数据流具有连续、近似无限、时变、有序及快速流动等特性，且实时数据流中数据点的出现顺序、速率、时刻均不可控制。

### 实时数据流和聚类方法的背景

* 实时数据流的特点：数据量巨大；时序性；快速变化；潜在无限；高维性；存储限制；时间限制；单边扫描或有限次扫描
* 数据流聚类算法：聚类簇数事先未知；聚类形状任意；对孤立点的分析能力

### 基于衰减窗口与剪枝维度树的数据流聚类

* 由于内存的限制，为了有效地对实时数据流进行在线聚类，必须采用相关技术在内中维护一个反映数据流特征的概要数据结构，以最大限度地保留对聚类有用的实时数据流信息，按处理模式不同可分为两类，一类是基于界标模型的技术，一类是基于滑动窗口模型的技术

### 实时数据流动动态模式发现于跟踪方法

### 增量式聚类方法与网格划分策略

* 传统的聚类方法恰恰大多是一种基于全局比较的聚类，要求所有数据必须预先给定，再考察数据空间中的所有数据才能决定类的划分

### 基于网格和密度维度树的增量聚类算法IGDStream

### 基于密度维度树的增量式网格聚类算法IGDDT

* 基于网格和密度的聚类算法存在的问题：网格单元粗细与聚类性能之间的矛盾；网格单元密度的计算；

### 基于距离和密度的实时数据流聚类及其边界检测技术的研究

---

## [Kernel-Based Distance Metric Learning for Supervised k-Means Clustering](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8617698)（IEEE T Neur Net Lear 2019）

* k-means算法的核心是找到一个合适的相似度计算函数，可以参试从已有的聚类数据集中进行有监督的学习
* 半监督聚类基本上是基于某种约束，如a和b属于同一类等等
* 学习一个距离计算方法来改善聚类效果，一个通常的假设是训练集和测试集共用距离计算方式
* 使用PUR、NMI和RI来评估，还考虑了运行时间

---

## [A Deep Relevance Matching Model for Ad-hoc Retrieval](http://www.bigdatalab.ac.cn/~gjf/papers/2016/CIKM2016a_guo.pdf)(CIKM2016)

* 针对专门的检索任务，通常这一任务会被转化为一个两个文档的匹配问题，大多数NLP任务为语义匹配，而这一任务为相关性匹配
* 手工的特征是耗时的、不完整的和过于具体的，深度神经网络作为一种表示学习的方法，能够从训练集中发现隐藏的结构和特征
* 深度匹配模型可分为两种，
  * 一种是representation-focused的模型，尝试用深度神经网络对单独的文本建立好的表示方法，是一种对称的结构
  * 另一种是interaction-focused的模型，首先基于简单的表示方法建立局部的交互信息，然后使用深度神经网络学习匹配的层级的交互模式
    ![](https://github.com/qiuxingfa/picture_/blob/master/2019/cd9ccfc2cdbf95199535e0cf1d176be.png)
* 语义匹配和相关匹配
  * 语义匹配的输入文本基本上是对齐的，语义匹配强调 相似度匹配，语言结构，整体匹配 三个因素
  * 相关匹配主要是关于判断文档和查询是否相关的任务，查询语句一般是简短的基于关键词的，而文档的长度则不确定，所以相关匹配强调 准确匹配，查询词的重要性，不同的匹配需求 三个因素
* 结构
  ![](https://github.com/qiuxingfa/picture_/blob/master/2019/647e571beaaba0bd07cd7b699de3a00.png)

* 语义匹配与位置无关
* 将输入词的相似度的直方图分布作为输入
* representation-focused的模型效果比传统的方法效果要差
  <br>

* 可以参考匹配的思路进行事件聚类

---

## [Text Matching as Image Recognition](https://arxiv.org/pdf/1602.06359.pdf)(AAAI 2016)

* 用图像识别的任务类比文本匹配的任务
* ![](https://github.com/qiuxingfa/picture_/blob/master/2019/df5fffce73101cc14fb194814d98dab.png)
* 有词级别，n-gram级别和句子级别的直接匹配和语义匹配等不同层级的匹配信号
* 使用动态池化策略来解决文本长度不一致的问题
* 在释义识别和论文引用匹配两个数据集上测试
  <br>
* 使用匹配信号而不是文本作为输入，使得模型更加灵活

---

## 《强化学习精要》

### 1 引言

* 强化学习关注的两个目标：学习效果和学习时间
* 监督学习的目标更明确，输入对应的是确定的输出，而且理论上一个输入只对应一个输出；而强化学习的目标没有这么明确，使当前状态获得最大回报的行动可能有很多。
* 强化学习更看重行动序列带来的整体回报，而不是单步行动的一致性
* **模仿学习**，存在以下问题：1）如何收集满足目标的样本；2）如何收集大量的样本；3）如何确保学习的模型拥有足够的泛化性

### 2 数学与机器学习基础

* 交叉熵损失只和正确分类的预测结果相关，而平方损失的梯度还和错误分类有关

### 3 优化算法

* 动量算法

### 4 TensorFlow入门

### 5 Gym与Baseline

### 6 强化学习基本算法

### 7 Q-Learning 基础

---

## [APPLYING DEEP LEARNING TO ANSWER SELECTION: A STUDY AND AN OPEN TASK](https://arxiv.org/pdf/1508.01585.pdf)(ASRU 2015)

* 面向QA问题，在所有候选问题中选出分数最高的一个，候选数为500
* 面向保险行业的QA
* 大部分模型的思路都是一致的：学习一种问句和候选答案的分布式向量表示方法，然后使用一种相似性度量方法来计算匹配程度
* 第一个模型是词袋模型，先训练词向量，带idf权重的词向量之和来表示句子，计算余弦相似度
* 第二个模型是weighted dependency model，即metzler-bendersky IR model
* 第三个模型是CNN模型
* 使用　hinge loss
* ![](https://github.com/qiuxingfa/picture_/blob/master/2019/0d8b56ecad93292be48a13e989bf49d.png)
* ![](https://github.com/qiuxingfa/picture_/blob/master/2019/01df076becb3cbbdcc5be04a401d631.png)
* 除了使用余弦相似度，还使用了GESD和AESD等相似度计算方法


* 和事件匹配的思路很像
* 1：500的比例不平衡

---

## [Event Detection in Noisy Streaming Data with Combination of Corroborative and Probabilistic Sources](https://arxiv.org/pdf/1911.09281.pdf)(arXiv 2019)

* 使用带有大量噪音的社交媒体文本，不提前假设概念的漂移，不依赖于人工标注
* 使用联合分类器
* 概念漂移问题，有的是用滑动窗口的方法实现，SAM-KNN对新数据选择最近的窗口，
* 结合了Corroborative source 和 Probabilistic supporting source ，前者是是确信的但延迟的，通过其调整分类器
* 关注于自然灾害，从相关网站和新闻获取 Corroborative source
* 从推特和脸书等社交网络获取 Probabilistic supporting source
* 假设：一旦在确信数据源中某个事件被检测出来，则同一时空背景下的流式数据自动认为相关
* 对于确信数据源的分类：标签来源于数据本身，如NOAA预测某地有滑坡的危险，则把高概率的预测作为ground truth
* 使用词向量的余弦相似作为距离度量，通过评估当前窗口和之前窗口的距离来检测漂移，如果发生漂移则成立新的窗口，如果没有则挑出k个最近的分类器
* ![](https://github.com/qiuxingfa/picture_/blob/master/2019/be887ef6eb294ecfc84b6b742ce4f4a.png)
  <br>

* 分类器发生漂移就新建分类器的方法会使得事件的正常波动和变化无法被检测，事件倾向于分裂
* 系统具体的算法和结构不清楚

---

## [Experiments with Convolutional Neural Network Models for Answer Selection](https://cs.uwaterloo.ca/~jimmylin/publications/Rao_etal_SIGIR2017.pdf)(SIGIR 2017)

* ![](https://github.com/qiuxingfa/picture_/blob/master/2019/208b829cd63e397382fbbb9c9d454d7.png)

---

## [Convolutional Neural Network Architectures for Matching Natural Language Sentences](https://www.researchgate.net/profile/Qingcai_Chen/publication/273471942_Convolutional_Neural_Network_Architectures_for_Matching_Natural_Language_Sentences/links/5614641708ae983c1b4083af/Convolutional-Neural-Network-Architectures-for-Matching-Natural-Language-Sentences.pdf)(NIPS 2014)

* 通过卷积和池化操作提取特定长度的文本表示，对于不同长度的文本采取补零的方法，在卷积过程中，设置一个gate，当输入全为0时，输出全为0
* 不同于一般学习到句子的表示向量后做内积，本文在文本对的匹配空间建立深度网络结构
* ![](https://github.com/qiuxingfa/picture_/blob/master/2019/af71fd6648429601a918dad78545732.png)
  <br>
* 如何保证输出向量维度

---

## [MIX: Multi-Channel Information Crossing for Text Matching](https://kopernio.com/viewer?doi=10.1145/3219819.3219928&route=1)(KDD2018)

* 针对文本匹配任务提出一种多通道的卷积神经网络，并增加了注意力机制，不同粒度的相似度矩阵作为不同通道，有两种通道，语义信息通道（unigrams, bigrams and trigrams）作为相似度计算，结构信息通道（POS，NER）作为注意力机制
* 在QQ浏览器实施A/B test，并在WiKiQA和在QQ浏览器上得到的搜索结果数据
* 文本匹配大致上有两种：基于表示的（CNN based， RNN based， tree-based RNN methods）和基于交互的
* 用idf的乘积、POS和位置信息来初始化注意力权重
* 利用卷积来融合不同的通道
* ![](https://github.com/qiuxingfa/picture_/blob/master/2019/1a4d146693bdefd015980299b12a54e.png)
* 评估：归一化折损累计增益（NDCG），mean average precision (MAP)
* 与位置信息相比和POS信息并没有得到更好的结果,weight则得到了最好的结果

<br>

* Steve Curry won his first MVP in 2014 和 What year did Lebron James win his first
  MVP 的例子很像 北京暴雨和深圳暴雨，作者用注意力机制解决这个问题

---

## 《智能问答》

* 自然语言问题大致分为七类：事实类（基于知识图谱或文本生成）、是非类（适合知识图谱或常识知识库）、定义类（基于知识图谱、词典或文本）、列表类（基于知识图谱）、比较类、意见类和指导类（基于问答社区的<答案，问题>对）



---

## [What’s Happening Around the World? A Survey and Framework on Event Detection Techniques on Twitter](https://link.springer.com/content/pdf/10.1007%2Fs10723-019-09482-2.pdf)（J Grid Computing 2019）



* 事件的一些定义：
  * Things that happen
  * something that happens at a specific time and place with consequences
  * something significant that happens at specific time and place
  * a set of posts sharing the same topic and words within a short time
  * an event is a real-world occurrence e with a time period Te and a stream of Twitter messages discussing the event during the period Te
  * in the context of online social networks, (significant) event e is something that causes a large number of actions in the online social network
  * an event is a way of referring to an observable activity at a certain time and place that involves or affects a group of people in a social network
* Specified Event Detection（SED） 和 Unspecified Event Detection（UED）
* 大多数数据来自于推特API和爬虫，数据平均时间为六个月，平均数量为2600万，标准数据集的产生的方法：1）关注热门事件；2）人工标注一部分；3）聚类之后进行标注
* 特征提取：1）Keyword-Based；2）Twitter-Based；3）Location-Based；4）Language-Based
* 事件相关的信息有三种，事件的发展，事件本身，事件的影响
* 谣言即 未被核实或故意错误的信息，机器账号即由计算机程序控制的账号
* 对于Specified Event Detection，即预先指定事件
* **Document** Pivot and **Feature** Pivot
* 监督的方法基本上基于事件的静态假设
* ![](https://github.com/qiuxingfa/picture_/blob/master/2019/4ac79814040ee16e83968a249de915c.png)
* 推特文本包含很多噪音，缩写、错字等，文本的处理其实也是一个耗费计算资源的过程，标准数据集的产生也是一个大问题，不同事件在热度、参与用户和文本数都不一样，多语言，多模态

---



## [Building a Large-scale Corpus for Evaluating Event Detection on Twitter](https://kopernio.com/viewer?doi=10.1145/2505515.2505695&route=1)(CIKM 2013)

* 产生了包含28天，从2012-10-10到2012-11-07的120m的推特的语料，包含美国总统选举，飓风Sandy等事件，去除了非英语推特，去除了一些垃圾信息，关联性判断的语料150k推特和500个事件，去除了转发，提供用户id和推特id，使用工具预聚类和Wikipedia挑选事件，
* 对于事件的定义还不一致，导致不同方法的比较相对困难，如有些人把选举当成一个事件，但有些人把选举当成多个事件
  * something that happens at some specic time and place, and the unavoidable consequences
  * (1) an associated time period Te and (2) a time-ordered stream of Twitter messages, of substantial volume, discussing the occurrence and published during time Te
  * a burst in the usage of a related group of terms
  * something that happens at some specic time and place
  * Something is signicant if it may be discussed in the media
* [数据下载](http://mir.dcs.gla.ac.uk/collections/events2012/dataset/Events2012.tar.bz2)
* TREC包含2周16M推特，包含多种语言，其中英文推特约4M，为ad-hoc retrieval任务所设计，对于基于事件的分析并不适用
* 认为对于事件的不真实的描述为不相关，而只是讨论并没有叙述的推特认为是相关

---



## [Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks](https://arxiv.org/pdf/1908.10084.pdf)（arXiv 2019）

* 使用BERT做句子对的相似度问题计算量大，1万个句子有约5000万种组合，在V100上inference需耗费大约65小时，对于有大量组合可能的任务不太适合，作者提出使用siamese and triplet结构的BERT以减少计算，计算时间减少到5秒
* 大部分句子表示方法为输出取平均或者选择[CLS]作为句子表示，但这种表示方法甚至比Glove向量平均更糟
* ![](https://github.com/qiuxingfa/picture_/blob/master/2019/66c108f8d55764d1115d18ab8e62dc9.png)
* ![](https://github.com/qiuxingfa/picture_/blob/master/2019/51517b3efb7f9b1ceca41e072be5291.png)
* 使用语义相似任务来做评估



---

## [TopicSketch: Real-time Bursty Topic Detection from Twitter](http://citeseerx.ist.psu.edu/viewdoc/download;jsessionid=632B3AEC809E16EBB6265E9BD3794CD9?doi=10.1.1.638.3816&rep=rep1&type=pdf)(IEEE T Knowl Data En 2016)

* 突发话题检测的难点在于：1）如何有效地保持合适的统计信息去触发detection；2）如何在无法像传统主题模型一样接触全部数据的情况下建模。部分工作提前定义了关键词。这是第一个在没有提前定义话题关键词的情况下进行实时突发话题检测的工作。
* 相关技术包括Offline，Online（计算量大）和Realtime（仍需提前指定）的方法
* 突发话题需是有一个突然的相关微博总数的增加表示热度，相关话题必须合理
* 三个主要的挑战：1）如何识别出突发话题（比如，话题的关键词是什么）；2）如何尽早地检测到突发话题（加速度）；3）如何在大范围的实时数据下有效地实施（维度约简）



---

## [Multiresolution Graph Attention Networks for Relevance Matching](https://arxiv.org/pdf/1902.10580.pdf)(CIKM 2018)

* 深度学习网络对于短-长文本对的相关性匹配效果并不好
* 使用图的方法来表示，顶点表示关键词（抽取命名实体，并用tf-idf的方法扩展挑选前20%），边表示关键词之间的相互作用程度（若词距离小于阈值，则建立联系权重为词距离的倒数）

* ![](https://github.com/qiuxingfa/picture_/blob/master/2019/3dbcaec687810092ce0b781826c0f66.png)

* 多层图卷积提供了多分辨率的节点表示
* 无监督方法效果不佳，基于特征的方法耗时
* 相关匹配更关注于是否相关而不关注语义是否相同，‘A man is playing basketball’ 与 ‘A man is playing football’ 是不相关的，相似匹配任务更注重于匹配信号
* 将 relevance matching 转化为 query-graph matching
* 挑选k个关键词进一步处理

___

## [A rule dynamics approach to event detection in Twitter with its application to sports and politics](https://pdf.sciencedirectassets.com/271506/1-s2.0-S0957417416X00061/1-s2.0-S0957417416300598/main.pdf?X-Amz-Security-Token=IQoJb3JpZ2luX2VjEAQaCXVzLWVhc3QtMSJIMEYCIQCF3x69XKi45SNJK7tGiAR2d6uAfV%2B0gQKqV86%2Fh4twWwIhAOBwIMbwMNgHlHDvqxlEPt2Kh8GsSmGjYllo2fw52FmTKtACCFwQAhoMMDU5MDAzNTQ2ODY1IgzaqhP%2FaXtjyIaiX%2FAqrQKtLB6az1qCC6r9v1gPCZk6kxD8GRjTrmaL3Q2V%2BlgSegLHEA86Y2JiTMkBvYwVHKrR8wGJaFoYX8brH6Y08%2BTBnaE41FH2udmRi7uNWNlxL8nEci6DnktjnicpeWbK4DV%2Bkp9UP3BO04EenayTf6FVEVuYDWFTyly8L7bk6KVatiNslO1PXy%2F6ph8lGdBS%2Fy8FyFqLryYGFe4flIejCwtNNkopHIP7TsEhkNWe4giJxLU8grzEa1uqoFztFkmGRwi%2F8uRegeK0sK2NmnQPKE4iTa6yuNtUE9O%2FjSpqNrn1DxBq7DE7jYhkIaI3VL%2F2n2IaK9BRDmn6MGGZipZ3LO%2FwDUjzJYK3MU09VeJKrIDdH7He%2BYKfYpesV4w8A0ZcT4BX9f5vuCDkrGb4k0COMIut2O8FOs4C%2B9Us3fXGixVv%2FGlp3PlFmUUxmvlAQIi8ZO04pmXKm%2FTx0J9QplFa4np%2BgNL9iYCJXJup0c%2FiM%2BA4qhSy5g3zQq%2FJlrd7chPx4txyBO%2Fu3dA8sfJEMC%2BK6WP%2Fwl3rcdNIg7wIcCkoR1FipNZJWeWjhcfBrA1YcCjhaUu3%2FRvBeQyS8IaAgX74Aun5lB8rXOFNc0YZSJt%2FB4qDrjp4pmiwjPiXRiyjIOSqOFJHQ2NcunxXM90EnmtiuBI4lJMIKPrP6U8VtLQ2zfDSCEVANLpzc1Uao%2BS4a5h0g653WyYcj987%2BKrXuyulUPLqbSS7Sj%2BCWP6beE1oeljQpbmxHUcqL%2F9gYhmtDfT0jOHYhGlAVS9PdfjKK72LgU6hxRFVsppZrSeq5z51p9LzVw6P9zwDOYgCUuhIyWpUPNWD0i3OHk24swHT3T3Wm1W3%2Fu0i0A%3D%3D&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Date=20191215T122122Z&X-Amz-SignedHeaders=host&X-Amz-Expires=300&X-Amz-Credential=ASIAQ3PHCVTYXQKDFVMK%2F20191215%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Signature=cda71e5e261856ac03e6e99b64db8b0c390523aee99ae6460be96f098216e065&hash=3f2e7d99520ee6ecf39328e77385980428ec54e9f3de26e1c6904d7d59f24766&host=68042c943591013ac2b2430a89b270f6af2c76d8dfd086a07176afe7c76c2c61&pii=S0957417416300598&tid=spdf-3a634fb1-dea8-44df-b96b-755584d089c4&sid=0fb4c4cb8067534e2d0ac5e5e4639da5897bgxrqa&type=client)(Expert Syst.Appl. 2016)

___

## [A SURVEY OF TECHNIQUES FOR EVENT DETECTION IN TWITTER](https://onlinelibrary.wiley.com/doi/pdf/10.1111/coin.12017)(Comput. Intell. 2015)

* 推特数据包含了大量噪音，the short length of tweets, the large number of spelling and grammatical errors, and the frequent use of informal and mixed language
* **A major challenge** facing event detection from Twitter streams is therefore to separate
  the mundane and polluted information from interesting real-world events
* **three major phases** ：
  * data preprocessing（filtering out stopwords and applying words stemming and tokenization techniques）
  * data representation（term vectors or bag of words）
  * data organization or clustering
* TDT主要分为两种：
  * retrospective event detection (RED) (iterative clustering algorithms)
  * new event detection(NED)(query-free retrieval tasks, incremental (greedy) algorithms)
* However, the TDT line of research **assumes** that all documents are relevant and contain some
  old or new events of interest. This assumption is clearly **violated** in Twitter data streams, where relevant events are buried in large amounts of noisy data.
* 对于UED，Sankaranarayanan et al. (2009) 先用朴素贝叶斯分类器判断是否新闻相关，再进行聚类（考虑时间信息），还使用了标签信息，还有一些方法先进行聚类，然后判断是否与现实世界相关
* partitioning clustering techniques such as K-means, K-median, and K-medoid or other approaches based on the expectation–maximization algorithm are also **not suitable** because they require a prior knowledge of the number of clusters (K)
* **Cosine similarity** is most commonly used within these online clustering algorithms to compute the distance between the (augmented) term vectors and the centers of clusters.
* 新事件检测类似于异常检测

---

## [A deep multiview learning framework for city event extraction from Twitter data streams](https://arxiv.org/pdf/1705.09975.pdf)(arXiv 2017)

* contributions
  * Automated real-time data collection wrappers for Twitter and city sensors;
  * A near real-time NLP component for classifying Twitter data;
  * A correlation analysis for detecting the dependencies between Twitter stream and city sensors and web driven data records;
  * A web interface for displaying and visualising the citys event highlights.

---

## [Event detection in Twitter stream using weighted dynamic heart beat graph approach](https://arxiv.org/pdf/1902.08522.pdf)(IEEE Computational Intelligence Magazine 2019)

* 关注于新事件的发现，一旦新事件被发现，随后便减少关注，WDHG对变化敏感
* 突发事件发现的方法忽略较小的事件
* ![](https://github.com/qiuxingfa/picture_/blob/master/2019/aa74ca77fe38681a2e556d5fd42f2ae.png)



---

## [A language-independent neural network for event detection]()(SCIENCE CHINA Information Sciences 2018)

* 之前的方法注重于大量的词汇和语法的特征，feature engineering is labor intensive and prone to error propagation
* Bi-LSTM+CNN的混合方法
* ![](https://github.com/qiuxingfa/picture_/blob/master/2019/b6141686d566bc4633bc155dfe9d70e.png)

---

## [Real-time event detection for online behavioral analysis of big social data](https://pdf.sciencedirectassets.com/271521/1-s2.0-S0167739X16X0009X/1-s2.0-S0167739X16300899/main.pdf?X-Amz-Security-Token=IQoJb3JpZ2luX2VjEAQaCXVzLWVhc3QtMSJHMEUCIQC2i0Wl3Qsu9LJjSFskveQLi02ft%2FwktjtwQCun8zgwegIgZBYTb%2FNlKPJdUYwx7y3GZ4xxlgMImaWrDjlDhkFn9MgqzwIIXBACGgwwNTkwMDM1NDY4NjUiDOxHKtSH3JSVqKjpnyqsAtm3Ph5mXf5nUSqYKb1wwax651KUC4jPOrF2sR7cP5UCaM%2FDmEkW2sfW7eodsdRapFcwI5QBuqWznkFyB%2FrMsgGP%2FGptNtwCYxFJvHLwHsfgdQyOF7MPbhreqXQtjkgzSS02SWjkm%2FneTjuS2NFpKp%2BWv07eL4pu3dtMmiq7luTm%2B4Ym5e0Yua0s%2FGInYpT%2FlMrwbmyzMdwDaRkIJ6Z9iCbmIpDO0VtxytkXWnkxWAD2GzhlSVoJ2yg75Q0HNhefiQuj2qpvmSLWJIdt4vBpasFClwvYieJxUBLc8f8mo690RUYFbqK%2BFclU9%2BBJnVmHheS%2Fv9hK9uUn2PxzFR9Jl7kqnXOZKUxArQOjeEsoTREX0V60WW6mQ7Po6wzAHTzppBmCUH1zK1eHLDEu7jDEqtjvBTrQAqisPHK0%2FrUfLD6H1CwguAEjBKGYcOPObml69pc%2BdM1JFWqV7uzaKbv97cCZRvmEor%2BkePoSaGpjnOMy8dE7je6xnDu4xDRfsI8CeIK2%2BqZDrA1T6S1K1CaNudxLEmH2XWGkT4JSUBTb9V5kZ1SqTIlVyYg0eT4J5ANMXSuH56n%2F8%2Fx8Qcihb%2BHyTXBuFqQYYUQINQOx67ogyLKlvo%2Bd0vfCMLTFWn%2Fxb40HihpheFd3FFObFXxlpyy2tig6bakv64GgxEWEckPZ7Eu3QYqYlg%2FJinhpCIlNWIrELj83ytdLSwNAiERspVpK%2BmXOnvZGT%2Bdh1hcziLrlRj5AH2Iejt5qwxHfmvgigXuR6Y1UJGyXt1u63PIhPto5Fu4HLo%2BRsRq%2Fh7bGqogzdElfh83AHTd%2BtcSbd8fZocMAWmcKxLQUzP%2FK2BVdWPKSu780n38%2Byg%3D%3D&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Date=20191215T123457Z&X-Amz-SignedHeaders=host&X-Amz-Expires=300&X-Amz-Credential=ASIAQ3PHCVTYSMXT755S%2F20191215%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Signature=4b6ab08fa682125c9b80e5eeba264bd49ec8cf44e6595ad19f6762808c6684be&hash=ca158e0851a4550b38782c2b98e7340b26263ee98ccd4656c0d54ddf50d600c2&host=68042c943591013ac2b2430a89b270f6af2c76d8dfd086a07176afe7c76c2c61&pii=S0167739X16300899&tid=spdf-1ff11b46-77f8-48b9-8578-41d8b3ecf644&sid=0fb4c4cb8067534e2d0ac5e5e4639da5897bgxrqa&type=client)(Futur. Gener.Comput. Syst. 2017)

---

## [INTWEEMS: A Framework for Incremental Clustering of Tweet Streams]([http://delivery.acm.org/10.1145/2850000/2843853/a87-khan_minhas.pdf?ip=202.120.224.53&id=2843853&acc=ACTIVE%20SERVICE&key=BF85BBA5741FDC6E%2E88014DC677A1F2C3%2E4D4702B0C3E38B35%2E4D4702B0C3E38B35&__acm__=1576413494_42f5149dd1783d5bffd267e8efe5fded](http://delivery.acm.org/10.1145/2850000/2843853/a87-khan_minhas.pdf?ip=202.120.224.53&id=2843853&acc=ACTIVE SERVICE&key=BF85BBA5741FDC6E.88014DC677A1F2C3.4D4702B0C3E38B35.4D4702B0C3E38B35&__acm__=1576413494_42f5149dd1783d5bffd267e8efe5fded))(iiWAS 2015)

* INTWEEMS uses cosine similarity for textual features and L1 distance for metadata based features

---

##  [Real-time event detection from the Twitter data stream using theTwitterNews+ Framework](https://reader.elsevier.com/reader/sd/pii/S0306457317305447?token=7E3DF044B3D935C12F4DB53B0546F929CDAC0FE5A7B1403D490EACFA7041FA7266AA723D0179419F2705442A5FD98F05)(Information Processing and Management 2018)

* 前一篇文章 [TwitterNews: Real time event detection from the Twitter data stream](https://www.researchgate.net/profile/Mahmud_Hasan8/publication/309426330_TwitterNews_Real_time_event_detection_from_the_Twitter_data_stream/links/581a12b008aeffb294130fd1/TwitterNews-Real-time-event-detection-from-the-Twitter-data-stream.pdf?_sg%5B0%5D=TnznRnCrOp6ZZclRyRwcqEB4IIRkbOvPhDGOkLY403iD2TAh87WNHmQ3YgnlhW8H_kxDJ4o4zW7AvOJIUF_s6Q.rvivN9t4tv9QsIZ-bRsBICXS9YUkxh3FnhWYdRYuE0aqJQP9dlN752umgbo-SgQC6Or5VYV4nH1rhlxVQ0vbrQ&_sg%5B1%5D=W7u992XqavOYSoexjseJ7Q97t4XvzkCJC-9DGowiktOVZ4moje8qyeerx2nuRO5dB9wPyEYIS7EGMEmtYDwb_bLE83W_ZeQ7AW1GznnRG27r.rvivN9t4tv9QsIZ-bRsBICXS9YUkxh3FnhWYdRYuE0aqJQP9dlN752umgbo-SgQC6Or5VYV4nH1rhlxVQ0vbrQ&_iepl=)（PeerJ Preprints 2016）
* provide a **low computational cost** solution to detect both major and minor newsworthy events in real-time from the Twitter data stream，The first stage
* event detection 的三种方法 (a) term-interestingness（**computation intensive**）, (b) topic-modeling, and (c) incremental-clustering
* McMinn and Jose (2015) 以实体为基础，计算与新推特有最近距离的有同一实体的推特相似度，选择相似度最高的推特所属的簇，否则建立新簇
* 增量聚类方法倾向于分裂，往往难以分辨发生在同一时间的相似事件
* 1)预处理:通过关键词筛选过滤无关的推特; 2)后处理:进一步筛选有价值的事件
* 使用 Euclidean dot product 计算cos相似
* 选择Event2012前三天的数据作为评估，包括31个事件,17M推特,  结合人工进行评估
* 平均处理1336条推特每秒
* 数据和评估方法不清,难复现



---

## [Cluster-discovery of Twitter messages for event detection and trending](https://pdf.sciencedirectassets.com/280179/1-s2.0-S1877750314X00074/1-s2.0-S1877750314001604/main.pdf?X-Amz-Security-Token=IQoJb3JpZ2luX2VjEAQaCXVzLWVhc3QtMSJIMEYCIQCX2IsLJZpliVcQTwAG47vEWVRqhqFZA1vtbfLU6pF84wIhAMentCwGvr9BcOd8T5pMqigUjhOfBS9MeiVp0BxS1u45KtACCF0QAhoMMDU5MDAzNTQ2ODY1IgxFDFMNFNuxUA4tgjcqrQIsj2yNTj5qM0E2Zs9dXiymxKk9taqUz9LOxJR%2FNc22x2fcrMvWNPfoyNLV0nA3MxVgBRH73oDISoMwRwhjhJa%2Fmb0BKQsQsl2Gnn1di1h8zimq0kHISPQcJZbzXFrgLhrfekb2pX1zJvJbdwraLFatVLxH67UypRCR9VFxF3rxXqQWSnuvRPxqlJ3h1bJKl9hLOqG0v2DRiBWf%2BVHC8uyvg1H09A97trkJCbqPctanafBuj5z5ulS8ta9HDIW1UBJfEyHEVmWIwVnwh5QfuhHskM2DYnBjQtD07cJSLNAGplaUeZLgui%2BrVKBXJDCa2i7EM2yExrHikbRFx7cyBy6SQZzomYtxWmFu%2BN5%2Bt0qgjjniUfV4DteDPXgOMkNUUz8AfEmEgGxWEmG0b%2B6DMMKy2O8FOs4CEo707Bn2Y6AIWAAFIGGbpEzIu86451EQ%2BKS8sBzHwIXOyPHomp3oNhJc00qHM0ReTHKO%2BU2WsDLbzQbOZA4acQX%2Bq1WiYtW7xGpMUXmMbQ9XiK2TC4xJJrkp1kzgPPHrNJdeziUDQ0X98NvQIJDXVGuc6OQsMAHeuNIxUeROMXnBXWDXlC8sqiUjdg6ZVUQq3F0IEHnDGtoWEmlmCFzWNDylf7JOsyi3eTj2h%2FvyHXe%2B6wuomC3BVenVfhfB42jGyb6g6P0NE1pPf5emQPoW69MvSZqFs5niEnJFe0bMD7xhPuiNpB9Ihy5wSlcSTK2OlvvkzUSt02jxnQyDxuh%2BueJ7EB1ONUo0S3yhsCFjdCCMItU9OdmlrC0G7DDf%2BrY8PwF58Mqus4mqEt3BQ0ZHkllwdPba0Qqk9hAMyevz9W6biBurMGpbOGk79vkfuw%3D%3D&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Date=20191215T123047Z&X-Amz-SignedHeaders=host&X-Amz-Expires=300&X-Amz-Credential=ASIAQ3PHCVTY4BKHVWHY%2F20191215%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Signature=d9e044c8578137a719bff4ca282ce883959226225e1b3ed2caf33a4e96384a6f&hash=b96be2d2c2d579c9b2cc8c9cf0ee7d26de23c505476efc99208c8711b16a880c&host=68042c943591013ac2b2430a89b270f6af2c76d8dfd086a07176afe7c76c2c61&pii=S1877750314001604&tid=spdf-15b230ad-d312-4bc4-a724-54af4baf2fcb&sid=0fb4c4cb8067534e2d0ac5e5e4639da5897bgxrqa&type=client)(J. Comput. Sci. 2015)



---

## [Fine-grained Event Categorization with Heterogeneous Graph Convolutional Networks](https://arxiv.org/pdf/1906.04580.pdf)(arXiv 2019)

* **Pairwise Popularity Graph Convolutional Network** (PP-GCN) based fine-grained social event categorization model
* build a **weighted adjacent matrix** as input to the PP-GCN model
* modeling social events is very complicated and ambiguous,
* we **first** present event instance (shown in short text message) as hyper-edge in an **HIN**,
* 利用NLP工具抽取构建，实体、同义词、话题和用户之间的关系，每个事件都是某些关系的总和是HIN的子图
* ![](https://github.com/qiuxingfa/picture_/blob/master/2019/ed45b8a778d65aa260f810d60006d43.png)
* 根据事件之间的路径来计算事件之间的相似度 KIES
* 用Doc2vec作为 event instance（即文档）特征
* 以正负均衡来挑选pair，但实际上负例远多于正例
* 使用半监督k-means方法进行聚类比较，学习到一个合适的相似度计算方法

---

## [CATI: An Active Learning System for Event Detection on Mibroblogs’ Large Datasets](https://pdfs.semanticscholar.org/bbc7/59fc00fa31020df86802246eba9d52d13ff9.pdf)（WEBIST 2019）

* 主动学习，即机器学习结合人工标注



---

## [Hot Topic Detection on Twitter Data Streams with Incremental Clustering Using Named Entities and Central Centroids ](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8713730)(RIVF 2019)

* 在Event2012数据集中获得0.92的Recall和0.85的NMI，并且在30min内处理500k推特
* many **challenges**: volume and velocity, real-time event detection, noise and veracity, feature engineering, and evaluation
* 使用最相似的一些推特来计算形心代表簇
* a **high computational cost** is a major problem for social data streams
* 新来的推特只与有共同命名实体的聚类进行比较
* ![](https://github.com/qiuxingfa/picture_/blob/master/2019/9b465dfc604dd992162041967c5ad67.png)
* If similar tweets exist, the current tweet will be input to the Clustering module. Otherwise, it is skipped as noise. 对于新事件不利
* 一般中心点由聚类的所有推特计算，这种计算方法计算成本高，本文挑选最相似的L条推特作为代表



---

## [Event Detection and Evolution Based on Knowledge Base](http://www.cse.ust.hk/~yqsong/papers/2018-KBCOM-Event.pdf)(KBCOM 2018)

* “American president” was the most relavant to “Obama” in 2010s, but related to “Trump” recently
* we present a novel **7-tuple model** based on event attributes, including time, location, participants, keywords, emotion, summary and most-related posts
* 图方法在单实体事件表现不佳
* 同一时间窗口下（10min）构建图
* 搜集 Weibo,Wechat, Forum and News Media 的数据，随机挑选某天中一百万条数据



---

## [Real-time event detection using recurrent neural network in social sensors](https://journals.sagepub.com/doi/pdf/10.1177/1550147719856492)(International Journal of Distributed Sensor Networks 2019)

* 使用多通道词向量,用CNN来辨别有效信息,用LSTM来进行事件检测
* ![](https://github.com/qiuxingfa/picture_/blob/master/2019/5a0178d3a80e4d863994a136eb3baf2.png)
* 在地震数据集上实验,作为是否是地震相关事件的二分类任务
* 对于时序模型,事件定义为正常时间序列数据的异常值



---

## [Mining Event-Oriented Topics in Microblog Stream with Unsupervised Multi-View Hierarchical Embedding](https://dl.acm.org/doi/pdf/10.1145/3173044?download=true)(ACM Transactions on Knowledge Discovery from Data 2018)

* 两种话题的表示方法 : (1) word phrases and (2) multinomial distributions over word vocabulary
* we design an unsupervised translation-based hierarchical embedding framework, and all of its training examples (the patterns of topics and relations) are automatically produced by our Mv-BRT in advance
* **Definition** (**Event**). An event is a public and significant issue that happens at some specific time and place. Usually, event appears in the form of word phrases.
* **Definition** (**Event-Oriented Topic**). An event-oriented topic ETi is a multinomial distribution over word vocabulary, i.e., ETi = {p(wj ) |j = 1, 2, . . . ,V }, whereV is the size of the vocabulary, and this word distribution can be interpreted with an event word phrase EPi
* 用KL散度计算两个topic之间的相似度, 对于训练好的topic向量进行谱聚类
* We employ the translation-based knowledge base embedding method to fulfill this task. the symbolic relations used in our cases are **join, left-absorb, right-absorb, and collapse**, 用transR训练话题向量
* 通过某种方法估计聚类数量
* TREC Tweets2011, weibo 和twitter 三个数据集



---

## [Real-time Event Detection on Social Data Streams](https://arxiv.org/pdf/1907.11229.pdf)(2019 KDD)

* we **model** events as a list of clusters of trending entities over time, propose novel metrics for **measuring** clustering quality
* we model an **event** as a list of clusters of trending entities indexed in time order, also referred to as a cluster chain
* scale ; brevity ; noise ; dynamic
* 分为 online 和 offline 两个部分,online关注于低延时和可变化, offline 关注于高品质
* **contribution** : 1)Tracking of event evolution over time ; 2)Differentiated focus on **quality** of clustering; 3)Novel real-time system design
* 基于文本的方法还需额外的工作去总结事件,而**基于特征**的方法可以将实体列表视为事件表示
* topic detection ; Bursty terms ; incremental clustering (may not be feasible to use for the Twitter Firehose due to the scale of update)
* 方法
  * 首先进行初步筛选(去除噪音和重复信息), 对每条推特提取<entity,domain, 1>三元组, 然后根据实体在短期和长期出现次数之比来对实体的出现次数进行评估, 最后进行排序
  * 之前的工作基本是同步进行趋势的检测和聚类,本文将这两部分开异步执行
  * 提取 Named entities,Hashtags, Internal knowledge graph entities, 
  * We use these **frequencies and cooccurrences** to compute similarities between entities, 用实体在推特的出现次数向量来表示实体,计算实体之间的cos相似,这种方法使得数据非常稀疏, 通过过滤器筛选实体之间的噪音, 构造实体图,以相似度为权重, 用图分割进行聚类, 
  * 对于每一分钟的聚类结果,都尝试和前一分钟的结果进行连接,如没有找到连接,则新建事件id
  * ![](https://github.com/qiuxingfa/picture_/blob/master/2019/a804d710b8701b67c64f4d447b77372.png)
* 评估:
  * 一天的美国地区的英语推特
  * For each chain, we select 20 representative tweets (10 most retweeted and 10 random tweets)
  * 最后包括2695个实体和460个事件
  * 提出一种新的评估方法, 考虑事件内的实体关系, 类似P,R和F1
  * online performance: our system is able to scale and process millions of entities per minute



---

## [On the reliable detection of concept drift from streaming unlabeled data](https://arxiv.org/pdf/1704.00023.pdf)(Expert Syst 2017)

## [Event detection and popularity prediction in microblogging](https://pdf.sciencedirectassets.com/271597/1-s2.0-S0925231214X00251/1-s2.0-S0925231214010893/main.pdf?X-Amz-Security-Token=IQoJb3JpZ2luX2VjEAQaCXVzLWVhc3QtMSJIMEYCIQCF3x69XKi45SNJK7tGiAR2d6uAfV%2B0gQKqV86%2Fh4twWwIhAOBwIMbwMNgHlHDvqxlEPt2Kh8GsSmGjYllo2fw52FmTKtACCFwQAhoMMDU5MDAzNTQ2ODY1IgzaqhP%2FaXtjyIaiX%2FAqrQKtLB6az1qCC6r9v1gPCZk6kxD8GRjTrmaL3Q2V%2BlgSegLHEA86Y2JiTMkBvYwVHKrR8wGJaFoYX8brH6Y08%2BTBnaE41FH2udmRi7uNWNlxL8nEci6DnktjnicpeWbK4DV%2Bkp9UP3BO04EenayTf6FVEVuYDWFTyly8L7bk6KVatiNslO1PXy%2F6ph8lGdBS%2Fy8FyFqLryYGFe4flIejCwtNNkopHIP7TsEhkNWe4giJxLU8grzEa1uqoFztFkmGRwi%2F8uRegeK0sK2NmnQPKE4iTa6yuNtUE9O%2FjSpqNrn1DxBq7DE7jYhkIaI3VL%2F2n2IaK9BRDmn6MGGZipZ3LO%2FwDUjzJYK3MU09VeJKrIDdH7He%2BYKfYpesV4w8A0ZcT4BX9f5vuCDkrGb4k0COMIut2O8FOs4C%2B9Us3fXGixVv%2FGlp3PlFmUUxmvlAQIi8ZO04pmXKm%2FTx0J9QplFa4np%2BgNL9iYCJXJup0c%2FiM%2BA4qhSy5g3zQq%2FJlrd7chPx4txyBO%2Fu3dA8sfJEMC%2BK6WP%2Fwl3rcdNIg7wIcCkoR1FipNZJWeWjhcfBrA1YcCjhaUu3%2FRvBeQyS8IaAgX74Aun5lB8rXOFNc0YZSJt%2FB4qDrjp4pmiwjPiXRiyjIOSqOFJHQ2NcunxXM90EnmtiuBI4lJMIKPrP6U8VtLQ2zfDSCEVANLpzc1Uao%2BS4a5h0g653WyYcj987%2BKrXuyulUPLqbSS7Sj%2BCWP6beE1oeljQpbmxHUcqL%2F9gYhmtDfT0jOHYhGlAVS9PdfjKK72LgU6hxRFVsppZrSeq5z51p9LzVw6P9zwDOYgCUuhIyWpUPNWD0i3OHk24swHT3T3Wm1W3%2Fu0i0A%3D%3D&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Date=20191215T124419Z&X-Amz-SignedHeaders=host&X-Amz-Expires=300&X-Amz-Credential=ASIAQ3PHCVTYXQKDFVMK%2F20191215%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Signature=12030eb0e29edabc88067cad256c6551e3d6eb863f30066cac4f8aa5efb1a45f&hash=f3a114d45355d273d3324a1cac3c0c14099ec7dd65da2fc0113667de743ce585&host=68042c943591013ac2b2430a89b270f6af2c76d8dfd086a07176afe7c76c2c61&pii=S0925231214010893&tid=spdf-0d8ff8ef-4555-4045-b19b-97a5abfe7f4e&sid=0fb4c4cb8067534e2d0ac5e5e4639da5897bgxrqa&type=client)(Neurocomputing 2015)
