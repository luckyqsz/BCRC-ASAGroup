# TDT_Paper Reading
`topic detection and tracking` `single pass clustering` `topic` `event` `story` `topic tracking` `topic detection` `story detection` `event detection` `aspect tracking` `storyline extraction` `news story` `Online Detection` `data stream`

---

## [基于增量型聚类的自动话题检测研究](http://search.cnki.net/down/default.aspx?filename=RJXB201206018&dbcode=CJFD&year=2012&dflag=pdfdown)

* 话题检测与跟踪(TDT)的目标就是检测相关信息并跟踪事件的发展变化
* 回顾式的话题检测即检测已有新闻库中尚未发现的话题,在线的话题检测即在线的检测当前到达的新闻所属的话题.
* 定义话题为讨论一致的话题或概念的新闻集,两篇讨论不同特定问题的新闻文档可以属于同一个话题
* 有些文章利用话题的名词实体来提高话题辨别能力,但是这些方法对话题检测性能的提高是有限的
* 话题之间的相似性由两个话题中最相似的两个子话题的相似值决定,在聚类的过程中不断提炼基于话题的高辨别性特征词向量,利用该向量表示话题,通过计算BIC(Bayesian information criterion)来判断两个类能否被合并
* 特征的话题辨别能力.它表示一个特征对于区分话题的能力,当一个特征在一个话题出现比较频繁而在其他话题出现比较稀少的情况下,该特征具有较高的话题辨别能力（类似tf-idf）
* 新闻事件都有一个持续时间的特性
<br>

* 模型选择方法，计算复杂度太高，不适合海量数据
* 类似于从下到上的分层聚类方法
* 文章使用了分块预聚类的方法

---
## 话题检测与跟踪算法改进研究_侯晓冲

### 相关技术
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

### 系统实验测评
* 评价指标：召回率、准确率、F1值、漏检率、误报率、归一化系统开销

---
## [DiaeTT: A Diachronic and eryable Topic-Tracking Model](https://ears2019.github.io/Nakamura-EARS2019.pdf) （SIGIR）
* 词语的意思会随着时间变化，如“云”，结合词语语义的变化检测长时间的话题的变化，diachronic and
queryable topic tracking (DiaQueTT)
* 将词语语义的变化和TDT结合起来，使用带时间特征的词向量，(Linux cluster, 1995) and
(cloud, 2015) are close to each other, but the vector (cloud, 1995) is far from them.
* 实验数据为学术文章
* 分别对每年的数据训练w2v词向量，不同年份之间的词向量是不可比的，使用了转换矩阵，
* 正则化的词向量*TD-IDF之和表示文档
* 把单词和时间作为共同输入寻找最接近的文档
* 用一个vMF分布表示一个话题，用贝叶斯信息准则决定话题数量，找到话题后，按年分开，表示话题随时间变化的情况
* 话题一致性评估
<br>

* 大部分语义随时间变化不大
* 将不同年份的词向量映射到同一语义向量空间，用vMF进行文章话题聚类，每个话题内按年表示话题随时间的变化，对结果进行话题一致性评估

---
## [Automatic Evaluation of Topic Coherence](https://www.researchgate.net/profile/David_Newman6/publication/220817098_Automatic_Evaluation_of_Topic_Coherence/links/0deec51cd651fd8e71000000/Automatic-Evaluation-of-Topic-Coherence.pdf?_sg%5B0%5D=djzA9EBIX9oWCf7UxZDFiUx7nX6fQgYvXQsqZovKAOiVbaB4l8i9hjc4pkzHeHxykCIOCLs09Mg8OsE2Kt8Vmw.Pls630zK4lpOfXQrB_HoIlX06T1zA6A6TgNIEP8dMJ9BuBVfkGtMG789eAIZdwbIxvhr5VdlsO9IMOgeccZJbg&_sg%5B1%5D=uX9Y576Ytq2oYHiHy2bBLqZenM_By9UL24IGqLDoxf_dtp5_Q9npc5oCTECEoK-60uUkvEtShJjgNaS9Da7ryrvKS38fXfNS8BSVHL4TvyXJ.Pls630zK4lpOfXQrB_HoIlX06T1zA6A6TgNIEP8dMJ9BuBVfkGtMG789eAIZdwbIxvhr5VdlsO9IMOgeccZJbg&_iepl=)
* 主题一致性的内在语义评估，提出一种全自动评估方法，达到接近人类的准确度
* 用LDA产生话题，top-n
* 人工评估
<br>

* 主题模型主要是从文本中提取关键词

---
## [Temporal Event Detection Using Supervised Machine Learning Based Algorithm](https://link.springer.com/content/pdf/10.1007%2F978-3-030-16681-6.pdf)
* 提取事件信息和时间信息的关系，使用有监督的机器学习技术和基于规则的方法
* 事件提取：从文本中提取事件元素（ACTION，STATE，OCCURRENCE...）
* 事件信息和时间信息分开抽取，然后再判断事件和时间的关系，使用神经网络的方法进行训练

---
## [Research on Topic Detection and Tracking for Online News Texts](https://kopernio.com/viewer?doi=10.1109/access.2019.2914097&route=6)
* 用LDA提取话题，用吉布斯采样计算参数，用单遍遍历追踪话题，用JS (JensenShannon) divergence表示话题相似性，增加时间衰减函数
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
## [An Adaptive Topic Tracking Approach Based on Single-Pass Clustering with Sliding Time Window ](https://kopernio.com/viewer?doi=10.1109/ICCSNT.2011.6182201&route=6)

* 通常在不同时间点同一话题的分布是独立的，传统的关键词搜索方法只是因为包含特定的关键词就返回信息，会导致有很多冗余信息
* 预处理过程包括分词，特征选择（名词，动词和命名实体），权重计算（TF-IDF）
* 计算余弦相似度
* 计算故事和话题之间的相似度，通过阈值控制聚类的粒度并发现新的类别，单遍聚类的缺点是聚类结果受输入顺序的影响
* 设置两个阈值0.7和0.45，以向量平均值表示类别，处于两个阈值之间的故事将会在k个时间窗口（2天）内多次计算进行话题分配
* TDT-4 数据集
<br>

* 现实中的新闻数据就是有时间顺序的，我们也不可能一次性把所有数据输入

---
## [Automatic Online News Issue Construction in Web Environment](https://kopernio.com/viewer?doi=10.1145/1367497.1367560&route=6)
* 没有考虑话题重叠
* 故事平均值表示话题
* 分层聚类的方法，合并相似度最高的两个类
* 350篇文章87话题，953篇文章108个话题，24872篇新闻文章和1339篇博客论坛文章
* 使用P，R和F1作为评估
* 在数据集2中使用不同的聚类方法实验
* 去除冗余句子提升效果，考虑标题提升效果，只考虑短文本标题效果最好，
* 实时结果和日/周/月排行（彩票出现在月排行），对结果进行定性评估

---
## [An Online Topic Modeling Framework with Topics Automatically Labeled](https://www.aclweb.org/anthology/W19-3624) (ACL2019)
* 手动标注了507篇报道6个类别，总的7076篇报道
* 之前的话题追踪方法基本上基于LDA
* 投票数，浏览数和文章长度表示文章质量
* 用报道的话题分布作为输入训练SVM分类器
* 话题一致性评估

---
## [Tracking Aspects in News Documents](https://kopernio.com/viewer?doi=10.1109/ICDMW.2018.00165&route=6)
* 使用HMM跟踪话题，考虑故事之间的转换状态，维特比算法选择最佳路径，HMM根据序列数据产生MM，基于两个假设，每个话题基于统一的马尔可夫模型结构，按年代排列的文章作为一个方面在概率上是可区分的
* 没有通用的框架去对形势追踪建模，对于突发新闻容易追踪错误
* 使用带标签的新闻数据，手动挑选特定几个话题的文章打乱，按时间排序，提取了文章中的专有名词和动词，KL散度表示距离，
* 选取特定话题的数据进行实验

---
## [话题检测与跟踪的评测及研究综述](http://search.cnki.net/down/default.aspx?filename=MESS200706014&dbcode=CJFD&year=2007&dflag=pdfdown)
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
## [Hierarchical clustering based on single-pass for breaking topic detection and tracking](http://search.cnki.net/down/default.aspx?filename=GJST201804005&dbcode=CJFD&year=2018&dflag=pdfdown)
* 单遍聚类算法在面对海量数据时优势明显，简单有效但是计算复杂度高，提前进行数据分块（分类假设，假设在共同目录下或有相似特征的文章比较接近，假设时间接近的文章更有可能讨论同一话题），话题可能随着时间发生变化
* 文章由单词和词性组成，命名实体以及对应的TF-IDF权重表示文章向量，词袋模型表示性能不好，有些句子相似度很高但属于不同事件，分为关键特征向量（标题）和命名实体特征向量，关键词包含事件，名词，动词给予较高权重，一般的单词和短语给予较低的权重。改进的TF-IDF表示，给予不同的词性，是否位于标题权重
* 多层级多粒度策略适合海量数据
* 核心报道的选择（代表话题）：与话题相似度更高的，有用信息更多的（命名实体数），最新的；选择k篇文章代表话题，融合成一篇，并不断更新，长文本使用余弦距离，短文本则用基于HowNet的杰卡德相似度
* 淘汰过时话题，话题能量由文章和话题相似度以及时间总体作用
* 利用网站分类信息或用朴素贝叶斯分类器
* 三层聚类：
    * 基于单遍遍历和时间窗口（一天）的局部聚类
    * 分别在不同类别中基于局部聚类结果的凝聚聚类
    * 在总体不同类别基于KNN的凝聚聚类，K=1
* 聚类性能由新浪新闻数据420篇10个话题数据，时间复杂度评估使用283638篇数据，从2015年10月到12月

---
## [Analyzing the Changes in Online Community based on Topic Model and Self-Organizing Map](https://www.researchgate.net/publication/281537461_Analyzing_the_Changes_in_Online_Community_based_on_Topic_Model_and_Self-Organizing_Map/fulltext/5641a11708aec448fa611d6d/Analyzing-the-Changes-in-Online-Community-based-on-Topic-Model-and-Self-Organizing-Map.pdf?_sg%5B0%5D=nbYwnOBM7HJKQCi9pi_YK39-EW_6_TZVg9h-WKMOAWhbAlrcAFwKwSYO8kERs0F7roVckxIzSYckc3U7Vb6_Pg.hnCvgeDfvniNr-_cZVBmqvom-Irjpn0Sut5kxg4siDfjwWHpheXZFN-ujZgvQPqWVEnWMfwbPvS8_TX3LLb-Sg&_sg%5B1%5D=TbdqCCLyB0E7Ks6EWK9bRUMDKHPs4YMt8IhlMxJqQESUgSstupfaUrpEPEbnmgJn4k30EdC132SYxlGvhtmUevW8MypNzpo7MiOcTmkuwAFe.hnCvgeDfvniNr-_cZVBmqvom-Irjpn0Sut5kxg4siDfjwWHpheXZFN-ujZgvQPqWVEnWMfwbPvS8_TX3LLb-Sg&_iepl=)
* a **topic** is a mixture component defining a distribution of words
* 分析话题和用户随时间在社区中的变化
* Temporal – Author – Recipient – Topic model (TART)

---
## [A single pass algorithm for clustering evolving datastreams based on swarm intelligence](https://kopernio.com/viewer?doi=10.1007/s10618-011-0242-x&route=6)
* 分散的自底向上的自自组织策略
* 指定聚类数目，对于过时和最近数据使用相同权重，没有捕捉到数据的变化
* 分为online和offline两个部分，
* CluStream有两个缺点，一是不能发现任意形状的类别，二是类别数需提前指定，基于密度的聚类方法可以克服以上缺点（DenStream，D-Stream，MR-Stream），
* 权重随着时间的变化指数衰减，对每一个新数据，判断属于哪一种微聚类，core-micro-cluster，potential c-micro-cluster，outlier micro-cluster，DenStream不能处理大量的数据因为它对于每一个新数据都需要寻找最近的微聚类
* FlockStream,用并行的随机局部多主体搜索取代了全局搜索最近邻的方法，每个中介只与可见距离内的中介进行比较，数据：DS1，DS2，DS3

---
## [Story Disambiguation: Tracking Evolving News Stories across News and Social Streams](https://arxiv.org/pdf/1808.05906.pdf)
* 基于分类的方法需要大量的正负标注样本，对于随时间变化的故事并不敏感，只能用于一种文本（如新闻文章）
* 之前的大部分方法集中于一种文本
* 实际上，一个故事由几个关键的命名实体决定，所以实体图是一种有效表示故事的方法
* 使用了一种半监督的方法自动更新故事的实体图和特征向量，以反映故事随时间的变化，分类器不变而是对特征进行更新
* 监督的TDT方法基本上假定于一个静态的环境
* 将故事追踪任务当成L2R（learning-to-rank）任务
* 目标故事由用户进行初始化，用半监督的方法挑选对故事进行更新
* 选择维基百科作为外部知识，用NLTK进行命名实体识别，并用词性标注识别名词短语以扩展命名实体列表，用TAGME进行实体消歧，将短时间内带有相同标签的推特组合起来作为一篇文章，以实体id，实体位置和消歧置信分数组成文本表示，可变的时间窗口增加前面出现的实体的权重，以时间窗口中的共现表示实体间的关系
* 用NetworkX构建故事实体图，以及计算biased PageRank权重
* 使用RandomForest作为分类器，选择“2016爱尔兰大选”作为训练集，选择其他八个故事作为测试集，选择900篇相关文章，1800篇相关推特，以及27000篇无关文章，模拟数据流，产生了15个故事，训练29.7k×15个故事-文本对，每50篇文章加入故事更新一篇，
* 删除了标题的关键词（类似标签）和推特的标签，十倍的相同时间窗口内的负样本

---
## [Hashtagger+: Efficient High-CoverageSocial Tagging of Streaming News](https://kopernio.com/viewer?doi=10.1109/tkde.2017.2754253&route=7)
* 一个故事可以有多个标签，在故事的不同阶段可能会有不同的标签
* 以分类的思想（数据稀疏性以及噪音），以主题模型的思想（新标签），深度神经网络多分类（静态）
* 使用L2R的方法将排序问题转为分类问题
* MCC方法：使用有标签的推特作为训练集，对于概念的变化需要重新训练，每五分钟使用过去4小时的数据重新训练
* 标签的噪音很多，因为很多用户的文本和标签不一定相关，不能算很好的标注数据

---
## [STREAMCUBE: Hierarchical Spatio-temporal HashtagClustering for Event Exploration over the TwitterStream](https://kopernio.com/viewer?doi=10.1109/ICDE.2015.7113425&route=6)

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
## [Mining Text and Social Streams: A Review](https://kopernio.com/viewer?doi=10.1145/2641190.2641194&route=6)
* 初始化k个类，后来的数据和已有的类做相似度计算，超过阈值的归入当前聚类，否则作为异常值，取代最不活跃的类
* 突发特征表示，使用一些突发性特征优化聚类结果，
* 计算当前文档和之前k个文档的相似度

---
## [LEARNING GENERAL PURPOSE DISTRIBUTED SENTENCE REPRESENTATIONS VIA LARGE SCALE MULTITASK LEARNING](https://arxiv.org/pdf/1804.00079.pdf)
* 将不同的句子表示方式融合进一个多任务学习的框架

---
## [BB_twtr at SemEval-2017 Task 4: Twitter Sentiment Analysis with CNNs and LSTMs](https://www.aclweb.org/anthology/S17-2094)
* 使用100万推特预训练词向量
* 预训练的词向量包含的情感极性信息很少，正负极性的词距离可能很近（“好”和“坏”），再用远程监督方法进行微调，

---
## [Embeddings for Word Sense Disambiguation: An Evaluation Study](https://www.aclweb.org/anthology/P16-1085)

---
## [On clustering network-valued data](https://papers.nips.cc/paper/7282-on-clustering-network-valued-data.pdf)

---
## [Short Text Topic Modeling Techniques,Applications, and Performance A Survey](https://arxiv.org/pdf/1904.07695.pdf)
* 短文本的词共现信息十分有限，传统方法在短文本上出现性能退化
* 三种方法：Dirichlet multinomial mixture，global word co-occurrences，self-aggregation
* 传统的主题模型：PLSA和LDA
* 某些短文本也不一定单主题
* DMM假设每篇文章只属于一个潜在的话题
* 把迪利克雷分布作为先验分布

---
## [See What’s enBlogue – Real-time Emergent TopicIdentification in Social Media](https://kopernio.com/viewer?doi=10.1145/2247596.2247636&route=6)

* 提出问题：
    * 在大量的数据流中，新兴产生的话题，通常不是由一个标签表示，而是由很多不同的标签集合而成，通过检测单独的标签来检测新兴话题是不合适的
    * 对于文档的标签，可直接给出或者通过话题分类或命名实体识别给出，但是，这些标签通常不精确且只强调了一个方面，比如，“France”,“Cuba”, 或 “Vancouver”的话题
* 论文贡献
    * 提出enBlogue，用来检测emergent topics，使用滑动窗口来计算标签和标签对的统计数据，这些数据用来检测不同寻常的变化，
    * 对于文档的tag，可直接给出或者通过话题分类或命名实体识别给出
* 两个标签的关联性由标签的交集程度或杰卡德相似度定义
* 三个部分：种子标签的选择，关联追踪，变化检测

---
## [Event Detection and Retrieval on Social Media](https://arxiv.org/pdf/1807.03675.pdf)
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
## [Cluster Ensembles – A Knowledge Reuse Framework for Combining Multiple Partitions](https://www.researchgate.net/profile/Joydeep_Ghosh6/publication/220320167_Cluster_Ensembles_-_A_Knowledge_Reuse_Framework_for_Combining_Multiple_Partitions/links/00463521fe3b4643bd000000/Cluster-Ensembles-A-Knowledge-Reuse-Framework-for-Combining-Multiple-Partitions.pdf?_sg%5B0%5D=TqIQ6wGXEMXtwA_iADV5oXchPiZn5AyK3Qt_jZkHftEVM3EOGGLpCeUW9fG664gBHxNrqQOF617AMq5JS51WQw.NSr2fq4jPKQyUAaBQVDQs2icrf8-IcLf3JWDcIn_JL_Lq9MYFvRm5lNnTE9mid-M-rI7ebbnQNwQhX9LTEpnKg&_sg%5B1%5D=9Tb4aPkeCn7Tt8nXduAzEoiZ9ute1SvcxtZ5-NgUaEmvb9GWn7naF9x3tYXMgr6IA3AwE2-vpR3iYm36Sfi--E_fnWoLIlb4MSWrN8Hntxf7.NSr2fq4jPKQyUAaBQVDQs2icrf8-IcLf3JWDcIn_JL_Lq9MYFvRm5lNnTE9mid-M-rI7ebbnQNwQhX9LTEpnKg&_iepl=)
* 聚类融合
* 进行聚类融合的动机是增加聚类算法的健壮性


---
## [Event Detection in Social Streams](http://charuaggarwal.net/sdm2012-camera.pdf)
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
* The assumption in the supervised case is that the training data about the social stream objects which are related to the event are available in the historical training data
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
* 使用标准正太分布来挑选阈值
* 社交网络相似度的计算没太看懂
* 以hashtag作为ground truth并不合理，如果能直接识别出hashtag，聚类将没有意义，对于多标签的处理也不明确

---
## [Identifying Content for Planned Events Across Social Media Sites](http://delivery.acm.org/10.1145/2130000/2124360/p533-becker.pdf?ip=202.120.224.53&id=2124360&acc=ACTIVE%20SERVICE&key=BF85BBA5741FDC6E%2E88014DC677A1F2C3%2E4D4702B0C3E38B35%2E4D4702B0C3E38B35&__acm__=1570539867_62c67a2fe4e6077fb3bc6f0e28ed3534)
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
* Planned Event ``Dataset``: We assembled a dataset of event records posted between May 13, 2011 and June 11, 2011 on four dierent event aggregation platforms
* comparison against human-produced queries, human evaluation of generated queries, and evaluation of document retrieval results. use the ``Jaccard coeficient`` to measure the similarity of the set of automatically generated queries G to the set of human-produced queries H for each event.
* asked two annotators to label 2,037 queries selected by our strategies for each event on a scale of 1-5, based on their relevance to the event.

### 思考
* 类似于从事件文本中提取代表文本、代表的关键词，提取有效信息，在选取事件表示时值得借鉴
* 事件是有时效性的
* 结合了基于准确率和基于召回率的方法，即考虑不同粒度
* 文章还使用了不同来源的文本，相互促进

---
## [Event Discovery in Social Media Feeds](http://aria42.com/pubs/events.pdf)
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
* 推特中的每一个词都有表示事件属性的标签

---
## [TwitterNews: Real time event detection from the Twitter data stream](https://www.researchgate.net/profile/Mahmud_Hasan8/publication/309426330_TwitterNews_Real_time_event_detection_from_the_Twitter_data_stream/links/581a12b008aeffb294130fd1/TwitterNews-Real-time-event-detection-from-the-Twitter-data-stream.pdf?_sg%5B0%5D=TnznRnCrOp6ZZclRyRwcqEB4IIRkbOvPhDGOkLY403iD2TAh87WNHmQ3YgnlhW8H_kxDJ4o4zW7AvOJIUF_s6Q.rvivN9t4tv9QsIZ-bRsBICXS9YUkxh3FnhWYdRYuE0aqJQP9dlN752umgbo-SgQC6Or5VYV4nH1rhlxVQ0vbrQ&_sg%5B1%5D=W7u992XqavOYSoexjseJ7Q97t4XvzkCJC-9DGowiktOVZ4moje8qyeerx2nuRO5dB9wPyEYIS7EGMEmtYDwb_bLE83W_ZeQ7AW1GznnRG27r.rvivN9t4tv9QsIZ-bRsBICXS9YUkxh3FnhWYdRYuE0aqJQP9dlN752umgbo-SgQC6Or5VYV4nH1rhlxVQ0vbrQ&_iepl=)

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
* 作者在第一阶段选择计算历史推特和当前推特的相似度，超过一定阈值才进入下一阶段，历史推特维持一个固定值，这个方法的计算开销也很大，如果把独立的推特当成独立的事件，直接计算推特和事件的相似度，合并两个阶段，计算量小的同时也能达到这样的效果，不过这种方法的优点是会减少事件的数量，不是所有推特都形成事件，只有在一定的时间窗口内有多条推特讨论时才考虑进行事件的聚类，这对于**讨论较少**的事件和**间隔较长**的事件不友好，这和数据窗口的长度（需指定）以及数据的完整有很大的关系
* 通过LSH（Locality Sensitive Hashing）方法优化较少计算量的方法值得进一步调研
* 作者也将事件设置了一个有效期（8-15分钟），时间有点短，但是时间一长容易导致事件数太多
* 作者的评估方法存疑，31个事件中识别出16个事件则召回率为0.52，认为识别出一个事件的标准是什么？和ground truth有多少重叠认为是同一事件？准确率也是根据随机挑选的100个事件来计算的，并不合理

---
## [Streaming First Story Detection with application to Twitter](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.170.9438&rep=rep1&type=pdf)
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
## [ELD: Event TimeLine Detection - A Participant-Based Approach to Tracking Events](http://delivery.acm.org/10.1145/3350000/3344921/p267-mamo.pdf?ip=65.49.38.140&id=3344921&acc=OPEN&key=4D4702B0C3E38B35%2E4D4702B0C3E38B35%2E4D4702B0C3E38B35%2E6D218144511F3437&__acm__=1571052520_5df392b95790ecb98b40e864b95da3ad)
* ELD’s inputs do not only include a seed set of keywords that describe the event , but also the lengths of two time windows. In the **first time window**, ELD uses the user’s seed set to collect tweets that shape its understanding of the event. In the **second time window**, ELD uses this understanding to follow the event and build its timeline
* we use this dataset to build Reed et al.’s Term Frequency- Inverse Corpus Frequency (**TF-ICF**) instead
* FIRE adopts a traditional incremental clustering algorithm as the　**document-pivot** approach. Incoming tweets join the most similar　cluster if the similarity exceeds a threshold; otherwise, they form　a new cluster.

### 思考
* 作者没有用TF-IDF，而是用的TF-ICF

---
## [BURSTY TOPIC DETECTION IN TWITTER](http://dspace.dtu.ac.in:8080/jspui/bitstream/repository/16327/1/SHIVANI%202K16SWE14%20Thesis%20report.pdf)
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
## [StoryMiner: An Automated and Scalable Framework for Story Analysis and Detection from Social Media](https://escholarship.org/content/qt9637m3j1/qt9637m3j1.pdf?t=pwzuo0)
### ABSTRACT OF THE DISSERTATION
* StoryMiner derives stories and narrative structures by automatically 
    * extracting and co-referencing the actants (entities such as people and objects) and their relationships from the text by proposing an Open **Information Extraction** system, 
    * assigning named-entity types and importance scores for entities and relationships using character-level neural language architectures and other traditional machine learning models, 
    * making use of context-dependent word embeddings to aggregate actant-relationships and form contextual story graphs in which the nodes are the actants and the edges are their relationships, and 
    * enriching the story graphs with additional layers of information such as sentiments or sequence orders of relationships.

### Introduction
* the main theme of this research is to introduce a narrative framework which is capable of identifying and representing narrative structures and contents from a large corpus of text

---
## [Adaptive Multi-Attention Network Incorporating Answer Information for Duplicate Question Detection](http://delivery.acm.org/10.1145/3340000/3331228/p95-liang.pdf?ip=202.120.224.53&id=3331228&acc=ACTIVE%20SERVICE&key=BF85BBA5741FDC6E%2E88014DC677A1F2C3%2E4D4702B0C3E38B35%2E4D4702B0C3E38B35&__acm__=1571709823_661dc7afae46acce82bc80b96e3b56ab#URLTOKEN#)

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
## [Pairwise Word Interaction Modeling with Deep Neural Networks for Semantic Similarity Measurement](https://www.aclweb.org/anthology/N16-1108.pdf)

* Most previous work use sentence modeling with a “Siamese” structure
### Model Overview
* Bidirectional Long Short-Term Memory Networks (**Bi-LSTMs**) are used for context modeling of input sentences, which serves as the basis for all following components.
* A novel **pairwise word interaction modeling** technique encourages direct comparisons between word contexts across sentences.
* A novel **similarity focus layer** helps the model identify important pairwise word interactions across sentences.
* A 19-layer deep convolutional neural network (**ConvNet**) converts the similarity measurement problem into a pattern recognition problem for final classification.
* this is the first neural network model, a novel hybrid architecture **combining** Bi-LSTMs and a deep ConvNet, that uses a **similarity focus mechanism** with selective attention to important pairwise word interactions for the STS problem.

---
## [Semi-supervised Question Retrieval with Gated Convolutions](https://arxiv.org/pdf/1512.05726.pdf)

### Abstract
* The task is difficult since 1) key pieces of information are often buried in extraneous details in the question body and 2) available annotations on similar questions are scarce and fragmented.
* Several factors make the problem difficult. 
    * First, submitted questions are often long and contain extraneous information irrelevant to the main question being asked.
    * The second challenge arises from the noisy annotations.

### 思考
* 作者在问句匹配任务中使用一正一负的句子对作为训练数据

---
## [Feature Driven Learning Framework for Cybersecurity Event Detection](https://asonamdata.com/ASONAM2019_Proceedings/pdf/papers/029_0196_083.pdf)
* 对于社交媒体的网络安全事件检测，之前的方法基本上是集中于无监督和弱监督的方法，这些方法在现实中效果不佳，存在特征稀疏、处理弱信号（在大量数据中的少量数据）能力不强，模型泛化能力不强等缺点，这篇论文提出了一种多任务学习的有监督模型
* 将不同种类的目标机构视为不同的任务
* 收集2014年1月至2016年12月的893个网络安全事件，将大量的推特数据分为训练集和测试集，用命名实体所属机构进行标注

* 标注方法存疑

---
## [Jointly Detecting and Extracting Social Events From Twitter Using Gated BiLSTM-CRF](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8866717&tag=1)
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
* 标注的方式很大程度上决定了解决的问题，文中根据关键词的方法进行事件检测，解决的就是针对特定领域事件的推特进行判断，正负均衡，泛化性不高

---

