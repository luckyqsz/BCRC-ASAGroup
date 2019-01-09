# GLUE

---

* paper<br>
[GLUE: A Multi-Task Benchmark and Analysis Platform for Natural Language Understanding](https://arxiv.org/pdf/1804.07461.pdf)
* website<br>
[gluebenchmark](https://gluebenchmark.com/)<br>
[GLUE-baselines](https://github.com/nyu-mll/GLUE-baselines)
* dataset [下载](https://pan.baidu.com/s/1hnVEQdxLZbR_noX9kmeGoA)<br>
![](https://github.com/qiuxingfa/picture_/blob/master/2019/61b84cdaf5188798004c942e59ffe47.png)<br>
<br>

* detail
1. [CoLA](https://nyu-mll.github.io/CoLA/)(The Corpus of Linguistic Acceptability)  [data](https://firebasestorage.googleapis.com/v0/b/mtl-sentence-representations.appspot.com/o/data%2FCoLA.zip?alt=media&token=46d5e637-3411-4188-bc44-5809b5bfb5f4)<br>
**T/D/V**：8.5k/1k/1k<br>
**来源**：从书籍和期刊中选取的文章<br>
**任务**：判断词序列是否构成一个符合语法的句子<br>
**类别**：0和1<br>
**评估**：Matthews correlation coefficient（马修斯相关系数）<br>
**示例**：<br>

|  source  | lable | the acceptability judgment as originally notated by the author | a sequence of words |
| :----: | :---: | :---: | :---: |
| gj04  | 1 |  | Our friends won't buy this analysis, let alone the next one we propose. | 
| gj04   | 0  | *  | They drank the pub.  | 

2. [SST-2](https://nlp.stanford.edu/sentiment/index.html)（The Stanford Sentiment Treebank）[data](https://firebasestorage.googleapis.com/v0/b/mtl-sentence-representations.appspot.com/o/data%2FSST-2.zip?alt=media&token=aabc5f6b-e466-44a2-b9b4-cf6337f84ac8) <br>
**T/D/V**：67k/872/1.8k<br>
**来源**：电影评论<br>
**任务**：情感分析<br>
**类别**：1和0 (positive/negative)<br>
**评估**：accuracy<br>
**示例**：<br>

|sentence|	label|
|:---:|---:|
|contains no wit , only labored gags | 	0|
|that loves its characters and communicates something rather beautiful about human nature| 	1|

3. [MRPC](https://www.microsoft.com/en-us/download/details.aspx?id=52398)（The Microsoft Research Paraphrase Corpus）[data](https://firebasestorage.googleapis.com/v0/b/mtl-sentence-representations.appspot.com/o/data%2Fmrpc_dev_ids.tsv?alt=media&token=ec5c0836-31d5-48f4-b431-7480817f1adc)<br>
**T/D/V**：3.7k/408/1.7k<br>
**来源**：网络新闻<br>
**任务**：判断一对句子语义是否相同<br>
**类别**：1和0<br>
**评估**：accuracy and F1 score<br>
**示例**：<br>

|Quality|#1 ID|#2 ID|#1 String|#2 String|
|:---:|---:|---:|---:|---:|
|1 | 702876|702977|Amrozi accused his brother , whom he called " the witness " , of deliberately distorting his evidence .| 	Referring to him as only " the witness " , Amrozi accused his brother of deliberately distorting his evidence .|
|0 | 264589|264502|The Nasdaq had a weekly gain of 17.27 , or 1.2 percent , closing at 1,520.15 on Friday .| 	The tech-laced Nasdaq Composite .IXIC rallied 30.46 points , or 2.04 percent , to 1,520.15 .|

4. [QQP](https://data.quora.com/First-Quora-Dataset-Release-Question-Pairs)（The Quora Question Pairs）[data](https://firebasestorage.googleapis.com/v0/b/mtl-sentence-representations.appspot.com/o/data%2FQQP.zip?alt=media&token=700c6acf-160d-4d89-81d1-de4191d02cb5)<br>
**T/D/V**：364k/40k/391k<br>
**来源**：Quora<br>
**任务**：判断一对问题语义是否相同<br>
**类别**：1和0 <br>
**评估**：accuracy and F1<br>
**示例**：<br>

|id|qid1|qid2|question1|question2|is_duplicate|
|:---:|---:|---:|---:|---:|---:|
|133273 | 213221|213222|How is the life of a math student?Could you describe your own experiences?|Could you describe your own experiences?|0|
|402555| 536040|536041|How do I control my horny emotions?|How do you control your horniness?|1|

5. [STS-B](http://ixa2.si.ehu.es/stswiki/index.php/STSbenchmark) （The Semantic Textual Similarity Benchmark）[data](https://firebasestorage.googleapis.com/v0/b/mtl-sentence-representations.appspot.com/o/data%2FSTS-B.zip?alt=media&token=bddb94a7-8706-4e0d-a694-1109e12273b5) <br>
**T/D/V**：7k/1.5k/1.4k<br>
**来源**：新闻提要，视频和图片注释等<br>
**任务**：判断一对问题语义是否相同<br>
**类别**：相似度分数1-5 <br>
**评估**：Pearson and Spearman correlation coefficients（相关系数）<br>
**示例**：<br>

|index|	genre|	filename|	year|	old_index|	source1|	source2|	sentence1|	sentence2|	score|
|:---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|0|	main-captions|	MSRvid|	2012test|	|0001|	none|	none|	A plane is taking off.|	An air plane is taking off.|5.000|
|1	|main-captions|	MSRvid|	2012test|	0004|	none|	none|	A man is playing a large flute.|	A man is playing a flute.|	3.800|

6. [MNLI](http://www.nyu.edu/projects/bowman/multinli/)（The Multi-Genre Natural Language Inference Corpus）[data](https://firebasestorage.googleapis.com/v0/b/mtl-sentence-representations.appspot.com/o/data%2FMNLI.zip?alt=media&token=50329ea1-e339-40e2-809c-10c40afff3ce) <br>
**T/D/V**：393k/20k/20k<br>
**来源**：演讲，小说，政府报告等<br>
**任务**：给出前提句（premise）和假设句（hypothesis），判断前提句是否蕴含（entails）假设句，或者与其矛盾（contradicts）或者中立（neutral）<br>
**类别**：entails，contradicts，neutral <br>
**评估**：matched (in-domain) and mismatched (cross-domain) accuracy<br>
**示例**：<br>

|index|	promptID|	pairID|	genre|	sentence1_binary_parse|	sentence2_binary_parse	| sentence1_parse|sentence2_parse|	sentence1|	sentence2|	label1|	gold_label|
|:---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|0	|31193|	31193n|	government|( ( Conceptually ( cream skimming ) ) ( ( has ( ( ( two ( basic dimensions ) ) - ) ( ( product and ) geography ) ) ) . ) )|( ( ( Product and ) geography ) ( ( are ( what ( make ( cream ( skimming work ) ) ) ) ) . ) )|(ROOT (S (NP (JJ Conceptually) (NN cream) (NN skimming)) (VP (VBZ has) (NP (NP (CD two) (JJ basic) (NNS dimensions)) (: -) (NP (NN product) (CC and) (NN geography)))) (. .)))|(ROOT (S (NP (NN Product) (CC and) (NN geography)) (VP (VBP are) (SBAR (WHNP (WP what)) (S (VP (VBP make) (NP (NP (NN cream)) (VP (VBG skimming) (NP (NN work)))))))) (. .)))|Conceptually cream skimming has two basic dimensions - product and geography.|Product and geography are what make cream skimming work. |neutral|neutral|
|5|	110116|	110116e|	telephone|( ( my walkman ) ( broke ( so ( i ( 'm ( upset ( now ( i ( just ( have ( to ( ( turn ( the stereo ) ) ( up ( real loud ) ) ) ) ) ) ) ) ) ) ) ) ) )|( ( ( ( I ( 'm ( upset ( that ( ( my walkman ) broke ) ) ) ) ) and ) ( now ( I ( have ( to ( ( turn ( the stereo ) ) ( up ( really loud ) ) ) ) ) ) ) ) . )|(ROOT (S (NP (PRP$ my) (NN walkman)) (VP (VBD broke) (SBAR (IN so) (S (NP (FW i)) (VP (VBP 'm) (ADJP (VBN upset) (SBAR (RB now) (S (NP (FW i)) (ADVP (RB just)) (VP (VBP have) (S (VP (TO to) (VP (VB turn) (NP (DT the) (NN stereo)) (ADVP (RB up) (RB real) (JJ loud)))))))))))))))|(ROOT (S (S (NP (PRP I)) (VP (VBP 'm) (ADJP (VBN upset) (SBAR (IN that) (S (NP (PRP my) (NN walkman)) (VP (VBD broke))))))) (CC and) (S (ADVP (RB now)) (NP (PRP I)) (VP (VBP have) (S (VP (TO to) (VP (VB turn) (NP (DT the) (NN stereo)) (ADVP (RB up) (RB really) (JJ loud))))))) (. .)))|How do you know? All this is their information again.|This information belongs to them.|entailment|entailment|
7. [SNLI](https://firebasestorage.googleapis.com/v0/b/mtl-sentence-representations.appspot.com/o/data%2FSNLI.zip?alt=media&token=4afcfbb2-ff0c-4b2d-a09a-dbf07926f4df)（论文未提）<br>
8. [QNLI](https://rajpurkar.github.io/SQuAD-explorer/) （The Stanford Question Answering Dataset）[data](https://firebasestorage.googleapis.com/v0/b/mtl-sentence-representations.appspot.com/o/data%2FQNLI.zip?alt=media&token=c24cad61-f2df-4f04-9ab6-aa576fa829d0)<br>
**T/D/V**：108k/5.7k/5.7k<br>
**来源**：Wikipedia<br>
**任务**：判断一个段落中是否包含问题的答案<br>
**类别**：entailment，not_entailment<br>
**评估**：accuracy<br>
**示例**：<br>

|index|	question|	sentence|	label|
|:---:|---:|---:|---:|
|0	|What is the Grotto at Notre Dame?|	Immediately behind the basilica is the Grotto, a Marian place of prayer and reflection.|	entailment|
|1	|What is the Grotto at Notre Dame?|	It is a replica of the grotto at Lourdes, France where the Virgin Mary reputedly appeared to Saint Bernadette Soubirous in 1858.|	not_entailment|
9. [RTE](https://aclweb.org/aclwiki/Recognizing_Textual_Entailment) （The Recognizing Textual Entailment）[data](https://firebasestorage.googleapis.com/v0/b/mtl-sentence-representations.appspot.com/o/data%2FRTE.zip?alt=media&token=5efa7e85-a0bb-4f19-8ea2-9e1840f077fb)<br>
**T/D/V**：2.5k/276/3k<br>
**来源**：news and Wikipedia text<br>
**任务**：判断两个句子的蕴含关系<br>
**类别**：entailment，not_entailment<br>
**评估**：accuracy<br>
**示例**：<br>

|index|	sentence1|	sentence2|	label|
|:---:|---:|---:|---:|
|0|	No Weapons of Mass Destruction Found in Iraq Yet.|	Weapons of Mass Destruction Found in Iraq.|	not_entailment|
|1|	A place of sorrow, after Pope John Paul II died, became a place of celebration, as Roman Catholic faithful gathered in downtown Chicago to mark the installation of new Pope Benedict XVI.|	Pope Benedict XVI is the new leader of the Roman Catholic Church.|	entailment|
10. [WNLI](https://cs.nyu.edu/faculty/davise/papers/WinogradSchemas/WS.html)（The Winograd Schema Challenge）（bert表格中没有）[data](https://firebasestorage.googleapis.com/v0/b/mtl-sentence-representations.appspot.com/o/data%2FWNLI.zip?alt=media&token=068ad0a0-ded7-4bd7-99a5-5e00222e0faf)<br>
**T/D/V**：634/71/146<br>
**来源**：小说<br>
**任务**：将句子中的代词换成它可能指代的事物，判断和原句是否有蕴含关系<br>
**类别**：0和1<br>
**评估**：accuracy<br>
**示例**：<br>

|index|	sentence1|	sentence2|	label|
|:---:|---:|---:|---:|
|0	|I stuck a pin through a carrot. When I pulled the pin out, it had a hole.|	The carrot had a hole.|	1|
|3	|Steve follows Fred's example in everything. He influences him hugely.|	Steve influences him hugely.|	0|
11. [diagnostic](https://gluebenchmark.com/diagnostics) |[data](https://goo.gl/dJ5GR4)<br>

|Lexical Semantics|	Predicate-Argument Structure|	Logic|	Knowledge|	Domain|	Premise	|Hypothesis|	Label|
|:---:|---:|---:|---:|---:|---:|---:|---:|
|	|	|Negation|		|Artificial|	The cat sat on the mat.	|The cat did not sit on the mat.|	contradiction|
|Morphological negation|	|	Negation|	|	Artificial|	The new gaming console is affordable.|	The new gaming console is unaffordable.|	contradiction|
|Lexical entailment	|	|Upward monotone|		|Artificial	|Some dogs like to scratch their ears.|	Some animals like to scratch their ears.|	entailment|
