# Data-to-Text Generation with Content Selection and Planning
>代码：[Github链接](https://github.com/ratishsp/data2text-plan-py)

## Abstract
数据到文本生成的最新进展已经导致使用大规模数据集和神经网络模型，这些模型是端到端训练的，太过黑盒。
在这项工作中，我们提出了一个神经网络架构，其中包含 **content selection和planing** 而不会牺牲端到端的训练。

我们将生成任务分解为两个阶段。
给定一组数据记录（与描述性文档配对），
>1. 我们首先生成一个content plan，突出哪些信息以及何种顺序，
>2. 然后在考虑content plan的同时生成文档。

自动和基于人工的评估实验表明，我们的模型1优于强基线，改善了最近发布的ROTOWIRE数据集的最新技术水平。

