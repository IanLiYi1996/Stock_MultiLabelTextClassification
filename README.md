### Multi-Label Text Classification for Stock Relevance

> models for MLTC in stock relevance

- **文章多标签分类简述**
- 目的：对于给定的文章推荐相关的股票标签，等价于分类问题
  - 语料来源两部分：新闻等资讯文章，知识图谱中的实体知识
- 难点：该任务定义在文章级别的粒度上，难以在股票和文章之间建立语义关联
- **多标签类别定义**
  - 文章类别可通过使用bi-affine模型来进行判断：
    - 第一个FFN来做二分类判断文章是否和投资相关（解决文章投资无关导致的歧义问题）
    - 第二个FFN来对文章与股票进行匹配和搜索，得到股票关于文章的得分（解决文本对于股票不定类别的分类问题，需要进一步缩小股票的搜索范围）
    - 将两个分类器按照bi-affine的方式进行级联，得到投资相关文章下的股票得分
  - 股票范围可按照industry产业字段进行分类：
    - 可将所有的股票分类为对应的66类产业中的一类，按照产业进行股票备选
    - 新闻文本也可按照产业信息进行多标签分类，得到分数最高的几个类别，再进行与股票的语义建模。
- **Model**
  - Knowledge Extraction -> 获得文章中的股票、公司实体、人物
  - BERT+ biaffine+FFN -> 按照板块对文章进行多标签分类
    - 细节：基于bert的模型使用最后一层中第一个标记（例如[CLS]）的隐藏状态，以及若干逻辑回归（例如linear + sigmoid）来预测多个标签。使用binary cross entropy损失。
  - Semantic Matching：使用语义搜索/阅读理解的方法来实现
- **Links**
  - Transformer based MLTC [基于BERT的文章多标签分类](https://github.com/laddie132/Transformers-MLTC)
  - Bert-Chinese-Text-Classification-Pytorch [BERT+CNN/RNN](https://github.com/649453932/Bert-Chinese-Text-Classification-Pytorch)
  - LWPT [Label-Wised Document Pre-Training for Multi-Label Text Classification](https://github.com/laddie132/LW-PT)
  - Biaffine [bi-affine attention](https://github.com/tdozat/Parser-v1)
  - BERT [transformer](https://github.com/huggingface/transformers)
- **Paper**
  - [Bi-affine attention](https://openreview.net/pdf?id=Hk95PK9le)
  - [BERT](https://arxiv.org/abs/1810.04805)
- **Requirements** 
  - pytorch|tensorflow
  - transformer
  - tqdm

