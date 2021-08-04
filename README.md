# 多标签与情感分类任务

## 数据
* **原始数据：/data/teapro.csv**
* **预处理后的数据：/data/teatext_preprocessed.pkl**
  * 数据探索与预处理：见preprocessor.ipynb
* **训练集、测试集划分：**
  * 训练集：/data/train.csv
  * 测试集：/data/test.csv

## 词向量 （TF-IDF & Word2Vec）
* **[Chinese-Word-Vectors](https://github.com/Embedding/Chinese-Word-Vectors（Wiki）)**
  * 300d
  * /data/sgns.wiki.word
  * merge效果应该更好，碍于内存并未使用
* **[ELMo（未使用...）](https://github.com/HIT-SCIR/ELMoForManyLangs)**
  * 1024d

## 停用词
* /data/stopWord.json

## 模型
* **机器学习模型**
  * 线性核SVM
  * 随机森林
  * 逻辑回归
* **深度学习模型**
  * TextCNN
  * BiLSTM
  * BiGRU
  * FastText
  * BERT

## 评价指标 ： ACC & F1
---

## 一、机器学习模型
<center><img src="D:\python\pyprojects\NLP\Text Classification\pics\机器学习模型情感分类ACC值.png"  style="zoom:30%;" width="70%"/></center>
<center>图 1 机器学习模型情感分类ACC值</center>


<center><img src="D:\python\pyprojects\NLP\Text Classification\pics\机器学习模型情感分类F1值.png"  style="zoom:30%;" width="70%"/></center>
<center>图 2 机器学习模型情感分类F1值</center>



<center><img src="D:\python\pyprojects\NLP\Text Classification\pics\机器学习模型文本分类ACC值.png"  style="zoom:30%;" width="70%"/></center>
<center>图 3 机器学习模型情感分类ACC值</center><center><img 


<center><img src="D:\python\pyprojects\NLP\Text Classification\pics\机器学习模型文本分类F1值.png"  style="zoom:30%;" width="70%"/></center>
<center>图 4 机器学习模型情感分类F1值</center><center><img 

---

## 二、TextCNN

### 论文
[Convolutional Neural Networks for Sentence Classification](https://arxiv.org/abs/1408.5882)

### 参考
* https://github.com/yoonkim/CNN_sentence
* https://github.com/dennybritz/cnn-text-classification-tf
* https://github.com/Shawn1993/cnn-text-classification-pytorch
* https://github.com/bigboNed3/chinese_text_cnn

### 模型评价
* **情感分类**
  - [x] **CNN-rand 随机初始化Embedding**
        Stop Training. best model: epoch : 15 **f1 : 0.9292 acc: 88.6960%**
  - [x] **CNN-static 使用预训练的静态词向量**
        Stop Training. best model: epoch : 15 **f1 : 0.9302 acc: 88.9176%**
  - [x] **CNN-not freeze 微调静态词向量**
        Stop Training. best model: epoch : 5 **f1 : 0.9359 acc: 89.7303%**
* **多标签分类**
  - [x] **CNN-rand 随机初始化Embedding**
        Stop training. Best model: epoch : 6 **f1 : 0.8237 acc: 68.8534%**
  - [x] **CNN-static 使用预训练的静态词向量**
		Stop training. Best model: epoch : 14 **f1 : 0.8169 acc: 68.2033%**
### 提升策略
**1. Filter_size：**这个参数决定了抽取n-gram特征的长度，由于文本长度普遍在80以内，用10以下就可以了。**[1,3,5]，[2,3,4]**都不错。
**2. Filter_num：**这个参数会影响最终特征的维度，维度太大的话训练变慢。这里在**100-600**之间调参即可。
**3. 多层卷积：**效果很差
**4. 正则化：**指对CNN参数的正则化，可以使用dropout或L2，但能起的作用很小，太大的dropout反而不好。
**5. 加深全连接：**之前看[卷积层和分类层，哪个更重要？](https://www.zhihu.com/question/270245936)加深全连接效果好，但是此处加深至3、4层提升微乎其微，可能是隐藏层维度没选好。

---

## 三、BiLSTM & BiGRU + Attention

### 参考
* **https://blog.csdn.net/qq_40900196/article/details/88998290**
* **https://github.com/AaronJny/emotional_classification_with_rnn**

### 模型评价
* **情感分类**
  - [x] ** BiLSTM**
        Stop Training. best model: epoch : 13 **f1 : 0.6922 acc: 88.4374%**
  - [x] **BiLSTM + Attention**
        Stop Training. best model: epoch : 11 **f1 : 0.9313 acc: 89.0284%**
  - [x] **BiGRU**
        Stop Training. best model: epoch : 18 **f1 : 0.7381 acc: 89.0654%**
  - [x] **BiGRU + Attention**
        Stop Training. best model: epoch : 6 **f1 : 0.7308 acc: 89.2501%**
* **多标签分类**
  - [x] **BiLSTM + Attention**
  		Stop training. Best model: epoch : 4 **f1 : 0.7985 acc: 65.6324%**
  - [x] **BiGRU + Attention**
  		Stop training. Best model: epoch : 5 **f1 : 0.8058 acc: 66.5780%**

### 提升策略
**1. 加深全连接：**效果一般
**2. dropout： **效果一般
**3. batch size: **改为32略有提升:Stop Training. best model: epoch : 4 **f1 : 0.7513 acc: 89.5826%**（BiGRU + Attention)
**4. hidden size：**80，100，120均尝试过，150显存不够，效果都不好

---

## 四、FastText

### 论文
**[Bag of Tricks for Efficient Text Classification](https://arxiv.org/pdf/1607.01759v3.pdf)**

### 参考
* **https://github.com/facebookresearch/fastText**
* **https://github.com/HowieMa/Chinese_Sentence_Classification**

### 模型评价
* **情感分类**
	- [x] **FastText 随机初始化Embedding**
        Stop Training. best model: epoch : 20 **f1 : 0.7036 acc: 87.7355%**
	- [x] **FastText-Word2Vec 使用预训练的静态词向量**
        Stop Training. best model: epoch : 29 **f1 : 0.6845 acc: 87.8094%**
* **多标签分类**

### 提升策略
**1. batch size：调小为64有提升：**
    Stop Training. best model: epoch : 24 **f1 : 0.7472 acc: 89.2501%（FastText）** 
    Stop Training. best model: epoch : 19 **f1 : 0.7504 acc: 88.6960%%（FastText-Word2Vec）**
    
## 五、BERT

### 论文
**[BERT: Pre-training of Deep Bidirectional Transformers forLanguage Understanding](https://static.aminer.cn/upload/pdf/program/5bdc31b417c44a1f58a0b8c2_0.pdf)**

### 参考
* **https://github.com/649453932/Bert-Chinese-Text-Classification-Pytorch**
* **https://github.com/real-brilliant/bert_chinese_pytorch**

### 模型评价
* **情感分类**
	**Test Loss:  0.17,  Test Acc: 92.95%   Test F1 : 0.797742**
* **多标签分类**

### 提升策略
**0. 微调（Fine-Tune）：**
**1. 不同的预训练模型：**如RoBERT、WWM、ALBERT
**2. 以BERT作为Embedding层，再将其喂给其他模型如CNN、RCNN等：**
    BERT-CNN：Test Loss:  0.29,  **Test Acc: 87.29%   Test F1 : 0.721232**
    BERT-RCNN：Test Loss:  0.27,  **Test Acc: 88.77%   Test F1 : 0.730973**
**3. 在领域数据集上进行增量预训练(Futher-Pretraining)**

## 六、模型与参数比较
* 机器学习模型普遍不好，acc只有不到0.65，速度倒是蛮快的，参数没怎么调，精度要求很低的情况下作为baseline不错。值得注意的是使用TF-IDF比Word2Vec要好，挺玄学的。
* 之后又试了MLP，作为一个入门模型，情感分类得到acc也能达到0.8，~~只能说运气不错吧。~~
* 在深度学习模型的预处理中，尝试了分词和基于字的tokenize（更改dataset文件中的相应tokenize函数即可），发现分词后的模型效果明显好于基于字的，说明分词很有必要！！
* 把模型变得更深更复杂确实有提升，但复杂到一定程度后再加深提升就很小了，而且此时不同模型之间的区别也不大。
* 血泪史：
    * 在最开始做情感分类的时候图快用了torchtext，对于传统的一些模型，例如CNN，LSTM等，使用起来还是比较方便的。但是由于torchtex封装太高层了，一些想要自定义的功能却很难实现。导致后面做BERT和多标签分类的时候又重新写了预处理的代码。torchtext相关代码保存在DL Models for Sentiment Classificaiton中，以供参考。
    * 多折训练：过拟合很严重，不如之前的模型
    * 对抗训练：理解相关原理很耗时，现在也是一知半解的，本质是对Embedding参数矩阵加扰动，凭感觉实现了一下。训练的时候在在正常的grad基础上，累加对抗训练的梯度即可。可惜的就是FGM的效果并不好，最终test loss只降了0.007左右...
``` python
class FGM():
    def __init__(self, model):
        self.model = model
        self.backup = {}

    def attack(self, epsilon=1., emb_name='emb'):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0:
                    r_at = epsilon * param.grad / norm
                    param.data.add_(r_at)

    def restore(self, emb_name='emb'):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name: 
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}
```

---

## 七、结语
	这次任务花了大量时间在预处理与语法上，还是学到了很多知识，还有很多构想没实现诸如BN和模型融合，学海无涯，需坚持不懈笃行之。

