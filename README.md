## 多标签与情感分类任务

## 数据
* **原始数据：/data/teapro.csv**
  * **数据探索：见pdata.ipynb**
* **预处理后的数据：/data/teatext_preprocessed.pkl**
  * **预处理：见preprocessor.ipynb**
* **训练集、测试集划分：**
  * **情感分类训练集：/data/sentiment_train.csv 和 tsv**
  * **情感分类测试集：/data/sentiment_test.csv 和 tsv**
  * **多标签分类训练集：/data/multilabel_train.csv 和 tsv**
  * **多标签分类测试集：/data/multilabel_test.csv 和 tsv**

## 词向量 （TF-IDF & Word2Vec）
* **https://github.com/Embedding/Chinese-Word-Vectors（Wiki）**
* **/data/sgns.wiki.word**

## 停用词
* **/data/stopWord.json**

## 模型
* **机器学习模型**
  * **线性核SVM**
  * **随机森林**
  * **逻辑回归**
* **深度学习模型**
  * **TextCNN**
  * **BiLSTM & BiGRU**
  * **FastText**
  * **BERT**
  * **ERNIE**

## 模型评价 (ACC & F1)
<center><img src="D:\python\pyprojects\NLP\Text Classification\pics\机器学习模型情感分类ACC值.png"  style="zoom:30%;" width="70%"/></center>
<center>图 1 机器学习模型情感分类ACC值</center>


<center><img src="D:\python\pyprojects\NLP\Text Classification\pics\机器学习模型情感分类F1值.png"  style="zoom:30%;" width="70%"/></center>
<center>图 2 机器学习模型情感分类F1值</center>



<center><img src="D:\python\pyprojects\NLP\Text Classification\pics\机器学习模型文本分类ACC值.png"  style="zoom:30%;" width="70%"/></center>
<center>图 3 机器学习模型情感分类ACC值</center><center><img 


<center><img src="D:\python\pyprojects\NLP\Text Classification\pics\机器学习模型文本分类F1值.png"  style="zoom:30%;" width="70%"/></center>
<center>图 4 机器学习模型情感分类F1值</center><center><img 


