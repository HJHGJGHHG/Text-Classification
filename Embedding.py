# tf-idf, word2vec
import numpy as np
from gensim.models import KeyedVectors
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer


def tfidf(x_train, x_test):
    """
    句子的TF-IDF表示
    """
    comments_train_concat = x_train.apply(lambda x: ' '.join(x))
    comments_test_concat = x_test.apply(lambda x: ' '.join(x))
    
    vectorizer = CountVectorizer()
    trans = TfidfTransformer()
    
    word_count_train = vectorizer.fit_transform(comments_train_concat)
    word_count_test = vectorizer.transform(comments_test_concat)
    
    return trans.fit_transform(word_count_train), trans.transform(word_count_test)

def word2vec(texts):
    """
    基于开源预训练的中文词向量
    地址：https://github.com/Embedding/Chinese-Word-Vectors
    """
    model = KeyedVectors.load_word2vec_format('data/sgns.wiki.word')
    vec_len = model['NLP'].shape[0]
    
    def comm_vec(c):
        vec_com = np.zeros(vec_len)
        coun = 0
        for w in c:
            if w in model:
                vec_com += model[w]
                coun += 1
        if vec_com.dot(vec_com.T) == 0:
            return np.zeros(vec_len)
        return vec_com / coun
    
    return np.vstack(texts.apply(comm_vec))



