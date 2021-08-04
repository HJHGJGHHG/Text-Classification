import jieba
import numpy as np
import pandas as pd
import torch

import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

import config


def tokenize_sentences(args, sentences, words_dict, rm_stop_word=False):
    """
    :param sentences: list
    :param words_dict: {}
    :param rm_stop_word: bool, 是否去除停用词
    :return:
        tokens 分词后的句子
        words_dict 词表
    """
    sentences_preprocessed = []
    for sentence in tqdm(sentences):
        if hasattr(sentence, "decode"):
            sentence = sentence.decode("utf-8")
        
        result = []
        tokens = [word for word in jieba.cut(sentence) if word.strip()]
        if rm_stop_word:
            with open("../data/stopWord.json", "r", encoding="utf-8") as f:
                stopWords = f.read().split("\n")
            tokens = [word for word in tokens if word not in stopWords]
        
        for word in tokens:
            word = word.lower()
            if word not in words_dict:
                words_dict[word] = len(words_dict)
            word_index = words_dict[word]
            result.append(word_index)
        
        sentences_preprocessed.append(result)
    
    return sentences_preprocessed, words_dict


def read_embedding_list(file_path):
    """
    return:
        embedding_list:         2M x 300
        embedding_word_dict:    {'word': id} length 2M
    """
    embedding_word_dict = {}
    embedding_list = []
    with open(file_path, encoding="utf-8") as f:
        for row in tqdm(f.read().split("\n")[1:-1]):
            data = row.split(" ")
            word = data[0]
            embedding = np.array([float(num) for num in data[1:-1]])
            embedding_list.append(embedding)
            embedding_word_dict[word] = len(embedding_word_dict)
    
    # embedding_list = np.array(embedding_list)
    return embedding_list, embedding_word_dict


def clear_embedding_list(embedding_list, embedding_word_dict, words_dict):
    """
    return:
        cleared_embedding_list:         W x 300, W is the number --> words_dict & embedding_word_dict
        cleared_embedding_word_dict:    {'word': id} length W
    """
    cleared_embedding_list = []
    cleared_embedding_word_dict = {}
    
    for word in words_dict:
        if word not in embedding_word_dict:
            continue
        word_id = embedding_word_dict[word]
        row = embedding_list[word_id]
        cleared_embedding_list.append(row)
        cleared_embedding_word_dict[word] = len(cleared_embedding_word_dict)
    
    return cleared_embedding_list, cleared_embedding_word_dict


def convert_tokens_to_ids(tokenized_sentences, words_list, embedding_word_dict, sentences_length):
    words_ids= []

    for sentence in tokenized_sentences:
        current_words = []
        for word_index in sentence:
            word = words_list[word_index]
            word_id = embedding_word_dict.get(word, len(embedding_word_dict) - 2)  # id in embedding_word_dict
            current_words.append(word_id)

        if len(current_words) >= sentences_length:
            current_words = current_words[:sentences_length]
        else:
            current_words += [len(embedding_word_dict) - 1] * (sentences_length - len(current_words))  # add END_WORD
        words_ids.append(current_words)
    return torch.LongTensor(words_ids)


class TeaRateCommentDataSet(Dataset):
    def __init__(self, args, phase):
        self.args = args
        
        self.UNKNOWN_WORD = '_UNK_'
        self.END_WORD = '_END_'
        self.dir_train = args.path + "train.csv"
        self.dir_test = args.path + "test.csv"
        self.x, self.y = self.load_dataset(phase=phase)
    
    def load_dataset(self, phase):
        # 加载数据
        print("============Loading data============")
        train_data = pd.read_csv(self.dir_train)
        test_data = pd.read_csv(self.dir_test)
        list_sentences_train = train_data[self.args.text_label].values
        list_sentences_test = test_data[self.args.text_label].values
        label_train = train_data[self.args.labels].values
        label_test = test_data[self.args.labels].values
        # Tokenize
        train_data_processed, words_dict = np.array(tokenize_sentences(self.args, list_sentences_train, {}))
        test_data_processed, words_dict = np.array(tokenize_sentences(self.args, list_sentences_test, words_dict))
        self.args.vocabulary_size = len(words_dict)
        
        # 加载Embedding
        # 使用预训练的词向量
        print("============Load Embedding============")
        embedding_list, embedding_word_dict = read_embedding_list(
            self.args.pretrained_path + self.args.pretrained_name)
        self.args.embedding_dim = len(embedding_list[0])
        
        # 精简Embedding表
        embedding_list, embedding_word_dict = clear_embedding_list(embedding_list, embedding_word_dict, words_dict)
        embedding_word_dict[self.UNKNOWN_WORD] = len(embedding_word_dict)
        embedding_list.append([0.] * self.args.embedding_dim)
        embedding_word_dict[self.END_WORD] = len(embedding_word_dict)
        embedding_list.append([-1.] * self.args.embedding_dim)

        id_to_word = dict((id, word) for word, id in words_dict.items())
        train_sentences_tokenized = convert_tokens_to_ids(
            train_data_processed,
            id_to_word,
            embedding_word_dict,
            100)
        test_sentences_tokenized = convert_tokens_to_ids(
            test_data_processed,
            id_to_word,
            embedding_word_dict,
            100)

        embedding_list = torch.FloatTensor(embedding_list)
        self.embedding = nn.Embedding(num_embeddings=len(embedding_word_dict),
                                 embedding_dim=self.args.embedding_dim)
        if self.args.static == True:
            self.embedding = nn.Embedding(num_embeddings=len(embedding_word_dict),
                                     embedding_dim=self.args.embedding_dim).from_pretrained(embedding_list,
                                                                                       freeze=self.args.non_static)
        
        if phase == "Train":
            return train_sentences_tokenized, label_train
        else:
            return test_sentences_tokenized, label_test
    
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        """ get one sample"""
        sample = dict()
        sample[self.args.text_label] = self.x[idx]
        
        sample["labels"] = self.y[idx].astype(np.float32)
        
        return sample


def get_dataset(args):
    train_dataset = TeaRateCommentDataSet(args=args, phase="Train")
    test_dataset = TeaRateCommentDataSet(args=args, phase="Test")
    embedding = train_dataset.embedding
    
    train_iter = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    test_iter = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    return train_iter, test_iter, embedding


