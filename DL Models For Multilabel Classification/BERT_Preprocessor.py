import pandas as pd
import torch
import time

from datetime import timedelta
from pytorch_pretrained_bert import BertModel, BertTokenizer

PAD, CLS, SEP = "[PAD]", "[CLS]", "[SEP]"  # padding符号, bert中综合信息符号


def load_dataset(data_df, args):
    contents = []
    pad_size = args.pad_size
    for i in range(len(data_df)):
        rateContent, sentiment = data_df["rateContent"][i], data_df["sentiment"][i]
        token = args.tokenizer.tokenize(rateContent[:(args.pad_size - 2)])
        token = [CLS] + token + [SEP]
        seq_len = len(token)
        mask = []
        token_ids = args.tokenizer.convert_tokens_to_ids(token)
        
        if len(token) < pad_size:
            mask = [1] * len(token_ids) + [0] * (pad_size - len(token))
            token_ids += ([0] * (pad_size - len(token)))
        else:
            mask = [1] * pad_size
            token_ids = token_ids[:pad_size]
            seq_len = pad_size
        contents.append((token_ids, int(sentiment), seq_len, mask))
    return contents


def build_dataset(args):
    args.tokenizer = BertTokenizer.from_pretrained(args.bert_pretrained)
    
    train = pd.read_csv(args.path + "sentiment_train.csv", encoding="utf-8")
    test = pd.read_csv(args.path + "sentiment_test.csv", encoding="utf-8")
    
    train_data = load_dataset(train, args)
    test_data = load_dataset(test, args)
    return train_data, test_data


class DatasetIterater(object):
    def __init__(self, batches, args):
        self.batch_size = args.batch_size
        self.batches = batches
        self.n_batches = len(batches) // args.batch_size
        self.residue = False  # 记录batch数量是否为整数
        if len(batches) % self.n_batches != 0:
            self.residue = True
        self.index = 0
        if args.device != -1 and torch.cuda.is_available():
            self.device = "cuda"
            args.cuda = True
        else:
            self.device = "cpu"
            args.cuda = False
    
    def _to_tensor(self, datas):
        x = torch.LongTensor([_[0] for _ in datas]).to(self.device)
        y = torch.LongTensor([_[1] for _ in datas]).to(self.device)
        
        # pad前的长度(超过pad_size的设为pad_size)
        seq_len = torch.LongTensor([_[2] for _ in datas]).to(self.device)
        mask = torch.LongTensor([_[3] for _ in datas]).to(self.device)
        return (x, seq_len, mask), y
    
    def __next__(self):
        if self.residue and self.index == self.n_batches:
            batches = self.batches[self.index * self.batch_size: len(self.batches)]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches
        
        elif self.index >= self.n_batches:
            self.index = 0
            raise StopIteration
        else:
            batches = self.batches[self.index * self.batch_size: (self.index + 1) * self.batch_size]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches
    
    def __iter__(self):
        return self
    
    def __len__(self):
        if self.residue:
            return self.n_batches + 1
        else:
            return self.n_batches


def build_iterator(dataset, args):
    iter = DatasetIterater(dataset, args)
    return iter

def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))