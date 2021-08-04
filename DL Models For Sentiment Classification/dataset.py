import re
import jieba
import logging
import torch
from torchtext.legacy.data import TabularDataset
from torchtext.legacy.data import Iterator
from torchtext.vocab import Vectors

jieba.setLogLevel(logging.INFO)

regex = re.compile(r'[^\u4e00-\u9fa5aA-Za-z0-9]')


def word_cut(text):
    text = regex.sub(' ', text)
    return [word for word in jieba.cut(text) if word.strip()]


def get_dataset(args, text_field, label_field):
    text_field.tokenize = word_cut
    train, dev = TabularDataset.splits(
        path=args.path,
        format='tsv', skip_header=True,
        train="sentiment_train.tsv", validation="sentiment_test.tsv",
        fields=[
            ('index', None),
            ("sentiment", label_field),
            ("rateContent", text_field)
        ]
    )
    return train, dev

def load_word_vectors(model_name, model_path):
    vectors = Vectors(name=model_name, cache=model_path)
    return vectors


def load_dataset(text_field, label_field, args, **kwargs):
    train_dataset, dev_dataset = get_dataset(args, text_field, label_field)
    if args.static and args.pretrained_name and args.pretrained_path:
        vectors = load_word_vectors(args.pretrained_name, args.pretrained_path)
        text_field.build_vocab(train_dataset, dev_dataset, vectors=vectors)
    else:
        text_field.build_vocab(train_dataset, dev_dataset)
    label_field.build_vocab(train_dataset, dev_dataset)

    # 参数补充
    args.vocabulary_size = len(text_field.vocab)
    args.ngram_vocabulary_size = len(text_field.vocab)
    if args.static:
        args.embedding_dim = text_field.vocab.vectors.size()[-1]
        args.vectors = text_field.vocab.vectors
    if args.multichannel:
        args.static = True
        args.non_static = True
    args.class_num = len(label_field.vocab)
    args.cuda = args.device != -1 and torch.cuda.is_available()
    args.filter_sizes = [int(size) for size in args.filter_sizes.split(',')]
    
    train_iter, dev_iter = Iterator.splits(
        (train_dataset, dev_dataset),
        batch_sizes=(args.batch_size, args.batch_size),
        sort_key=lambda x: len(x.rateContent),
        **kwargs)
    return train_iter, dev_iter

