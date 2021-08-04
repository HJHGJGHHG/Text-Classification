import torch
import torch.nn as nn
import torch.nn.functional as F


def get_embeddings(args):
    chanel_num = 1
    vocabulary_size = args.vocabulary_size
    embedding_dimension = args.embedding_dim
    embedding = nn.Embedding(vocabulary_size, embedding_dimension)
    if args.static:
        embedding = embedding.from_pretrained(args.vectors, freeze=not args.non_static)
    if args.multichannel:
        embedding2 = nn.Embedding(vocabulary_size, embedding_dimension).from_pretrained(args.vectors)
        chanel_num += 1
    else:
        embedding2 = None
    return embedding, embedding2


class TextCNN(nn.Module):
    def __init__(self, args):
        super(TextCNN, self).__init__()
        self.args = args
        
        class_num = args.class_num
        chanel_num = 1
        filter_num = args.filter_num
        filter_sizes = args.filter_sizes
        
        self.embedding, self.embedding2 = get_embeddings(args)
        if args.multichannel:
            chanel_num += 1
        
        self.convs = nn.ModuleList(
            [nn.Conv2d(chanel_num, filter_num, (size, args.embedding_dim)) for size in filter_sizes])
        self.dropout = nn.Dropout(args.dropout)
        self.fc = nn.Linear(len(filter_sizes) * filter_num, class_num)
    
    def forward(self, x):
        """
        :param x: 输入，形状为(batch_size,seq_len)
        """
        if self.embedding2:
            x = torch.stack([self.embedding(x), self.embedding2(x)], dim=1)
        else:
            x = self.embedding(x)
            x = x.unsqueeze(1)
        
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs]
        x = [F.max_pool1d(item, item.size(2)).squeeze(2) for item in x]
        x = torch.cat(x, 1)
        x = self.dropout(x)
        logits = self.fc(x)
        return logits


class BiLSTM_Attention(nn.Module):
    def __init__(self, args):
        super(BiLSTM_Attention, self).__init__()
        self.args = args
        
        chanel_num = 1
        
        self.embedding, self.embedding2 = get_embeddings(args)
        if args.multichannel:
            chanel_num += 1
        
        # bidirectional设为True即得到双向循环神经网络
        self.encoder = nn.LSTM(input_size=args.embedding_dim,
                               hidden_size=args.hidden_size,
                               num_layers=args.layers_num,
                               batch_first=False,
                               bidirectional=True)
        # 初始时间步和最终时间步的隐藏状态作为全连接层输入
        self.w_omega = nn.Parameter(torch.Tensor(args.hidden_size * 2, args.hidden_size * 2))
        self.u_omega = nn.Parameter(torch.Tensor(args.hidden_size * 2, 1))
        self.decoder1 = nn.Linear(2 * args.hidden_size, args.class_num)
        self.decoder2 = nn.Linear(args.class_num, args.class_num)
        
        nn.init.uniform_(self.w_omega, -0.1, 0.1)
        nn.init.uniform_(self.u_omega, -0.1, 0.1)
    
    def forward(self, x):
        """
        :param x: 输入，形状为(batch_size,seq_len)
        """
        x.t_()  # (seq_len,batch_size)
        embeddings = self.embedding(x)
        outputs, _ = self.encoder(embeddings)  # output, (h, c)
        x = outputs.permute(1, 0, 2)  # 现在x形状为(batch_size, seq_len, 2 * hidden_size)，2表示双向
        
        # Attention过程
        u = torch.tanh(torch.matmul(x, self.w_omega))
        # u形状是(batch_size, seq_len, 2 * hidden_size)
        att = torch.matmul(u, self.u_omega)
        # att形状是(batch_size, seq_len, 1)
        att_score = F.softmax(att, dim=1)
        # att_score形状仍为(batch_size, seq_len, 1)
        scored_x = x * att_score
        # scored_x形状是(batch_size, seq_len, 2 * hidden_size)
        # Attention过程结束
        
        feat = torch.sum(scored_x, dim=1)  # 加权求和
        
        # feat形状是(batch_size, 2 * hidden_size)
        logits = self.decoder1(feat)
        logits = self.decoder2(logits)
        # logits形状是(batch_size, 2)
        return logits


class BiGRU_Attention(nn.Module):
    def __init__(self, args):
        super(BiGRU_Attention, self).__init__()
        self.args = args
        
        self.embedding, self.embedding2 = get_embeddings(args)
        
        # bidirectional设为True即得到双向循环神经网络
        self.encoder = nn.GRU(input_size=args.embedding_dim,
                              hidden_size=args.hidden_size,
                              num_layers=args.layers_num,
                              batch_first=False,
                              bidirectional=True)
        # 初始时间步和最终时间步的隐藏状态作为全连接层输入
        self.w_omega = nn.Parameter(torch.Tensor(args.hidden_size * 2, args.hidden_size * 2))
        self.u_omega = nn.Parameter(torch.Tensor(args.hidden_size * 2, 1))
        self.decoder = nn.Linear(2 * args.hidden_size, args.class_num)
        
        nn.init.uniform_(self.w_omega, -0.1, 0.1)
        nn.init.uniform_(self.u_omega, -0.1, 0.1)
    
    def forward(self, x):
        """
        :param x: 输入，形状为(batch_size,seq_len)
        """
        x.t_()  # (seq_len,batch_size)
        embeddings = self.embedding(x)
        outputs, _ = self.encoder(embeddings)  # output, h
        x = outputs.permute(1, 0, 2)  # 现在x形状为(batch_size, seq_len, 2 * num_hiddens)，2表示双向
        
        # Attention过程
        u = torch.tanh(torch.matmul(x, self.w_omega))
        # u形状是(batch_size, seq_len, 2 * num_hiddens)
        att = torch.matmul(u, self.u_omega)
        # att形状是(batch_size, seq_len, 1)
        att_score = F.softmax(att, dim=1)
        # att_score形状仍为(batch_size, seq_len, 1)
        scored_x = x * att_score
        # scored_x形状是(batch_size, seq_len, 2 * num_hiddens)
        # Attention过程结束
        
        feat = torch.sum(scored_x, dim=1)  # 加权求和
        # feat形状是(batch_size, 2 * num_hiddens)
        logits = self.decoder(feat)
        # logits形状是(batch_size, 2)
        return logits


class FastText(nn.Module):
    def __init__(self, args):
        super(FastText, self).__init__()
        self.args = args
        embedding,embedding2 = get_embeddings(args)
        self.embedding_word = embedding
        self.embedding_bigram = nn.Embedding(args.ngram_vocabulary_size, args.embedding_dim,
                                             padding_idx=args.ngram_vocabulary_size - 1)
        self.embedding_trigram = nn.Embedding(args.ngram_vocabulary_size, args.embedding_dim,
                                              padding_idx=args.ngram_vocabulary_size - 1)

        self.dropout = nn.Dropout(args.dropout)
        self.fc1 = nn.Linear(args.embedding_dim * 3, args.hidden_size)
        self.fc2 = nn.Linear(args.hidden_size, args.class_num)
    
    def forward(self, x):
        """
        :param x: 输入，形状为(batch_size,seq_len)
        """
        embed_bow = self.embedding_word(x)
        embed_bigram = self.embedding_bigram(x)
        embed_trigram = self.embedding_trigram(x)

        # 将bow和bag-of-ngrams的结果拼接（或者直接相加，原文应该是直接相加，效果几乎一样）
        logits = torch.cat((embed_bow, embed_bigram, embed_trigram), -1)
        # 取均值
        logits = logits.mean(dim=1)
        logits = self.dropout(logits)
        logits = self.fc1(logits)
        logits = self.fc2(logits)
        logits = F.relu(logits)

        return logits