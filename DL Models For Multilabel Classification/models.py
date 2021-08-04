import torch
import torch.nn as nn
import torch.nn.functional as F


class TextCNN(nn.Module):
    def __init__(self, args):
        super(TextCNN, self).__init__()
        self.args = args
        
        if self.args.multilabel == True:
            labels_num = len(args.labels)
        else:
            labels_num = 2
        chanel_num = 1
        self.embedding = args.embedding
        
        self.convs = nn.ModuleList(
            [nn.Conv2d(chanel_num, args.filter_num, (size, args.embedding_dim)) for size in args.filter_sizes])
        self.dropout = nn.Dropout(args.dropout)
        self.fc = nn.Linear(len(args.filter_sizes) * args.filter_num, labels_num)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        """
        :param x: 输入，形状为(batch_size,seq_len)
        """
        x = self.embedding(x)
        x = x.unsqueeze(1)
        
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs]
        x = [F.max_pool1d(item, item.size(2)).squeeze(2) for item in x]
        x = torch.cat(x, 1)
        x = self.dropout(x)
        logits = self.fc(x)
        if self.args.multilabel:
            return self.sigmoid(logits)
        return logits


class BiLSTM_Attention(nn.Module):
    def __init__(self, args):
        super(BiLSTM_Attention, self).__init__()
        self.args = args
        
        if self.args.multilabel == True:
            labels_num = len(args.labels)
        else:
            labels_num = 2
        self.embedding = args.embedding
        
        # bidirectional设为True即得到双向循环神经网络
        self.encoder = nn.LSTM(input_size=args.embedding_dim,
                               hidden_size=args.hidden_size,
                               num_layers=args.layers_num,
                               batch_first=False,
                               bidirectional=True)
        # 初始时间步和最终时间步的隐藏状态作为全连接层输入
        self.w_omega = nn.Parameter(torch.Tensor(args.hidden_size * 2, args.hidden_size * 2))
        self.u_omega = nn.Parameter(torch.Tensor(args.hidden_size * 2, 1))
        self.decoder = nn.Linear(2 * args.hidden_size, labels_num)
        self.sigmoid = nn.Sigmoid()
        
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
        logits = self.decoder(feat)
        # logits形状是(batch_size, labels_num)
        if self.args.multilabel:
            return self.sigmoid(logits)
        return logits


class BiGRU_Attention(nn.Module):
    def __init__(self, args):
        super(BiGRU_Attention, self).__init__()
        self.args = args
        
        if self.args.multilabel == True:
            labels_num = len(args.labels)
        else:
            labels_num = 2
        self.embedding = args.embedding
        
        # bidirectional设为True即得到双向循环神经网络
        self.encoder = nn.GRU(input_size=args.embedding_dim,
                              hidden_size=args.hidden_size,
                              num_layers=args.layers_num,
                              batch_first=False,
                              bidirectional=True)
        # 初始时间步和最终时间步的隐藏状态作为全连接层输入
        self.w_omega = nn.Parameter(torch.Tensor(args.hidden_size * 2, args.hidden_size * 2))
        self.u_omega = nn.Parameter(torch.Tensor(args.hidden_size * 2, 1))
        self.decoder = nn.Linear(2 * args.hidden_size, labels_num)
        self.sigmoid = nn.Sigmoid()
        
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
        logits = self.decoder(feat)
        # logits形状是(batch_size, labels_num)
        if self.args.multilabel:
            return self.sigmoid(logits)
        return logits


class FastText(nn.Module):
    def __init__(self, args):
        super(FastText, self).__init__()
        self.args = args
        
        if self.args.multilabel == True:
            labels_num = len(args.labels)
        else:
            labels_num = 2
        self.embedding_word = args.embedding
        self.embedding_bigram = nn.Embedding(args.ngram_vocabulary_size, args.embedding_dim,
                                             padding_idx=args.ngram_vocabulary_size - 1)
        self.embedding_trigram = nn.Embedding(args.ngram_vocabulary_size, args.embedding_dim,
                                              padding_idx=args.ngram_vocabulary_size - 1)
        
        self.dropout = nn.Dropout(args.dropout)
        self.fc1 = nn.Linear(args.embedding_dim * 3, args.hidden_size)
        self.fc2 = nn.Linear(args.hidden_size, labels_num)
        self.sigmoid = nn.Sigmoid()
    
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
        
        if self.args.multilabel:
            return self.sigmoid(logits)
        return logits
