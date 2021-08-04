import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_pretrained_bert import BertModel, BertTokenizer


class BERT(nn.Module):
    def __init__(self, args):
        super(BERT, self).__init__()
        self.bert = BertModel.from_pretrained(args.bert_pretrained)
        for param in self.bert.parameters():
            param.requires_grad = True
        self.fc = nn.Linear(args.bert_hidden_size, args.class_num)
    
    def forward(self, x):
        """
        :param x: tuple, 第一个元素为语句数据，形状为(batch_size,pad_size), 第二个元素为语句长度，形状为(batch_size),
                  第三个元素为mask, 形状为(batch_size,pad_size)
        :return:
        """
        context = x[0]
        mask = x[2]
        encoder_out, pooled = self.bert(context, attention_mask=mask, output_all_encoded_layers=False)
        logits = self.fc(pooled)
        return logits


class BERT_CNN(nn.Module):
    def __init__(self, args):
        super(BERT_CNN, self).__init__()
        self.args = args
        
        chanel_num = 1
        
        args.filter_sizes = [int(size) for size in args.filter_sizes.split(',')]
        self.bert = BertModel.from_pretrained(args.bert_pretrained)
        for param in self.bert.parameters():
            param.requires_grad = True
        self.convs = nn.ModuleList(
            [nn.Conv2d(chanel_num, args.filter_num, (size, args.bert_hidden_size)) for size in args.filter_sizes])
        self.dropout = nn.Dropout(args.dropout)
        
        self.fc = nn.Linear(len(args.filter_sizes) * args.filter_num, args.class_num)
    
    def forward(self, x):
        """
        :param x: tuple, 第一个元素为语句数据，形状为(batch_size,pad_size), 第二个元素为语句长度，形状为(batch_size),
                  第三个元素为mask, 形状为(batch_size,pad_size)
        """
        context = x[0]
        mask = x[2]
        encoder_out, pooled = self.bert(context, attention_mask=mask, output_all_encoded_layers=False)
        x = encoder_out.unsqueeze(1)
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs]
        x = [F.max_pool1d(item, item.size(2)).squeeze(2) for item in x]
        out = torch.cat(x, 1)
        out = self.dropout(out)
        out = self.fc(out)
        return out


class BERT_RCNN(nn.Module):
    def __init__(self, args):
        super(BERT_RCNN, self).__init__()
        self.bert = BertModel.from_pretrained(args.bert_pretrained)
        for param in self.bert.parameters():
            param.requires_grad = True
        self.lstm = nn.LSTM(input_size=args.bert_hidden_size,
                            hidden_size=args.hidden_size,
                            num_layers=args.layers_num,
                            bidirectional=True, batch_first=False, dropout=args.dropout)
        self.maxpool = nn.MaxPool1d(args.pad_size)
        self.fc = nn.Linear(args.hidden_size * 2 + args.bert_hidden_size, args.class_num)

    def forward(self, x):
        context = x[0]  # 输入的句子
        mask = x[2]  # 对padding部分进行mask，和句子一个size，padding部分用0表示，如：[1, 1, 1, 1, 0, 0]
        encoder_out, text_cls = self.bert(context, attention_mask=mask, output_all_encoded_layers=False)
        out, _ = self.lstm(encoder_out)
        out = torch.cat((encoder_out, out), 2)
        out = F.relu(out)
        out = out.permute(0, 2, 1)
        out = self.maxpool(out).squeeze()
        out = self.fc(out)
        return out


