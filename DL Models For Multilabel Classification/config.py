import argparse


def args_initialization():
    parser = argparse.ArgumentParser(description="Text Classifier")
    
    # 基本参数
    parser.add_argument("-path", type=str,
                        default="../data/", help="数据位置")
    parser.add_argument("-lr", type=float, default=0.001, help="初始学习率 [默认: 0.001]")
    parser.add_argument("-epochs", type=int, default=20, help="Epoch数 [默认: 20]")
    parser.add_argument("-early-stop", type=int, default=1000,
                        help="早停的Batch数，即经过多少Batch数没有提升则停止训练 [默认: 1000]")
    parser.add_argument("-batch-size", type=int, default=128, help="Batch Size [默认: 128]")
    parser.add_argument("-save-dir", type=str, default="", help="模型存放位置")
    parser.add_argument("-multilabel", type=bool, default=True, help="是否为多标签分类任务")
    parser.add_argument("-labels", type=list,
                        default=["package", "quality", "price", "service", "logistics", "other"], help="各标签")
    parser.add_argument("-text-label", type=str,
                        default="rateContent", help="文本的标签名")
    parser.add_argument("-class-num", type=list, default=[2,2,2,2,2,2], help="各分类任务的类别数")
    
    parser.add_argument("-dropout", type=float, default=0.5, help="dropout率 [默认: 0.5]")
    parser.add_argument('-max-norm', type=float, default=3.0, help='l2 constraint of parameters [default: 3.0]')
    parser.add_argument("-embedding-dim", type=int, default=300, help="Embedding维数 [默认: 300]")
    
    # TextCNN相关
    parser.add_argument("-filter-num", type=int, default=100, help="各filter大小，即卷积输出通道数 [默认: 100]")
    parser.add_argument("-filter-sizes", type=list, default=[1,3,5],
                        help="与卷积核大小相关，每层filter的卷积核大小为(filter_size,embedding_dim)")
    
    # BiLSTM & BiGRU相关
    parser.add_argument("-hidden-size", type=int, default=50,
                        help="隐藏层大小(即隐藏层节点数量)，输出向量的维度等于隐藏层大小(单向) [默认: 50]")
    parser.add_argument("-layers-num", type=int, default=1, help="循环神经网络层数 [默认: 1]")
    
    # BERT相关
    parser.add_argument("-bert-epochs", type=int, default=3, help="BERT模型的Epoch数 [默认: 3]")
    parser.add_argument("-pad-size", type=int, default=32, help="每句话处理成的长度(短填长切) [默认: 32]")
    parser.add_argument("-bert-pretrained", type=str,
                        default="../../model/bert_pretrained", help="BERT预训练模型位置")
    parser.add_argument("-bert-hidden-size", type=int, default=768, help="BERT隐藏层大小 [默认: 768]")
    parser.add_argument("-bert-lr", type=float, default=5e-5, help="BERT初始学习率 [默认: 5e-5]")
    
    # Embedding相关
    parser.add_argument("-static", type=bool, default=True, help="是否使用静态(预训练)的词向量")
    parser.add_argument("-non-static", type=bool, default=True, help="是否微调静态(预训练)的词向量")
    parser.add_argument("-pretrained-name", type=str, default="sgns.wiki.word",
                        help="预训练词向量文件名")
    parser.add_argument("-pretrained-path", type=str, default="D:/python/pyprojects/NLP/Text Classification/data/",
                        help="预训练词向量文件位置")
    
    # 设备
    parser.add_argument("-device", type=int, default=-1,
                        help="基于何种设备，-1表示CPU [默认: -1]")
    
    args = parser.parse_args(args=[])
    return args
