import models
import train
import config
import torch

from dataset import get_dataset

def main():
    # 调参
    args = config.args_initialization()
    
    args.save_dir = "D:/python/pyprojects/NLP/Text Classification/model/TextCNN"
    
    args.epochs = 20
    args.static = False
    args.non_static = False
    args.cuda = args.device != -1 and torch.cuda.is_available()
    
    """
    args.multilabel = False
    args.labels = ["sentiment"]
    args.class_num = [2]
    """
    
    args.multilabel = True
    args.labels = ["package", "quality", "price", "service", "logistics", "other"]
    args.class_num = [2]*6
    
    train_iter, test_iter, embedding = get_dataset(args)
    args.embedding = embedding
    args.ngram_vocabulary_size = args.vocabulary_size

    print("============Model:BiLSTM_Attention============")
    model = models.FastText(args)
    """
    if args.cuda:
        torch.cuda.set_device(args.device)
        model = model.cuda()
    """
    print("============Start Training============")
    train.train(train_iter, test_iter, model, args)


if __name__ == '__main__':
    main()