import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import f1_score, accuracy_score


def train(train_iter, dev_iter, model, args):
    if args.cuda:
        model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    steps = 0
    best_acc = 0
    best_f1 = 0
    best_epoch = 0
    
    # 模型训练
    for epoch in range(0, args.epochs):
        for batch in train_iter:
            model.train()
            
            # 其他任务改此处
            feature, target = batch.rateContent, batch.sentiment
            feature.t_()
            if args.cuda:
                feature, target = feature.cuda(), target.cuda()
            
            optimizer.zero_grad()
            logits = model(feature)
            loss = F.cross_entropy(logits, target)
            loss.backward()
            optimizer.step()
            steps += 1
            corrects = (torch.max(logits, 1)[1].view(target.size()).data == target.data).sum()
            train_acc = 100.0 * corrects / batch.batch_size
            sys.stdout.write(
                "\rEpoch {} Steps: {}/{} - loss: {:.6f}  acc: {:.4f}%({}/{})".format(epoch,
                                                                                     steps,
                                                                                     len(train_iter),
                                                                                     loss.item(),
                                                                                     train_acc,
                                                                                     corrects,
                                                                                     batch.batch_size))
        steps = 0
        
        # 模型验证
        print("\n")
        print("Evaluating Epoch {}".format(epoch))
        model.eval()
        dev_acc, dev_f1 = eval(dev_iter, model, args)
        if dev_acc > best_acc and dev_f1 > best_f1:
            best_acc = dev_acc
            best_f1 = dev_f1
            best_epoch = epoch
            print("Saving the best model, epoch : {} f1 : {:.4f} acc: {:.4f}%".format(epoch,best_f1, best_acc * 100))
            print("-----------------------------------------------------------")
            save(model, args.save_dir, 'best', epoch)
        else:
            print("-----------------------------------------------------------")
            continue
    
    print("Stop Training. best model: epoch : {} f1 : {:.4f} acc: {:.4f}%".format(best_epoch,best_f1, best_acc * 100))


def eval(data_iter, model, args):
    model.eval()
    corrects, avg_loss = 0, 0
    target_value, pred_value = [],[]
    for batch in data_iter:
        # 其他任务改此处
        feature, target = batch.rateContent, batch.sentiment
        feature.t_()
        if args.cuda:
            feature, target = feature.cuda(), target.cuda()
        
        logits = model(feature)
        loss = F.cross_entropy(logits, target)
        avg_loss += loss.item()
        
        if not args.multilabel: # 若为单标签分类任务，则将各个batch的预测合并，一起验证
            target_value.extend(target.tolist())
            pred_value.extend(torch.max(logits, 1)[1].view(target.size()).tolist())
        
        f1 = f1_score(y_true=target_value, y_pred=pred_value)
        acc = accuracy_score(y_true=target_value, y_pred=pred_value)
    print("Evaluation - loss: {:.6f}  f1: {:.4f}  acc: {:.4f}%".format(avg_loss, f1, acc * 100))
    return acc, f1


def save(model, save_dir, save_prefix, epoch):
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    save_prefix = os.path.join(save_dir, save_prefix)
    save_path = '{}_epoch_{}.pt'.format(save_prefix, epoch)
    torch.save(model.state_dict(), save_path)
