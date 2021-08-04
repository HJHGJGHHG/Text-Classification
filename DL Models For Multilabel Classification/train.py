import os
import sys
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, classification_report


def booled(batch_label):
    result = []
    for logit in batch_label:
        if logit >= 0.5:
            result.append(1)
        else:
            result.append(0)
    return result


def train(train_iter, dev_iter, model, args):
    if args.cuda:
        model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    steps = 0
    best_acc = 0
    best_f1 = 0
    best_epoch = 0
    best_loss = np.inf
    
    # 模型训练
    for epoch in range(0, args.epochs):
        for batch_num, batch_sample in enumerate(train_iter):
            model.train()
            
            feature, target = batch_sample[args.text_label], batch_sample["labels"]
            # feature 形状为 (bs,seq_len), target 形状为 (bs,labels_num)
            if args.cuda:
                feature, target = feature.cuda(), target.cuda()
            optimizer.zero_grad()
            logits = model(feature)  # 模型的预测输出, 形状为 (bs,labels_num)
            if not args.multilabel:
                target = target.squeeze(dim=1)
                target = target.long()
                loss = F.cross_entropy(logits, target)
                logits_list = torch.max(logits, 1)[1].tolist()
            else:
                loss_func = nn.BCELoss()
                loss = loss_func(logits, target)
                logits_list = [booled(batch_label) for batch_label in logits.tolist()]
            target_list = target.tolist()
            loss.backward()
            optimizer.step()
            steps += 1
            train_acc = accuracy_score(y_true=target_list, y_pred=logits_list)
            sys.stdout.write(
                "\rEpoch {}/{} Steps: {}/{} - loss: {:.6f}  acc: {:.4f}%".format(epoch,
                                                                                 args.epochs - 1,
                                                                                 steps,
                                                                                 len(train_iter),
                                                                                 loss.item(),
                                                                                 train_acc * 100))
        steps = 0
        
        # 模型验证
        print("\n")
        print("Evaluating Epoch {}".format(epoch))
        model.eval()
        if not args.multilabel:
            dev_acc, dev_f1, dev_loss = eval(dev_iter, model, args)
        else:
            dev_acc, dev_f1, dev_report, dev_loss = eval(dev_iter, model, args)
        if dev_loss < best_loss:
            if not args.multilabel:
                best_acc = dev_acc
                best_f1 = dev_f1
                best_epoch = epoch
                best_loss = dev_loss
                print(
                    "Saving the best model, epoch : {} f1 : {:.4f} acc: {:.4f}%".format(epoch, best_f1, best_acc * 100))
                print("-----------------------------------------------------------")
                save(model, args.save_dir, 'best', epoch)
            else:
                best_acc = dev_acc
                best_f1 = dev_f1
                best_report = dev_report
                best_epoch = epoch
                best_loss = dev_loss
                print(
                    "Saving the best model, epoch : {} f1 : {:.4f} acc: {:.4f}%".format(epoch, best_f1, best_acc * 100))
        else:
            print("-----------------------------------------------------------")
            continue
    
    if not args.multilabel:
        print("Stop training. Best model: epoch : {} f1 : {:.4f} acc: {:.4f}%".format(best_epoch, best_f1,
                                                                                      best_acc * 100))
    else:
        print("Stop training. Best model: epoch : {} f1 : {:.4f} acc: {:.4f}%".format(best_epoch, best_f1,
                                                                                      best_acc * 100))
        print("\t\t\tClassification report:\n" + best_report)


def eval(dev_iter, model, args):
    model.eval()
    avg_loss = 0
    target_value, pred_value = [], []
    for batch_num, batch_sample in enumerate(dev_iter):
        feature, target = batch_sample[args.text_label], batch_sample["labels"]
        # feature 形状为 (bs,seq_len), target 形状为 (bs,labels_num)
        if args.cuda:
            feature, target = feature.cuda(), target.cuda()
        
        logits = model(feature)
        
        if not args.multilabel:
            target = target.squeeze(dim=1)
            target = target.long()
            loss = F.cross_entropy(logits, target)
            logits_list = torch.max(logits, 1)[1].tolist()
            avg_loss += loss.item()
        else:
            loss_func = nn.BCELoss()
            loss = loss_func(logits, target).tolist()
            logits_list = [booled(batch_label) for batch_label in logits.tolist()]
            avg_loss += loss
        target_list = target.tolist()
        target_value.extend(target_list)
        pred_value.extend(logits_list)
    
    avg_loss /= len(dev_iter)
    dev_f1 = f1_score(y_true=target_value, y_pred=pred_value, average="micro")
    dev_acc = accuracy_score(y_true=target_value, y_pred=pred_value)
    report = classification_report(y_true=target_value, y_pred=pred_value)
    if not args.multilabel:
        print("Evaluation - loss: {:.6f}  f1: {:.4f}  acc: {:.4f}%".format(avg_loss, dev_f1, dev_acc * 100))
        return dev_acc, dev_f1, avg_loss
    else:
        print("Evaluation - loss: {:.6f} f1 : {:.4f}  acc: {:.4f}%".format(avg_loss, dev_f1, dev_acc * 100))
        return dev_acc, dev_f1, report, avg_loss


def save(model, save_dir, save_prefix, epoch):
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    save_prefix = os.path.join(save_dir, save_prefix)
    save_path = '{}_epoch_{}.pt'.format(save_prefix, epoch)
    torch.save(model.state_dict(), save_path)
