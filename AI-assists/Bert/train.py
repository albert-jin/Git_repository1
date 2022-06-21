# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 23:55:43 2020

@author: zhaog
"""
import os
import torch
from torch.utils.data import DataLoader
from data import DataPrecessForSentence
from utils import train, validate
from transformers import BertTokenizer
from model import BertModel
from transformers.optimization import AdamW
import matplotlib.pyplot as plt

def main(train_file, dev_file, target_dir, 
         epochs=10,
         batch_size=32,
         lr=2e-05,
         patience=3,
         max_grad_norm=10.0,
         checkpoint=None):
    bert_tokenizer = BertTokenizer.from_pretrained('bert-base-chinese', do_lower_case=True)
    device = torch.device("cuda")
    print(20 * "=", " Preparing for training ", 20 * "=")
    # 保存模型的路径
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    # -------------------- Data loading ------------------- #
    print("\t* Loading training data...")
    train_data = DataPrecessForSentence(bert_tokenizer, train_file)
    train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
    print("\t* Loading validation data...")
    dev_data = DataPrecessForSentence(bert_tokenizer,dev_file)
    dev_loader = DataLoader(dev_data, shuffle=True, batch_size=batch_size)
    # -------------------- Model definition ------------------- #
    print("\t* Building model...")
    model = BertModel().to(device)
    # -------------------- Preparation for training  ------------------- #
    # 待优化的参数
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
            {
                    'params':[p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
                    'weight_decay':0.01
            },
            {
                    'params':[p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
                    'weight_decay':0.0
            }
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=lr)
    #optimizer = torch.optim.Adam(optimizer_grouped_parameters, lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", 
                                                           factor=0.85, patience=0)
    best_score = 0.0
    start_epoch = 1
    # Data for loss curves plot
    epochs_count = []
    train_losses = []
    valid_losses = []
    # Continuing training from a checkpoint if one was given as argument
    if checkpoint:
        checkpoint = torch.load(checkpoint)
        start_epoch = checkpoint["epoch"] + 1
        best_score = checkpoint["best_score"]
        print("\t* Training will continue on existing model from epoch {}...".format(start_epoch))
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        epochs_count = checkpoint["epochs_count"]
        train_losses = checkpoint["train_losses"]
        valid_losses = checkpoint["valid_losses"]
     # Compute loss and accuracy before starting (or resuming) training.
    _, valid_loss, valid_accuracy, auc = validate(model, dev_loader)
    print("\t* Validation loss before training: {:.4f}, accuracy: {:.4f}%, auc: {:.4f}".format(valid_loss, (valid_accuracy*100), auc))
    valid_accuracy_list = [valid_accuracy]
    train_accuracy_list = []
    valid_auc_list = [auc]
    # -------------------- Training epochs ------------------- #
    print("\n", 20 * "=", "Training Bert model on device: {}".format(device), 20 * "=")
    patience_counter = 0
    for epoch in range(start_epoch, epochs + 1):
        epochs_count.append(epoch)
        print("* Training epoch {}:".format(epoch))
        epoch_time, epoch_loss, epoch_accuracy = train(model, train_loader, optimizer, epoch, max_grad_norm)
        train_losses.append(epoch_loss)
        print("-> Training time: {:.4f}s, loss = {:.4f}, accuracy: {:.4f}%"
              .format(epoch_time, epoch_loss, (epoch_accuracy*100)))
        train_accuracy_list.append(epoch_accuracy)
        print("* Validation for epoch {}:".format(epoch))
        epoch_time, epoch_loss, epoch_accuracy , epoch_auc= validate(model, dev_loader)
        valid_losses.append(epoch_loss)
        print("-> Valid. time: {:.4f}s, loss: {:.4f}, accuracy: {:.4f}%, auc: {:.4f}\n"
              .format(epoch_time, epoch_loss, (epoch_accuracy*100), epoch_auc))
        valid_accuracy_list.append(epoch_accuracy)
        valid_auc_list.append(epoch_auc)
        # Update the optimizer's learning rate with the scheduler.
        scheduler.step(epoch_accuracy)
        # Early stopping on validation accuracy.
        if epoch_accuracy < best_score:
            patience_counter += 1
        else:
            best_score = epoch_accuracy
            patience_counter = 0
            torch.save({"epoch": epoch, 
                        "model": model.state_dict(),
                        "best_score": best_score,
                        "epochs_count": epochs_count,
                        "train_losses": train_losses,
                        "valid_losses": valid_losses},
                        os.path.join(target_dir, "best.pth.tar"))
        if patience_counter >= patience:
            print("-> Early stopping: patience limit reached, stopping...")
            break
    #  train_losses valid_losses   valid_accuracy_list train_accuracy_list valid_auc_list 这个五个可以做几张图:

    #training-loss随epoch的变化
    plt.subplot(2,2,1)
    plt.plot(range(1, len(train_losses)+1), train_losses, '.-')  # , label='Train loss', markevery=20
    plt.xlabel('Training Loss vs. epoches')
    plt.ylabel('Training Loss')
    #valid-loss随epoch的变化
    plt.subplot(2, 2, 2)
    plt.plot(range(1, len(valid_losses)+1), valid_losses, 'o-')  # , label='Train loss', markevery=20
    plt.xlabel('Valid Loss vs. epoches')
    plt.ylabel('Valid Loss')
    #train 数据集的准确率随epoch的变化
    plt.subplot(2, 2, 3)
    plt.plot(range(1, len(train_accuracy_list)+1), train_accuracy_list, '*-')  # , label='Train loss', markevery=20
    plt.xlabel('Training Accuracy vs. epoches')
    plt.ylabel('Training Accuracy')
    #valid数据集的准确率随epoch的变化
    plt.subplot(2, 2, 4)
    plt.plot(range(len(valid_accuracy_list)), valid_accuracy_list, '.-')  # , label='Train loss', markevery=20
    plt.xlabel('Valid Accuracy vs. epoches')
    plt.ylabel('Valid Accuracy')
    plt.legend()
    plt.show()
    #valid数据集的AUC率随epoch的变化
    plt.subplot(1, 1, 1)
    plt.plot(range(len(valid_auc_list)), valid_auc_list, '*-')  # , label='Train loss', markevery=20
    plt.xlabel('Valid AUC vs. epoches')
    plt.ylabel('Valid AUC')
    plt.legend()
    plt.show()
    #valid数据集的AUC率随epoch的变化
    plt.subplot(1, 1, 1)
    plt.plot(range(len(train_accuracy_list)), train_accuracy_list, '*-')  # , label='Train loss', markevery=20
    plt.plot(range(1,len(valid_accuracy_list)), valid_accuracy_list[1:], 'o-')  # , label='Train loss', markevery=20
    plt.xlabel('Training&Valid Accuracy vs. epoches')
    plt.ylabel('Training&Valid Accuracy')
    plt.legend()
    plt.show()


    
if __name__ == "__main__":
    main("../data/LCQMC_train.csv", "../data/LCQMC_dev.csv", "models")
