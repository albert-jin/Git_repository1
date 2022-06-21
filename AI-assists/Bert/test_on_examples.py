# -*- coding: utf-8 -*-


import torch
from sys import platform
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from model import BertModelTest
from utils import test_on_examples
from data import DataPrecessForSentence
import pandas as pd


def main(test_file, pretrained_file, batch_size=32):
    device = torch.device("cuda")
    bert_tokenizer = BertTokenizer.from_pretrained('bert-base-chinese', do_lower_case=True)
    print(20 * "=", " Preparing for testing ", 20 * "=")
    if platform == "linux" or platform == "linux2":
        checkpoint = torch.load(pretrained_file)
    else:
        checkpoint = torch.load(pretrained_file, map_location=device)
    print("\t* 加载待预测数据...")
    test_data = DataPrecessForSentence(bert_tokenizer, test_file)
    test_loader = DataLoader(test_data, shuffle=False, batch_size=batch_size)
    print("\t* 构建模型...")
    model = BertModelTest().to(device)
    model.load_state_dict(checkpoint["model"])
    print(20 * "=", " 测试 BERT 模型 中 ", 20 * "=")
    all_labels, all_prob = test_on_examples(model, test_loader)
    df = pd.read_csv(test_file)
    for sentence1, sentence2, label, prob in zip(df['sentence1'].values,df['sentence2'].values,df['label'].values, all_prob):
        print(f'{sentence1} <===> {sentence2} 相似度真实标签:{label}, 模型预测标签: {prob}.')

if __name__ == "__main__":
    main("../data/LCQMC_dev.csv", "models/best.pth.tar")