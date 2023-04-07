
######################################## 4.模型的训练 ##################################################

import json
import torch
import torch.utils.data as Data
from torch import nn, optim
import numpy as np
import time
from tqdm import tqdm
from model_GPT import *


# 把秒数表示为分钟和秒
def epoch_time(start_time, end_time): 
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

# 每一个eopch的训练
def train_step(model,data_loader,optimizer,criterion,clip=1,print_every=None):  
    # 训练模式
    model.train() 

    if print_every == 0:
        print_every = 1

    print_loss_total = 0  # 每次打印都重置，统计一定batch数内(默认10)的loss，每10个batch打印一次

    epoch_loss = 0 # epoch的总loss

    for i, (dec_inputs, dec_outputs) in enumerate(tqdm(data_loader)): #dec_inputs: [b, tgt_len] , dec_outputs: [b, tgt_len]
        optimizer.zero_grad()
        dec_inputs, dec_outputs =dec_inputs.to(device), dec_outputs.to(device)
        # outputs: [batch_size * tgt_len, tgt_vocab_size]       tgt_len<=30

        # with torch.cuda.amp.autocast(): # 半精度训练
        outputs, dec_self_attns = model(dec_inputs)
        loss = criterion(outputs, dec_outputs.view(-1)) # outputs :(b * tgt_len, vocab_size),dec_outputs.view(-1) :(b * tgt_len)       tgt_len<=300


        print_loss_total += loss.item()
        epoch_loss += loss.item()
        loss.backward() # 梯度反向传播


        # 梯度裁剪，防止梯度爆炸。如果loss超过clip，将梯度值缩小为原来的(loss/clip)分之一
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

        optimizer.step() # 更新模型权重

        if print_every and (i + 1) % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('\tCurrent Loss: %.4f' % print_loss_avg)

    return epoch_loss / len(data_loader)

def train(model,data_loader):
    criterion = nn.CrossEntropyLoss(ignore_index=0).to(device) # 损失函数
    optimizer = optim.Adam(model.parameters(), lr=1e-4) # 优化器

    for epoch in range(epochs):
        start_time = time.time()
        train_loss = train_step(model, data_loader, optimizer, criterion, CLIP, print_every=100) # 训练一个epoch
        end_time = time.time()

        torch.save(model.state_dict(), r'weights\01\GPT2-%d.pt'%epoch) # 保存模型权重

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)  # 把秒数表示为分钟和秒
        print(f'Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f}')



def print_num_parameters(model):
    # Find total parameters and trainable parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f'{total_params:,} total parameters.')
    total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad)
    print(f'{total_trainable_params:,} training parameters.')


if __name__ == '__main__':

    # 获取数据
    with open('dataset.txt', 'r', encoding='utf-8') as f:
        datas = f.readlines()

    # print(len(datas))  # 打印数据总量(498491)
    train_data = make_data(datas[::]) # 取dataset.txt的一部分，进行处理(把制表符替换成<sep>)，返回处理后的列表
    # 想测试能不能运行需要先把切片的步数设置为-1，这样可以先把最长的向量组抽出，如果最长的向量组不会爆显存，就可以运行。

    train_num_data = [[word2id[word] for word in line] for line in train_data] # 把每一个字替换成对应的id号码

    batch_size = 22 # 实测4g显存可设置22，6g显存可以设置32
    epochs = 30
    dataset = GPTDataset(train_num_data)
    data_loader = Data.DataLoader(dataset, batch_size=batch_size, collate_fn=dataset.padding_batch) #对每个batch单独调用collate_fn处理，因为batch内的句子长短不一，不能直接用torch的默认方法
    # 每次取出的batch，形状[b,decoder_input_maxlen], [b,decoder_output_maxlen]  type为torch.long。 decoder_output_maxlen是这个batch内最长的长度，每个不同的batch这个值也都不一样
    # 为什么不用shuffle=True，这里是因为如果使用了，则会在batch中随机选取每个数据。随机到的段落有的长有的短，这样短的就要被填充很多个<pad>。如果batch里面段落都差不多长，可以提高训练的效率。

    model = GPT().to(device)

    # model.load_state_dict(torch.load('GPT2.pt'))  #加载权重

    train(model,data_loader)

