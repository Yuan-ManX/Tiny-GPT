
######################################## 3.构建数据集和GPT模型 ##################################################

import json
import torch
import torch.utils.data as Data
from torch import nn, optim
import numpy as np
import random
import time
from tqdm import tqdm


device = torch.device("cuda:0") # 指定用哪块gpu
dict_datas = json.load(open('dict_datas.json', 'r')) # 读取json文件，包含word2id字典和id2word列表
word2id, id2word = dict_datas['word2id'], dict_datas['id2word'] # word2id字典和id2word列表
vocab_size = len(word2id) # 词典有多少个字  (7548字)
max_pos = 300  # 一段话最多300个字
d_model = 768  # Embedding Size
d_ff = 2048  # FeedForward dimension
d_k = d_v = 64  # dimension of K(=Q), V
n_layers = 6  # number of Encoder of Decoder Layer
n_heads = 8  # number of heads in Multi-Head Attention
CLIP = 1

print('数据集字典共: %d 字'%vocab_size)


# 输入读取完的dataset.txt文件列表，输出将制表符替换成<sep>的列表
def make_data(datas):
    train_datas = []
    for data in datas:
        data = data.strip() # 除去空格
        train_data = [i if i != '\t' else "<sep>" for i in data] + ['<sep>'] # 将制表符替换成<sep>
        train_datas.append(train_data)

    return train_datas


######################################## 数据集 ##################################################

# 定义数据集
class GPTDataset(Data.Dataset):
    def __init__(self, datas):
        self.datas = datas # 这个列表是先取dataset.txt的一部分，进行处理(把制表符替换成<sep>)，返回处理后的列表，再把列表里的每一个字替换成对应的id号码。

    def __getitem__(self, item): # 从上面的列表中按item索引取出一个数据(一段对话)，构造gpt的输入和输出，打包成字典返回
        data = self.datas[item]  # 从上面的列表中按item索引取出一个数据(一段对话)
        decoder_input = data[:-1] # 输入和输出错开一个位置
        decoder_output = data[1:]

        decoder_input_len = len(decoder_input)    # 这个句子的长度，其实输入和输出长度是一样的
        decoder_output_len = len(decoder_output)

        return {"decoder_input": decoder_input, "decoder_input_len": decoder_input_len,
                "decoder_output": decoder_output, "decoder_output_len": decoder_output_len}

    def __len__(self):
        return len(self.datas)

    # 这个方法会作为DataLoader的collate_fn的参数。猜测是因为如果不写这个，torch会调用默认的collate_fn，也就是把这个batch列表的数据转为torch矩阵，但是这里batch内每个数据长度都不一样，无法直接转为矩阵，就会报错。
    def padding_batch(self, batch): # 接收getitem方法返回的batch，
        decoder_input_lens = [d["decoder_input_len"] for d in batch]    # 取出batch里面每一个输入数据(每一段话)的长度
        decoder_output_lens = [d["decoder_output_len"] for d in batch]  # 取出batch里面每一个输出数据(每一段话)的长度

        decoder_input_maxlen = max(decoder_input_lens)    # batch里面一段话的最大长度
        decoder_output_maxlen = max(decoder_output_lens)

        for d in batch: # 对当前batch的每一个decoder_input和decoder_output数据填充"<pad>"，填充到和batch里面的有的最大长度为止
            d["decoder_input"].extend([word2id["<pad>"]] * (decoder_input_maxlen - d["decoder_input_len"]))
            d["decoder_output"].extend([word2id["<pad>"]] * (decoder_output_maxlen - d["decoder_output_len"]))
        decoder_inputs = torch.tensor([d["decoder_input"] for d in batch], dtype=torch.long) #转type
        decoder_outputs = torch.tensor([d["decoder_output"] for d in batch], dtype=torch.long)

        return decoder_inputs, decoder_outputs # 形状[b,decoder_input_maxlen], [b,decoder_output_maxlen]  type为torch.long


######################################## 模型构建 ##################################################

# 把数据里面<pad>对应的字符给mask掉，让后面Q和K相似度矩阵的softmax中这些pad都为0，就不会被后续的V考虑
def get_attn_pad_mask(seq_q, seq_k): # 形状都是[b, tgt_len <300]

    batch_size, len_q = seq_q.size()  # len_q = len_k = tgt_len
    batch_size, len_k = seq_k.size()
    # eq(zero) is PAD token.就是把数据里面<pad>对应的字符给mask掉，让后面Q和K的softmax不考虑这些<pad>
    pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)  # [b, 1, tgt_len], id为0(也就是<pad>的id)的位置为True，其他位置为False。后面会把Ture位置的mask掉
    return pad_attn_mask.expand(batch_size, len_q, len_k)  # [b, tgt_len, tgt_len]


# 上三角矩阵mask，这是因为用当前信息预测下一个字的时候，是无法看到后续的信息的。
def get_attn_subsequence_mask(seq): #seq: [b, tgt_len]

    attn_shape = [seq.size(0), seq.size(1), seq.size(1)] # [b, tgt_len, tgt_len]
    subsequence_mask = np.triu(np.ones(attn_shape), k=1)  # Upper triangular matrix(上三角矩阵)
    subsequence_mask = torch.from_numpy(subsequence_mask).byte()
    subsequence_mask = subsequence_mask.to(device)
    return subsequence_mask  # [b, tgt_len, tgt_len] 上三角矩阵,下0上1,dtype=torch.uint8


# 计算Q和K的相似度矩阵，然后乘V。
class ScaledDotProductAttention(nn.Module): 
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    # 前三者形状相同[b, n_heads, tgt_len, d_k=64]，attn_mask:[b, n_heads, tgt_len, tgt_len]
    def forward(self, Q, K, V, attn_mask): 

        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k)  # Q和K的相似度矩阵scores : [b, n_heads, tgt_len, tgt_len]
        scores.masked_fill_(attn_mask, -1e9)  # Fills elements of self tensor with value where mask is True.
        # 就是scores矩阵里面和attn_mask=1对应位置的元素全部替换成-1e9，使其在下一步的softmax中变为0

        attn = nn.Softmax(dim=-1)(scores) # [b, n_heads, tgt_len, tgt_len]
        context = torch.matmul(attn, V)  # [b, n_heads, tgt_len, d_v]
        return context, attn

# 多头注意力机制
class MultiHeadAttention(nn.Module): 
    def __init__(self):
        super(MultiHeadAttention, self).__init__()
        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=False)  # d_model=768 ,  d_v = d_k = 64 ,  n_heads=8
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=False)
        self.fc = nn.Linear(n_heads * d_v, d_model, bias=False)
        self.layernorm = nn.LayerNorm(d_model)

    # 前三者形状相同，都是[b, tgt_len, d_model]  , attn_mask: [b, tgt_len, tgt_len]
    def forward(self, input_Q, input_K, input_V, attn_mask): 

        residual, batch_size = input_Q, input_Q.size(0)  #
        # [b, tgt_len, d_model] --> [b, tgt_len, d_k * n_heads] -split-> (b, tgt_len, n_heads, d_k) -trans-> (b, n_heads, tgt_len, d_k)
        Q = self.W_Q(input_Q).view(batch_size, -1, n_heads, d_k).transpose(1, 2)  # Q: [b, n_heads, tgt_len, d_k=64]
        K = self.W_K(input_K).view(batch_size, -1, n_heads, d_k).transpose(1, 2)  # K: [b, n_heads, tgt_len, d_k=64]
        V = self.W_V(input_V).view(batch_size, -1, n_heads, d_v).transpose(1, 2)  # V: [b, n_heads, tgt_len, d_v=64]

        attn_mask = attn_mask.unsqueeze(1).repeat(1, n_heads, 1, 1)  # 添加n_heads维度并复制。attn_mask : [b, n_heads, tgt_len, tgt_len]

        context, attn = ScaledDotProductAttention()(Q, K, V, attn_mask)  # 参考图解，context形状[b, n_heads, tgt_len, d_v]
        context = context.transpose(1, 2).reshape(batch_size, -1, n_heads * d_v)  # context: [b, tgt_len, n_heads * d_v]
        output = self.fc(context)  # [batch_size, tgt_len, d_model]
        return self.layernorm(output + residual), attn


# [b,tgt_len,d_model] -> [b,tgt_len,d_model]     输入和输出形状不变
class PoswiseFeedForwardNet(nn.Module):   
    def __init__(self):
        super(PoswiseFeedForwardNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=False),
            nn.ReLU(),
            nn.Linear(d_ff, d_model, bias=False)
        )
        self.layernorm = nn.LayerNorm(d_model)

    def forward(self, inputs):
        '''
        inputs: [batch_size, seq_len, d_model]
        '''
        residual = inputs
        output = self.fc(inputs)
        return self.layernorm(output + residual)  # [batch_size, seq_len, d_model]


# 解码器模块
class DecoderLayer(nn.Module):
    def __init__(self):
        super(DecoderLayer, self).__init__()
        self.dec_self_attn = MultiHeadAttention() # 多头注意力
        # self.dec_enc_attn = MultiHeadAttention()
        self.pos_ffn = PoswiseFeedForwardNet()

    # dec_inputs: [b, tgt_len, d_model]    dec_self_attn_mask: [b, tgt_len, tgt_len]
    def forward(self, dec_inputs, dec_self_attn_mask): 

        #dec_outputs: [b, tgt_len, d_model], dec_self_attn: [b, n_heads, tgt_len, tgt_len]
        dec_outputs, dec_self_attn = self.dec_self_attn(dec_inputs, dec_inputs, dec_inputs, dec_self_attn_mask)

        dec_outputs = self.pos_ffn(dec_outputs)  # [b, tgt_len, d_model]
        return dec_outputs, dec_self_attn  # [b, tgt_len, d_model] , [b, n_heads, tgt_len, tgt_len]


# 编码器模块
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.tgt_emb = nn.Embedding(vocab_size, d_model)  # 以矩阵形式抽取一行，会比直接用mlp高效。因为mlp会多很多无用运算      emb矩阵形状(vocab_size,768)
        self.pos_emb = nn.Embedding(max_pos, d_model)     # 可学习的位置编码    emb矩阵形状(300,768)
        self.layers = nn.ModuleList([DecoderLayer() for _ in range(n_layers)])

    def forward(self, dec_inputs): # 输入dec_inputs形状[b,tgt_len]

        seq_len = dec_inputs.size(1) # tgt_len ，表示batch内最大长度，不会超过300
        pos = torch.arange(seq_len, dtype=torch.long, device=device) # 给位编码准备的值，[0,1,2,3,...,seq_len-1]
        pos = pos.unsqueeze(0).expand_as(dec_inputs)  # [tgt_len] -> [b, tgt_len]

        dec_outputs = self.tgt_emb(dec_inputs) + self.pos_emb(pos)  # [b, tgt_len, d_model=768]
        # 此时的dec_outputs就包含了这段话的信息和位编码的信息

        dec_self_attn_pad_mask = get_attn_pad_mask(dec_inputs, dec_inputs)  # [b, tgt_len, tgt_len]  把<pad>给mask掉
        dec_self_attn_subsequence_mask = get_attn_subsequence_mask(dec_inputs)  # [b, tgt_len, tgt_len] 上三角矩阵
        dec_self_attn_mask = torch.gt((dec_self_attn_pad_mask + dec_self_attn_subsequence_mask), 0)  # [b, tgt_len, tgt_len] 矩阵大于0的全为1，否则为0

        dec_self_attns = []
        for layer in self.layers:
            # dec_outputs: [b, tgt_len, d_model], dec_self_attn: [b, n_heads, tgt_len, tgt_len], dec_enc_attn: [b, h_heads, tgt_len, src_len]
            dec_outputs, dec_self_attn = layer(dec_outputs, dec_self_attn_mask)
            dec_self_attns.append(dec_self_attn)

        return dec_outputs, dec_self_attns


# Tiny-GPT模型
class GPT(nn.Module):
    def __init__(self):
        super(GPT, self).__init__()
        self.decoder = Decoder()
        self.projection = nn.Linear(d_model, vocab_size) # 768->vocab_size,也就是把最后的隐藏层节点768投影到字典个数的节点上

    def forward(self, dec_inputs): # 输入dec_inputs形状[b,tgt_len]         tgt_len<=300 (tgt_len是batch内最大长度)
        dec_outputs, dec_self_attns = self.decoder(dec_inputs)  # dec_outpus: [b, tgt_len, d_model=768], dec_self_attns: [n_layers, b, n_heads, tgt_len, tgt_len]
        dec_logits = self.projection(dec_outputs)  # dec_logits: [b, tgt_len, vocab_size]
        return dec_logits.view(-1, dec_logits.size(-1)), dec_self_attns    # 左边那个输出形状[b *tgt_len,vocab_size]

    def greedy_decoder(self, dec_input): # dec_input :[1,tgt_len]   此时tgt_len就是句子长度

        terminal = False
        start_dec_len = len(dec_input[0])
        # 一直预测下一个单词，直到预测到"<sep>"结束，如果一直不到"<sep>"，则根据长度退出循环，并在最后加上”<sep>“字符
        while not terminal:
            if len(dec_input[0]) - start_dec_len > 100:
                next_symbol = word2id['<sep>']
                dec_input = torch.cat(
                    [dec_input.detach(), torch.tensor([[next_symbol]], dtype=dec_input.dtype, device=device)], -1)
                break

            # forward
            dec_outputs, _ = self.decoder(dec_input)
            projected = self.projection(dec_outputs) #[1, tgt_len, vocab_size]

            prob = projected.squeeze(0).max(dim=-1, keepdim=False)[1] # [1]是索引，我们只要索引就行了。[0]是具体概率数值，不需要       形状[tgt_len]
            next_word = prob.data[-1] # 最后一个字对应的id
            next_symbol = next_word
            if next_symbol == word2id["<sep>"]: # 如果预测到"<sep>"则结束
                terminal = True

            dec_input = torch.cat(
                [dec_input.detach(), torch.tensor([[next_symbol]], dtype=dec_input.dtype, device=device)], -1)

        return dec_input   # [1,tgt_len+n]  因为多了n个预测的字

    # 具有随机性
    def random_decoder(self, dec_input,top_n): # dec_input :[1,tgt_len]   此时tgt_len就是句子长度

        terminal = False
        start_dec_len = len(dec_input[0])
        # 一直预测下一个单词，直到预测到"<sep>"结束，如果一直不到"<sep>"，则根据长度退出循环，并在最后加上”<sep>“字符
        while not terminal:
            if len(dec_input[0]) - start_dec_len > 100:
                next_symbol = word2id['<sep>']
                dec_input = torch.cat(
                    [dec_input.detach(), torch.tensor([[next_symbol]], dtype=dec_input.dtype, device=device)], -1)
                break

            # forward
            with torch.no_grad():
                dec_outputs, _ = self.decoder(dec_input)
                projected = self.projection(dec_outputs) # [1, tgt_len, vocab_size]

            # prob = projected.squeeze(0).max(dim=-1, keepdim=False)[1] #[1]是索引，我们只要索引就行了。[0]是具体概率数值，不需要       形状[tgt_len]
            # next_word = prob.data[-1] #最后一个字对应的id

            a = projected.to('cpu') #[1, tgt_len, vocab_size]
            b = a.squeeze(0)[-1]  #  [vocab_size]
            c, idx1 = torch.sort(b, descending=True)  #c是预测的概率大小 ， idx1是对应的索引
            c = np.array(c[:top_n])**2  # 取前n个概率最大的
            idx1 = np.array(idx1[:top_n])


            sum = 0 # 总概率值
            for i in c:
                sum += i #前top_n个字的每一个的概率值

            d = sum * random.uniform(0, 1) # 随机数

            for i, j in enumerate(c):
                d -= j  # 随机数减去概率值
                if d <= 0:
                    next_word = idx1[i] # 当前预测的字对应的id号码
                    break

            next_symbol = next_word
            if next_symbol == word2id["<sep>"]: # 如果预测到"<sep>"则结束
                terminal = True

            dec_input = torch.cat(
                [dec_input.detach(), torch.tensor([[next_symbol]], dtype=dec_input.dtype, device=device)], -1)

        return dec_input   # [1,tgt_len+n]  因为多了n个预测的字

    # sentence是人输入的字符串  [n]   n是句子有多少个字符
    def answer(self, sentence): 
        # 把原始句子替换成对应id，\t替换成”<sep>“的id，get(word, 1)是找word对应值，如果找不到对应键则默认返回1，1对应<ukn>表示未知字符
        dec_input = [word2id.get(word, 1) if word != '\t' else word2id['<sep>'] for word in sentence]  #句子对应的每个字的id号
        dec_input = torch.tensor(dec_input, dtype=torch.long, device=device).unsqueeze(0) #[n] -> [1,n]  转type，并放入指定设备

        # output = self.greedy_decoder(dec_input).squeeze(0)  # [1,n] -> [1,n+m]   #m是新预测出的字数，n是本来输入的问题

        output = self.random_decoder(dec_input,top_n=3).squeeze(0)  # [1,n] -> [1,n+m]   #m是新预测出的字数，n是本来输入的问题

        out = [id2word[int(id)] for id in output]  #把id列表转为对应的字符列表
        # 统计"<sep>"字符在结果中的索引
        sep_indexs = []
        for i in range(len(out)):
            if out[i] == "<sep>":
                sep_indexs.append(i)

        # 取最后两个sep中间的内容作为回答，前面的是输入的问题，可以直接丢掉，不需要显示
        answer = out[sep_indexs[-2] + 1:-1]

        answer = "".join(answer)

        return answer


