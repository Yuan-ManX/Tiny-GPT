
######################################## 5.测试模型效果 ##################################################

import torch
from model_GPT import GPT


if __name__ == '__main__':

    # 加载模型
    device = torch.device('cuda:0')
    model = GPT().to(device)
    model.load_state_dict(torch.load(r'weights\XXXXX.pt'))

    # 推理模式
    model.eval() 

    # 初始输入是空，每次加上后面的对话信息
    sentence = ''

    while True:
        sentence = ''
        temp_sentence = input("你好:")
        sentence += (temp_sentence + '\t')

        if len(sentence) > 200:
            # 由于该模型输入最大长度为300，避免长度超出限制长度过长需要进行裁剪
            t_index = sentence.find('\t')
            sentence = sentence[t_index + 1:]

        print("Tiny-GPT:", model.answer(sentence))

