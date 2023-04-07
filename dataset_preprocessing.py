
######################################## 1.数据预处理-01 ##################################################

# 这个文件是处理原始数据集train.txt文件，生成处理好的dataset.txt文件
# 如果计算资源有限，可以只取一部分数据。会把长度大于300的数据舍弃

# 1.读取原始txt数据文件。
with open('train.txt','r',encoding='utf-8') as f:
    lines = f.readlines()

# 2.处理数据
train_datas = [] # 总的列表，处理好的数据会被添加进去
temp_data = '' # 临时字符串
# 遍历原始txt的每一行
for line in lines: 
    if line!='\n': # 如果不只有换行符，那就是正常句子
        line = line.strip() # 除去句子左右两端的空格
        temp_data+=(line+'\t') # 当前字符串拼接一个制表符
    else: # 如果只有换行符
        train_datas.append(temp_data) # 把之前拼接好的字符串添加进列表
        temp_data='' # 清空临时字符串

# 3.按字符串长度排序
train_datas = sorted(train_datas,key =lambda x:len(x))

# 4.取长度小于300的数据
new_train_datas=[] # 新的总列表，会把处理好的数据添加进去
for train_data in train_datas: # 把长度小于300的数据添加进新列表
    if len(train_data)<300:
        new_train_datas.append(train_data)

# (可选功能)只取原始数据集的一半
# new_train_datas=new_train_datas[::2] # 每隔两部取一条数据，相当于只取了一半数据，如果计算资源不够可以用这个

# 5.写入处理好的数据
with open('dataset.txt','w',encoding='utf-8') as f: # 把处理好的数据写入dataset.txt
    for train_data in new_train_datas:
        f.write(train_data+'\n')

# 6.输出处理结果
print('处理完成,行数：%d'%len(new_train_datas))
