# -*- coding:utf-8 -*-
# @Time: 2021/5/30 22:20
# @Author: duiya duiyady@163.com

from bert4keras.tokenizers import Tokenizer
import os
import jieba
import numpy as np
import pickle
import json
jieba.initialize()

pre_model = 'NEZHA-Base'
path_drive = '../base_model/' + pre_model
config_path = os.path.join(path_drive, 'bert_config.json')
checkpoint_path = os.path.join(path_drive, 'model.ckpt-900000')
dict_path = os.path.join(path_drive, 'vocab.txt')

tokenizer = Tokenizer(
    dict_path,
    do_lower_case=True,
    pre_tokenize=lambda s: jieba.lcut(s, HMM=False)
)
# 将文本转换为token, 分词
def text_token(text):
    tokens = [
        tokenizer._token_translate.get(token) or token
        for token in tokenizer._tokenize(text)
    ]
    return tokens


variants = [
    u'短短匹配A类',
    u'短短匹配B类',
    u'短长匹配A类',
    u'短长匹配B类',
    u'长长匹配A类',
    u'长长匹配B类',
]

fs = []
for i, var in enumerate(variants):

    tmp_path = '../datasets/sohu2021_open_data_clean/%s/train.txt' % var
    tmp_path = tmp_path.encode("utf-8")
    fs.append(tmp_path)
    tmp_path = '../datasets/sohu2021_open_data_clean/%s/valid.txt' % var
    tmp_path = tmp_path.encode("utf-8")
    fs.append(tmp_path)


    tmp_path = '../datasets/round2/%s.txt' % var
    tmp_path = tmp_path.encode("utf-8")
    fs.append(tmp_path)

    tmp_path = '../datasets/round3/%s/train.txt' % var
    tmp_path = tmp_path.encode("utf-8")
    fs.append(tmp_path)

    tmp_path = '../datasets/rematch/%s/train.txt' % var
    tmp_path = tmp_path.encode("utf-8")
    fs.append(tmp_path)
    tmp_path = '../datasets/rematch/%s/valid.txt' % var
    tmp_path = tmp_path.encode("utf-8")
    fs.append(tmp_path)

seq_count = 0
word_count = {}



for f in fs:
    print(f)
    with open(f, encoding="utf8") as f:
        for l in f:
            l = json.loads(l)
            source = l['source']
            target = l['target']

            source_token = text_token(source)
            target_token = text_token(target)
            source_token_ids = tokenizer.tokens_to_ids(source_token)
            target_token_ids = tokenizer.tokens_to_ids(target_token)

            now = []

            for val in source_token_ids:
                if val not in now:
                    now.append(val)
                    if val not in word_count.keys():
                        word_count[val] = 0
                    word_count[val] += 1
            for val in target_token_ids:
                if val not in now:
                    now.append(val)
                    if val not in word_count.keys():
                        word_count[val] = 0
                    word_count[val] += 1
            seq_count += 1
for key in word_count.keys():
    word_count[key] = np.log(seq_count/(word_count[key]+1))

word_count[tokenizer.token_to_id("[CLS]")] = 0.0
word_count[tokenizer.token_to_id("[UNK]")] = 0.0
word_count[tokenizer.token_to_id("[MASK]")] = 0.0
word_count[tokenizer.token_to_id("[CLS]")] = 0.0
word_count[tokenizer.token_to_id("[SEP]")] = 0.0

tmp_adj = {}

for key in word_count.keys():
    tmp_adj[tokenizer.id_to_token(key)] = word_count[key]

for key in tmp_adj.keys():
    word_count[key] = tmp_adj[key]


for i in range(21127):
    if i not in word_count.keys():
        word_count[i] = 0.0

pickle.dump(word_count, open("./compute_word_idf.pkl", "wb"))

