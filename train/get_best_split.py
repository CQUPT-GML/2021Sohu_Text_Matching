# -*- coding:utf-8 -*-
# @Time: 2021/5/31 14:05


import os, json
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np
from bert4keras.models import build_transformer_model
from bert4keras.tokenizers import Tokenizer
from keras.layers import Input, Embedding, Reshape, GlobalAveragePooling1D, Dense
from keras.models import Model
import jieba
import pickle


jieba.initialize()


# 基本信息
maxlen = 512
epochs = 10
batch_size = 8
learing_rate = 2e-5

# bert配置
pre_model = 'NEZHA-Base'
path_drive = '../base_model/' + pre_model
config_path = os.path.join(path_drive, 'bert_config.json')
dict_path = os.path.join(path_drive, 'vocab.txt')


variants = [
    u'短短匹配A类',
    u'短短匹配B类',
    u'短长匹配A类',
    u'短长匹配B类',
    u'长长匹配A类',
    u'长长匹配B类',
]


# 模型部分
c_in = Input(shape=(1,))
c = Embedding(len(variants), 128)(c_in)
c = Reshape((128,))(c)
model = build_transformer_model(
    config_path,
    checkpoint_path=None,
    model='nezha',
    layer_norm_cond=c,
    additional_input_layers=c_in
)
output = GlobalAveragePooling1D()(model.output)
output = Dense(2, activation='softmax')(output)
model = Model(model.inputs, output)
# model.summary()


# 建立分词器
tokenizer = Tokenizer(
    dict_path,
    do_lower_case=True,
    pre_tokenize=lambda s: jieba.lcut(s, HMM=False)
)

def text_token(text):
    tokens = [
        tokenizer._token_translate.get(token) or token
        for token in tokenizer._tokenize(text)
    ]
    return tokens

# 将两个文本分词，并转换为id,连在一起
def encode(source, target):
    first_token = text_token(source)
    second_token = text_token(target)
    if len(first_token) + len(second_token) > 509:
        if len(first_token) > 255:
            if len(second_token) > 254:
                first_token = first_token[: 255]
                second_token = second_token[: 254]
            else:
                f_e = min(len(first_token), 509-len(second_token))
                first_token = first_token[: f_e]
        else:
            s_e = min(len(second_token), 509-len(first_token))
            second_token = second_token[:s_e]
    first_token.insert(0, tokenizer._token_start)
    first_token.append(tokenizer._token_end)
    second_token.append(tokenizer._token_end)

    first_token_ids = tokenizer.tokens_to_ids(first_token)
    second_token_ids = tokenizer.tokens_to_ids(second_token)
    first_segment_ids = [0] * len(first_token_ids)
    second_segment_ids = [1] * len(second_token_ids)
    first_token_ids.extend(second_token_ids)
    first_segment_ids.extend(second_segment_ids)
    return first_token_ids, first_segment_ids

def get_pre(data_path, cond):
    key = 'labelA' if 'A' in data_path else 'labelB'
    data_path = data_path.encode("utf-8")
    result = {}
    count = 0

    with open(data_path, encoding="utf-8") as f:
        for l in f:
            print('\r', count, end='', flush=True)
            l = json.loads(l)
            first_token_ids, first_segment_ids = encode(l['source'], l['target'])
            first_token_ids = np.array([first_token_ids])
            first_segment_ids = np.array([first_segment_ids])
            conds = np.array([[cond], ])
            pre = model.predict([first_token_ids, first_segment_ids, conds])
            result[count] = {}
            result[count]["pre"] = pre[0][1]
            if key in l:
                result[count]["true"] = int(l[key])
            if "id" in l:
                result[count]["id"] = l["id"]
            count += 1
    return result


def get_f1(y_pre, y_true, split=0.5):
    tp, fp = 0.0, 0.0
    fn, tn = 0.0, 0.0
    for i in range(len(y_pre)):
        if y_true[i] == 1:
            if y_pre[i] >= split:
                tp += 1
            else:
                fn += 1
        else:
            if y_pre[i] >= split:
                fp += 1
            else:
                tn += 1
    P = tp / (tp + fp)
    R = tp / (tp + fn)
    return 2 * P * R / (P + R)

def get_best_split_f1():
    best_split_pre = {}

    for i, var in enumerate(variants):
        key = 'labelA' if 'A' in var else 'labelB'
        f = '../datasets/rematch/%s/valid.txt' % var
        f = f.encode("utf-8")
        val_split_pre = np.arange(0.3, 0.7, 0.001)
        val_split_pre_f1 = []
        y_true = []
        y_pre = []
        best_f1 = -1
        best_sp = 0.5
        ccc = 0
        with open(f, encoding="utf-8") as f:
            for l in f:
                print('\r', ccc, end='', flush=True)
                ccc += 1
                l = json.loads(l)
                first_token_ids, first_segment_ids = encode(l['source'], l['target'])
                y_true.append(int(l[key]))
                first_token_ids = np.array([first_token_ids])
                first_segment_ids = np.array([first_segment_ids])
                conds = np.array([[i], ])
                pre = model.predict([first_token_ids, first_segment_ids, conds])
                y_pre.append(pre[0][1])
            for v_p in val_split_pre:
                fs = get_f1(y_pre, y_true, split=v_p)
                val_split_pre_f1.append(fs)
                if fs > best_f1:
                    best_f1 = fs
                    best_sp = v_p
        # print(val_split_pre_f1)
        print(best_sp, best_f1)
        best_split_pre[var] = best_sp

    print("====================")
    print(best_split_pre)
    return best_split_pre


if __name__ == '__main__':
    model_paths = ["./2_nezha_base_sort_split_random.weights"]

    best_split_pre = {}

    for model_path in model_paths:
        model.load_weights(model_path)
        print("\n\n")
        print(model_path)
        best_split_pre[model_path] = get_best_split_f1()

    pickle.dump(best_split_pre, open("./best_split.pkl", "wb"))