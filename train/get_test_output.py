# -*- coding:utf-8 -*-
# @Time: 2021/5/31 13:59


import os, json
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np
from bert4keras.models import build_transformer_model
from bert4keras.tokenizers import Tokenizer
from bert4keras.snippets import open
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


if __name__ == '__main__':
    """
    conds = [
    u'短短匹配A类' 0,
    u'短短匹配B类' 1,
    u'短长匹配A类' 2,
    u'短长匹配B类' 3,
    u'长长匹配A类' 4,
    u'长长匹配B类' 5,
]
    """

    best_split = pickle.load(open("./best_split.pkl", "rb"))
    model_paths = ["./2_nezha_base_sort_split_random.weights"]

    all_pre = {}
    for model_path in model_paths:
        all_pre[model_path] = {}
        model.load_weights(model_path)
        for i, var in enumerate(variants):
            all_pre[model_path][var] = {}
            path = '../datasets/final_test/%s/test_with_id.txt' % var
            path = path.encode("utf-8")
            result = get_pre(path, i)
            for key, val in result.items():
                all_pre[model_path][var][val["id"]] = val["pre"]

    result_save = {}
    for i, var in enumerate(variants):
        for key_id in all_pre[model_path[0]][var].keys():
            count0 = 0
            count1 = 0
            for model_path in model_paths:
                if all_pre[model_path][var][key_id] >= best_split[model_path][var]:
                    count1 += 1
                else:
                    count0 += 1
            if count1 > count0:
                result_save[key_id] = 1
            else:
                result_save[key_id] = 0

    with open("result_zh_split.csv", 'w') as f:
        f.write('id,label\n')
        for key, val in result_save.items():
            f.write('%s,%s\n' % (key, val))