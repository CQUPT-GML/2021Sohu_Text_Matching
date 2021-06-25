# -*- coding:utf-8 -*-
# @Time: 2021/5/31 13:59
# @Author: duiya duiyady@163.com


import os, json

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np
from bert4keras.models import build_transformer_model
from bert4keras.tokenizers import Tokenizer
from bert4keras.snippets import open
from keras.layers import Input, Embedding, Reshape, GlobalAveragePooling1D, Dense
from keras.models import Model
import jieba
import argparse
import time
t1 = time.time()

jieba.initialize()

parser = argparse.ArgumentParser()
parser.add_argument("-i", "-input", type=str, required=True, help="输入文件")
parser.add_argument("-o", "-output", type=str, required=True, help="输出文件")
args = parser.parse_args()
path_input = args.i
path_output = args.o
# 基本信息
maxlen = 512
epochs = 10
batch_size = 8
learing_rate = 2e-5

# bert配置
pre_model = 'NEZHA-Base'
path_drive = 'base_model/' + pre_model
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
var_name = ["ss_a", "ss_b", "sl_a", "sl_b", "ll_a", "ll_b"]

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
                f_e = min(len(first_token), 509 - len(second_token))
                first_token = first_token[: f_e]
        else:
            s_e = min(len(second_token), 509 - len(first_token))
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


def get_pre(data_list, cond):
    key = 'labelA' if 'a' in data_list else 'labelB'
    result = {}
    count = 0
    for l in data_list:
        print('\r', count, end='', flush=True)
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


def get_all_pre(data_path):
    res_ss_a = []
    res_ss_b = []
    res_sl_a = []
    res_sl_b = []
    res_ll_a = []
    res_ll_b = []
    cond_ss_a = 0
    cond_ss_b = 1
    cond_sl_a = 2
    cond_sl_b = 3
    cond_ll_a = 4
    cond_ll_b = 5
    with open(data_path, encoding="utf-8") as f:
        for l in f:
            # print('\r', count, end='', flush=True)
            l = json.loads(l)
            if l['id'].startswith("ss") and l['id'].endswith("a"):
                res_ss_a.append(l)
                _var = '短短匹配A类'
            if l['id'].startswith("ss") and l['id'].endswith("b"):
                res_ss_b.append(l)
                _var = '短短匹配B类'
            if l['id'].startswith("sl") and l['id'].endswith("a"):
                res_sl_a.append(l)
                _var = '短长匹配A类'
            if l['id'].startswith("sl") and l['id'].endswith("b"):
                res_sl_b.append(l)
                _var = '短长匹配B类'
            if l['id'].startswith("ll") and l['id'].endswith("a"):
                res_ll_a.append(l)
                _var = '长长匹配A类'
            if l['id'].startswith("ll") and l['id'].endswith("b"):
                res_ll_b.append(l)
                _var = '长长匹配B类'

    return get_pre(res_ss_a, cond_ss_a), get_pre(res_ss_b, cond_ss_b), \
           get_pre(res_sl_a, cond_sl_a), get_pre(res_sl_b, cond_sl_b), \
           get_pre(res_ll_a, cond_ll_a), get_pre(res_ll_b, cond_ll_b)


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

    best_split = {}
    best_split["weights/1_nezha_base_124_gp_lah.weights"] = {}
    best_split["weights/1_nezha_base_124_gp_lah.weights"]["短短匹配A类"] = 0.599
    best_split["weights/1_nezha_base_124_gp_lah.weights"]["短短匹配B类"] = 0.507
    best_split["weights/1_nezha_base_124_gp_lah.weights"]["短长匹配A类"] = 0.523
    best_split["weights/1_nezha_base_124_gp_lah.weights"]["短长匹配B类"] = 0.452
    best_split["weights/1_nezha_base_124_gp_lah.weights"]["长长匹配A类"] = 0.480
    best_split["weights/1_nezha_base_124_gp_lah.weights"]["长长匹配B类"] = 0.514

    best_split["weights/2_nezha_base_sort_split_random.weights"] = {}
    best_split["weights/2_nezha_base_sort_split_random.weights"]["短短匹配A类"] = 0.655
    best_split["weights/2_nezha_base_sort_split_random.weights"]["短短匹配B类"] = 0.638
    best_split["weights/2_nezha_base_sort_split_random.weights"]["短长匹配A类"] = 0.515
    best_split["weights/2_nezha_base_sort_split_random.weights"]["短长匹配B类"] = 0.623
    best_split["weights/2_nezha_base_sort_split_random.weights"]["长长匹配A类"] = 0.414
    best_split["weights/2_nezha_base_sort_split_random.weights"]["长长匹配B类"] = 0.690

    best_split["weights/2_nezha_base_new2.weights"] = {}
    best_split["weights/2_nezha_base_new2.weights"]["短短匹配A类"] = 0.574
    best_split["weights/2_nezha_base_new2.weights"]["短短匹配B类"] = 0.448
    best_split["weights/2_nezha_base_new2.weights"]["短长匹配A类"] = 0.451
    best_split["weights/2_nezha_base_new2.weights"]["短长匹配B类"] = 0.405
    best_split["weights/2_nezha_base_new2.weights"]["长长匹配A类"] = 0.346
    best_split["weights/2_nezha_base_new2.weights"]["长长匹配B类"] = 0.465

    best_split["weights/3_nezha_base_4_gp_lah.weights"] = {}
    best_split["weights/3_nezha_base_4_gp_lah.weights"]["短短匹配A类"] = 0.399
    best_split["weights/3_nezha_base_4_gp_lah.weights"]["短短匹配B类"] = 0.361
    best_split["weights/3_nezha_base_4_gp_lah.weights"]["短长匹配A类"] = 0.528
    best_split["weights/3_nezha_base_4_gp_lah.weights"]["短长匹配B类"] = 0.463
    best_split["weights/3_nezha_base_4_gp_lah.weights"]["长长匹配A类"] = 0.585
    best_split["weights/3_nezha_base_4_gp_lah.weights"]["长长匹配B类"] = 0.419

    best_split["weights/3_nezha_base_4_sort_split_random.weights"] = {}
    best_split["weights/3_nezha_base_4_sort_split_random.weights"]["短短匹配A类"] = 0.531
    best_split["weights/3_nezha_base_4_sort_split_random.weights"]["短短匹配B类"] = 0.441
    best_split["weights/3_nezha_base_4_sort_split_random.weights"]["短长匹配A类"] = 0.424
    best_split["weights/3_nezha_base_4_sort_split_random.weights"]["短长匹配B类"] = 0.435
    best_split["weights/3_nezha_base_4_sort_split_random.weights"]["长长匹配A类"] = 0.388
    best_split["weights/3_nezha_base_4_sort_split_random.weights"]["长长匹配B类"] = 0.371

    model_paths=["weights/1_nezha_base_124_gp_lah.weights","weights/2_nezha_base_sort_split_random.weights","weights/2_nezha_base_new2.weights","weights/3_nezha_base_4_gp_lah.weights","weights/3_nezha_base_4_sort_split_random.weights"]
    all_pre = {}
    for model_path in model_paths:
        print(model_path)
        all_pre[model_path] = {}
        model.load_weights(model_path)
        result_ss_a, result_ss_b, result_sl_a, result_sl_b, result_ll_a, result_ll_b = get_all_pre(path_input)

        all_pre[model_path]['短短匹配A类'] = {}
        for key, val in result_ss_a.items():
            all_pre[model_path]['短短匹配A类'][val["id"]] = val["pre"]
        all_pre[model_path]['短短匹配B类'] = {}
        for key, val in result_ss_b.items():
            all_pre[model_path]['短短匹配B类'][val["id"]] = val["pre"]

        all_pre[model_path]['短长匹配A类'] = {}
        for key, val in result_sl_a.items():
            all_pre[model_path]['短长匹配A类'][val["id"]] = val["pre"]
        all_pre[model_path]['短长匹配B类'] = {}
        for key, val in result_sl_b.items():
            all_pre[model_path]['短长匹配B类'][val["id"]] = val["pre"]

        all_pre[model_path]['长长匹配A类'] = {}
        for key, val in result_ll_a.items():
            all_pre[model_path]['长长匹配A类'][val["id"]] = val["pre"]
        all_pre[model_path]['长长匹配B类'] = {}
        for key, val in result_ll_b.items():
            all_pre[model_path]['长长匹配B类'][val["id"]] = val["pre"]
    result_save = {}
    for i, var in enumerate(variants):
        for key_id in all_pre["weights/1_nezha_base_124_gp_lah.weights"][var].keys():
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

    with open(path_output, 'w') as f:
        f.write('id,label\n')
        for key, val in result_save.items():
            f.write('%s,%s\n' % (key, val))

    print('\n')
    t2 = time.time()
    print(t2-t1)