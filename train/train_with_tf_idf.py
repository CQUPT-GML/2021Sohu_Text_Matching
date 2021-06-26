# -*- coding:utf-8 -*-
# @Time: 2021/5/30 19:20
# @Author: duiya duiyady@163.com

import os, json, sys
sys.path.insert(0, '/liliu_fix/share/sohu/')
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np
from bert4keras.backend import keras, search_layer
from bert4keras.models import build_transformer_model
from bert4keras.tokenizers import Tokenizer
from bert4keras.optimizers import Adam, extend_with_exponential_moving_average
from bert4keras.snippets import sequence_padding, DataGenerator
# import tensorflow as tf
from keras.layers import Input, Embedding, Reshape, Dense
from keras.layers import GlobalAveragePooling1D
from keras.models import Model
from keras.engine.base_layer import Layer, InputSpec
from keras import backend as K
from tqdm import tqdm
import jieba
import random
import pickle
import time



jieba.initialize()

# trick
TRAIN_DATA_SPLIT = False  # 短长是否要切


# 基本信息
maxlen = 512
epochs = 10
batch_size = 8
learing_rate = 2e-5

# bert配置
pre_model = 'NEZHA-Base'
path_drive = '../base_model/' + pre_model
config_path = os.path.join(path_drive, 'bert_config.json')
checkpoint_path = os.path.join(path_drive, 'model.ckpt-900000')
dict_path = os.path.join(path_drive, 'vocab.txt')

variants = [
    u'短短匹配A类',
    u'短短匹配B类',
    u'短长匹配A类',
    u'短长匹配B类',
    u'长长匹配A类',
    u'长长匹配B类',
]

class Lookahead(object):
    """Add the [Lookahead Optimizer](https://arxiv.org/abs/1907.08610) functionality for [keras](https://keras.io/).
    """

    def __init__(self, k=5, alpha=0.5):
        self.k = k
        self.alpha = alpha
        self.count = 0

    def inject(self, model):
        """Inject the Lookahead algorithm for the given model.
        The following code is modified from keras's _make_train_function method.
        See: https://github.com/keras-team/keras/blob/master/keras/engine/training.py#L497
        """
        if not hasattr(model, 'train_function'):
            raise RuntimeError('You must compile your model before using it.')

        model._check_trainable_weights_consistency()

        if model.train_function is None:
            inputs = (model._feed_inputs +
                      model._feed_targets +
                      model._feed_sample_weights)
            if model._uses_dynamic_learning_phase():
                inputs += [K.learning_phase()]
            fast_params = model._collected_trainable_weights

            with K.name_scope('training'):
                with K.name_scope(model.optimizer.__class__.__name__):
                    training_updates = model.optimizer.get_updates(
                        params=fast_params,
                        loss=model.total_loss)
                    slow_params = [K.variable(p) for p in fast_params]
                fast_updates = (model.updates +
                                training_updates +
                                model.metrics_updates)

                slow_updates, copy_updates = [], []
                for p, q in zip(fast_params, slow_params):
                    slow_updates.append(K.update(q, q + self.alpha * (p - q)))
                    copy_updates.append(K.update(p, q))

                # Gets loss and metrics. Updates weights at each call.
                fast_train_function = K.function(
                    inputs,
                    [model.total_loss] + model.metrics_tensors,
                    updates=fast_updates,
                    name='fast_train_function',
                    **model._function_kwargs)

                def F(inputs):
                    self.count += 1
                    R = fast_train_function(inputs)
                    if self.count % self.k == 0:
                        K.batch_get_value(slow_updates)
                        K.batch_get_value(copy_updates)
                    return R

                model.train_function = F


def get_idf_value(path):
    IDF = pickle.load(open(path, "rb"))
    return IDF

class WordPooling(Layer):

    def __init__(self, data_format='channels_last', **kwargs):
        super(WordPooling, self).__init__(**kwargs)
        self.input_spec = InputSpec(ndim=3)
        self.data_format = K.normalize_data_format(data_format)
        self.supports_masking = True


    def call(self, inputs, mask=None, word_weight=None):
        word_weight = K.cast(word_weight, K.floatx())
        x = K.batch_dot(word_weight, inputs)
        return x

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[2])

    def compute_mask(self, inputs, mask=None, word_weight=None):
        return None

    def get_config(self):
        config = {'data_format': self.data_format}
        base_config = super(Layer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))



# 模型部分
c_in = Input(shape=(1,))
word_weight = Input(shape=(None, ), name="wordinput")
c = Embedding(len(variants), 128)(c_in)
c = Reshape((128,))(c)
model = build_transformer_model(
    config_path,
    checkpoint_path,
    model='nezha',
    layer_norm_cond=c,
    additional_input_layers=c_in
)

output = WordPooling()(model.output, word_weight=word_weight)
output = Dense(2, activation='softmax')(output)
all_inputs = model.inputs
all_inputs.append(word_weight)
model = Model(all_inputs, output)
model.summary()
AdamEMA = extend_with_exponential_moving_average(Adam, name='AdamEMA')
optimizer = AdamEMA(learing_rate, ema_momentum=0.9999)

def sparse_categorical_crossentropy(y_true, y_pred):
    """自定义稀疏交叉熵。
    """
    y_true = K.reshape(y_true, K.shape(y_pred)[:-1])
    y_true = K.cast(y_true, 'int32')
    y_true = K.one_hot(y_true, K.shape(y_pred)[-1])
    return K.categorical_crossentropy(y_true, y_pred)


def loss_with_gradient_penalty(y_true, y_pred, epsilon=1):
    """带梯度惩罚的loss
    """
    # print(y_true.shape, y_pred.shape)
    loss = K.mean(sparse_categorical_crossentropy(y_true, y_pred))
    embeddings = search_layer(y_pred, 'Embedding-Token').embeddings
    gp = K.sum(K.gradients(loss, [embeddings])[0].values**2)
    return loss + 0.5 * epsilon * gp

model.compile(
    loss=loss_with_gradient_penalty,
    # loss='sparse_categorical_crossentropy',
    optimizer=optimizer,
    metrics=['accuracy']
)
lookahead = Lookahead(k=5, alpha=0.5)
lookahead.inject(model)


# 建立分词器
tokenizer = Tokenizer(
    dict_path,
    do_lower_case=True,
    pre_tokenize=lambda s: jieba.lcut(s, HMM=False)
)
IDF = get_idf_value("./compute_word_idf.pkl")

# 将文本转换为token, 分词
def text_token(text):
    tokens = [
        tokenizer._token_translate.get(token) or token
        for token in tokenizer._tokenize(text)
    ]
    return tokens


# 将一个长的token切成几段小的token
def split_seq_to_dix_len(text_token, split_len):
    result = []
    start = 0
    while start < len(text_token):
        end = start + split_len
        if end > len(text_token):
            end = len(text_token)
        if end - start > 100 or start == 0:
            result.append(text_token[start:end])
        start = end
    return result


# 将句子切割匹配好
def get_tokens(source, target, split=True):
    source_token = text_token(source)
    target_token = text_token(target)
    if split:
        if len(source_token) > 255:
            source_token = source_token[:255]
        source_token_len = len(source_token)
        target_seq_max_len = 509 - source_token_len
        target_tokens = split_seq_to_dix_len(target_token, target_seq_max_len)
    else:
        if len(source_token) + len(target_token) > 509:
            if len(source_token) > 255:
                if len(target_token) > 254:
                    source_token = source_token[: 255]
                    target_token = target_token[: 254]
                else:
                    f_e = min(len(source_token), 509 - len(target_token))
                    source_token = source_token[: f_e]
            else:
                s_e = min(len(target_token), 509 - len(source_token))
                target_token = target_token[:s_e]
        target_tokens = [target_token, ]
    return source_token, target_tokens


# 导入训练数据 切分
def load_train_data(need=[1, 2]):
    ori_data = []
    ori_data_len = []
    for i, var in enumerate(variants):
        key = 'labelA' if 'A' in var else 'labelB'
        fs = []
        if 1 in need:
            tmp_path = '../datasets/sohu2021_open_data_clean/%s/train.txt' % var
            tmp_path = tmp_path.encode("utf-8")
            fs.append(tmp_path)
            tmp_path = '../datasets/sohu2021_open_data_clean/%s/valid.txt' % var
            tmp_path = tmp_path.encode("utf-8")
            fs.append(tmp_path)

        if 2 in need:
            tmp_path = '../datasets/round2/%s.txt' % var
            tmp_path = tmp_path.encode("utf-8")
            fs.append(tmp_path)
        if 3 in need:
            tmp_path = '../datasets/round3/%s/train.txt' % var
            tmp_path = tmp_path.encode("utf-8")
            fs.append(tmp_path)
        if 4 in need:
            tmp_path = '../datasets/rematch/%s/train.txt' % var
            tmp_path = tmp_path.encode("utf-8")
            fs.append(tmp_path)

        for f in fs:
            with open(f, encoding="utf8") as f:
                for l in f:
                    l = json.loads(l)
                    # l = eval(l)
                    source = l['source']
                    target = l['target']
                    if TRAIN_DATA_SPLIT and (i == 2 or i == 3):
                        if int(l[key]) == 0:
                            source_token, target_tokens = get_tokens(source, target, True)
                        else:
                            source_token, target_tokens = get_tokens(source, target, False)
                    else:
                        source_token, target_tokens = get_tokens(source, target, False)

                    source = "".join(source_token)
                    source_token_len = len(source_token)
                    for target_token in target_tokens:
                        target = "".join(target_token)
                        ori_data.append((i, source, target, int(l[key])))
                        ori_data_len.append(source_token_len + len(target_token))

    return ori_data, ori_data_len


# 导入验证集 不切分
def load_valid_data(need=[1,2]):
    ori_data = []
    ori_data_len = []
    for i, var in enumerate(variants):
        key = 'labelA' if 'A' in var else 'labelB'
        fs = []
        if 1 in need:
            tmp_path = '../datasets/sohu2021_open_data_clean/%s/valid.txt' % var
            tmp_path = tmp_path.encode("utf-8")
            fs.append(tmp_path)
        if 4 in need:
            tmp_path = '../datasets/rematch/%s/valid.txt' % var
            tmp_path = tmp_path.encode("utf-8")
            fs.append(tmp_path)

        for f in fs:
            with open(f, encoding="utf8") as f:
                for l in f:
                    l = json.loads(l)
                    # l = eval(l)
                    source = l['source']
                    target = l['target']
                    source_token, target_tokens = get_tokens(source, target, False)
                    source = "".join(source_token)
                    source_token_len = len(source_token)
                    for target_token in target_tokens:
                        target = "".join(target_token)
                        ori_data.append((i, source, target, int(l[key])))
                        ori_data_len.append(source_token_len + len(target_token))
    return ori_data, ori_data_len


# 将分词后长度相同的文本尽量放在一起，随机的时候
def create_data_split(data_len):
    data_split = []
    now_split = []
    start_len = data_len[0]
    last_len = data_len[0]
    for i in range(len(data_len)):
        if data_len[i] == last_len or data_len[i] >= 509:
            now_split.append(i)
        elif data_len[i] != last_len:
            if data_len[i] - start_len > 3:
                if len(now_split) > 200:
                    data_split.append(now_split)
                    now_split = [i, ]
                    start_len = data_len[i]
                    last_len = data_len[i]
                else:
                    now_split.append(i)
                    last_len = data_len[i]
            else:
                now_split.append(i)
                last_len = data_len[i]
    if len(now_split) > 0:
        data_split.append(now_split)
    return data_split

# 获取tf-idf值
def get_tf_idf(first_token, second_token):
    word_count = {}
    count = len(first_token) + len(second_token)
    for val in first_token:
        if val not in word_count.keys():
            word_count[val] = 0
        word_count[val] += 1
    for val in second_token:
        if val not in word_count.keys():
            word_count[val] = 0
        word_count[val] += 1
    first_token_tfidf = [word_count[val]*IDF[val]/count for val in first_token]
    second_token_tfidf = [word_count[val]*IDF[val]/count for val in second_token]
    all_sum = sum(first_token_tfidf) + sum(second_token_tfidf)
    first_token_tfidf = [val / all_sum for val in first_token_tfidf]
    second_token_tfidf = [val / all_sum for val in second_token_tfidf]
    return first_token_tfidf, second_token_tfidf



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

    first_token_ids = tokenizer.tokens_to_ids(first_token)
    second_token_ids = tokenizer.tokens_to_ids(second_token)

    first_token_tfidf, second_token_idf = get_tf_idf(first_token_ids, second_token_ids)
    # cls权重为0,表示不考虑他的权重
    first_token_tfidf.insert(0, 0.0)
    first_token_ids.insert(0, tokenizer.token_to_id(tokenizer._token_start))
    # 第一句话和第二句话分隔的seq 0.0
    first_token_tfidf.append(0.0)
    first_token_ids.append(tokenizer.token_to_id(tokenizer._token_end))
    # 第二句话结尾的seq 0.0
    second_token_idf.append(0.0)
    second_token_ids.append(tokenizer.token_to_id(tokenizer._token_end))
    first_segment_ids = [0] * len(first_token_ids)
    second_segment_ids = [1] * len(second_token_ids)
    first_token_ids.extend(second_token_ids)
    first_segment_ids.extend(second_segment_ids)
    first_token_tfidf.extend(second_token_idf)
    return first_token_ids, first_segment_ids, first_token_tfidf


# 数据生成器
class data_generator(DataGenerator):
    def __iter__(self, random=False):
        batch_token_ids, batch_segment_ids = [], []
        batch_conds, batch_labels = [], []
        batch_words_weights = []
        for is_end, (cond, source, target, label) in self.sample(random):
            token_ids, segment_ids, word_weight = encode(source, target)
            # print(word_weight)
            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)
            batch_words_weights.append(word_weight)
            batch_conds.append([cond])
            batch_labels.append([label])
            if len(batch_token_ids) == self.batch_size or is_end:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                batch_conds = sequence_padding(batch_conds)
                batch_words_weights = sequence_padding(batch_words_weights)
                batch_labels = sequence_padding(batch_labels)
                # batch_words_weights = np.expand_dims(batch_words_weights, axis=1)
                # print(batch_token_ids.shape, batch_segment_ids.shape, batch_conds.shape, batch_words_weights.shape, batch_labels.shape)
                yield [
                    batch_token_ids, batch_segment_ids, batch_conds, batch_words_weights
                ], batch_labels
                batch_token_ids, batch_segment_ids = [], []
                batch_words_weights = []
                batch_conds, batch_labels = [], []

    def sample(self, random=False):
        for i in range(len(self.search_index)):
            if i == len(self.search_index)-1:
                yield True, self.data[self.search_index[i]]
            else:
                yield False, self.data[self.search_index[i]]
        if random:
            self.data_random()

    def data_random(self):
        self.search_index = []
        random.shuffle(self.data_split)
        for val in self.data_split:
            random.shuffle(val)
            self.search_index.extend(val)

    def set_data_split(self, data_split):
        self.data_split = data_split
        self.search_index = []
        for val in self.data_split:
            self.search_index.extend(val)


# 评测函数 A、B两类分别算F1然后求平均
def evaluate(data):
    total_a, right_a = 0., 0.
    total_b, right_b = 0., 0.
    for x_true, y_true in tqdm(data):
        y_pred = model.predict(x_true).argmax(axis=1)

        y_true = y_true[:, 0]
        flag = x_true[2][:, 0] % 2
        total_a += ((y_pred + y_true) * (flag == 0)).sum()
        right_a += ((y_pred * y_true) * (flag == 0)).sum()
        total_b += ((y_pred + y_true) * (flag == 1)).sum()
        right_b += ((y_pred * y_true) * (flag == 1)).sum()
    f1_a = 2.0 * right_a / total_a
    f1_b = 2.0 * right_b / total_b
    return {'f1': (f1_a + f1_b) / 2, 'f1_a': f1_a, 'f1_b': f1_b}


# 评估与保存
class Evaluator(keras.callbacks.Callback):
    def __init__(self, valid_generator, save_path="best_model.weights"):
        self.best_val_f1 = 0.
        self.valid_generator = valid_generator
        self.save_path = save_path

    def on_epoch_end(self, epoch, logs=None):
        optimizer.apply_ema_weights()
        metrics = evaluate(self.valid_generator)
        if metrics['f1'] > self.best_val_f1:
            self.best_val_f1 = metrics['f1']
            model.save_weights(self.save_path)
        optimizer.reset_old_weights()
        metrics['best_f1'] = self.best_val_f1
        model.save_weights(str(epoch) + "_" + self.save_path)
        print(metrics)


if __name__ == '__main__':
    model_save_path = "word_pooling_model"

    s_time = time.time()
    train_data, train_data_len = load_train_data(need=[1, 2, 4])
    valid_data, valid_data_len = load_valid_data(need=[4])

    train_data_len, train_data = zip(*sorted(zip(train_data_len, train_data)))
    train_generator = data_generator(train_data, batch_size=batch_size)
    train_data_split = [[i for i in range(len(train_data))], ]

    train_data_split = create_data_split(train_data_len)
    train_generator.set_data_split(train_data_split)

    valid_generator = data_generator(valid_data, batch_size=batch_size)
    valid_data_split = [[i for i in range(len(valid_data))], ]
    valid_generator.set_data_split(valid_data_split)
    e_time = time.time()
    print("训练数据处理完：", e_time - s_time)

    evaluator = Evaluator(valid_generator, save_path=model_save_path)

    model.fit(
        train_generator.forfit(random=True),
        steps_per_epoch=len(train_generator),
        epochs=epochs,
        callbacks=[evaluator]
    )