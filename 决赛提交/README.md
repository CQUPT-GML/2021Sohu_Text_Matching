# 2021搜狐校园文本匹配算法大赛
## 决赛提交内容

## 目录结构

```
.
├── base_model
│   └── NEZHA-Base
│       ├── bert_config.json
│       └── vocab.txt
├── data
│   ├── input         #存放需要测试的文件
│   └── output      #输出预测结果
├── new_get_test_output.py   #预测文件
├── requirements.txt
└── weights    #将下载的weights文件存放在此目录下
    ├── 1_nezha_base_124_gp_lah.weights
    ├── 2_nezha_base_new2.weights
    ├── 2_nezha_base_sort_split_random.weights
    ├── 3_nezha_base_4_gp_lah.weights
    └── 3_nezha_base_4_sort_split_random.weights
```

## 运行环境

```
bert4keras==0.10.6
h5py==2.9.0
jieba==0.42.1
Keras==2.2.4
matplotlib==3.3.4
numpy==1.19.2
pandas==1.1.5
scikit-learn==0.24.1
scipy==1.5.2
tensorflow-gpu==1.14.0
tqdm==4.61.0
```

## 训练权重

下载地址[[百度网盘](https://pan.baidu.com/s/1PJcN3t6wzB2ei_BBg30o5Q ),提取码:6666]

## 运行

将需要预测的文件存放`data/input/`在目录下并重命名为`test.txt`

运行如下命令,即可在`/data/output/`目录下得到`pred.csv`预测文件

`python new_get_test_output.py -input /data/input/test.txt -output /data/output/pred.csv`

