# 实操题（二）：大语言模型在特定任务下的微调

## 模型介绍：

采用 gemma-2b 作为基础模型

## 任务介绍

QQP(The Quora Question Pairs, Quora 问题对数集)，相似性和释义任务，是社区问答网站 Quora 中问题对的集合。任务是确定一对问题在语义上是否等效(duplicate)。QQP 是正负样本不均衡的，不同是的 QQP 负样本占 63%，正样本是 37%。

CoLA(The Corpus of Linguistic Acceptability，语言可接受性语料库)，单句子分类任务，语料来自语言理论的书籍和期刊，每个句子被标注为是否合乎语法的单词序列。本任务是一个二分类任务，标签共两个，分别是 0 和 1，其中 0 表示不合乎语法，1 表示合乎语法(grammar)。

## 题目

### 数据集介绍与预处理过程

1）两个数据集的训练集存放在/home/aiservice/worksapce/data/questionBData 目录下；
2）其中{dataset}\_train.tsv 用作训练数据集，{dataset}\_dev.tsv 用于待推理数据集。
3）需要使用 data_process.py 对两个数据集进行预处理，将结果存放在工作目录 pre_data 各数据集目录中，保存名称为{dataset}\_train.json 与{dataset}\_dev.json，
4）上述{dataset}为 CoLA 或 QQP。

CoLA 数据集由于没有表头，题目中给出每条数据头，待推理数据没有 label。

```
    template_id   label  difficult            sentence
       gj04	        1	    	    Our friends won't buy this analysis, let alone the next one we propose.
       cj99	        0	    *	    You get angrier, the more we eat, don't we.
```

由于训练集数量巨大，请通过修改 data_process.py 以控制训练数据的数量（建议使用 4000 条样本，其中一部分样本可用于使用 predict.py 判断模型是否可提交）

![Alt text](1722305529986.jpg)
![Alt text](image.png)

需打开 terminal 执行 python 代码，可右键预览 readme 查看图片（打开 terminal 的方式），例如：

```
    python data_process.py
```

每条原始数据预处理的结果为如下示例，其中 instruction 需要选手自行填写，特别提醒【instruction 会影响微调模型的结果好坏】：

```
    {
        'instruction': '', # 给予模型的任务提示，数据集为英文，建议提示也是英语
        'input': '', # 数据集每个样本内容QQP是两个句子，CoLA是一个句子
        'output': '0', # 训练集的标签
    }
```

给出一种instruction 示例：

```
 如果处理的是QQP数据，则instruction = Are the scentences meaning the same ? same output 1 , otherwise 0 .
 如果处理的是CoLA数据，则 instruction = Is the grammar of the scentence right ? right output 1 , otherwise 0 .
```

特别提醒：待推理数据集没有'output'，需要选手使用训练的模型对待推理数据集进行推理得到结果，参见

### 模型训练

可修改 train.py 和 config.py 的相关参数，包括但不限于模型位置、数据集位置、lora 等参数。其中需要修改 config.py 中的 DATA_SET 以更换数据集，用于训练（train.py）和推理（predict.py）

修改完成后可通过 【python train.py】 执行微调过程

结束微调后，请将训练好的模型保存到工作目录的 saved_model（lora_path）文件夹中。

### 模型推理

可修改 predict.py 相关参数，加载微调过的模型（已存放在 saved_model 文件夹）

在执行推理任务前，请检查config.py中的数据集名称和数据集地址是否正确。例如本次要推理是QQP，则需要指定的是QQP_dev.json的位置（在pre_data中）

执行 【python predict.py】输出模型对于两个数据集任务的推理结果。

'''
注：
推理过程只能单句输入，单句输出，由于时间关系，只选用 {dataset}_dev.json中前150 个样本作为待推理数据；
推理结果需按要求保存到 workspace/questionB/ 目录中的 result_QQP.txt 和 result_CoLA.txt 中，可参考目录中已给出的 result_{dataset}.txt 示例格式
'''

## 评分规则(满分100分)

1. 根据在 pre_data 是否存在进行数据预处理后的 json 文件且文件格式合规（10 分）
2. saved_model 中保存正确路径的微调模型且文件大小和格式合规（20 分）
3. result_{dataset}.txt 预测结果的准确率给予分数（70 分）。注：准确率 * 70

## 标准答案
下面给出标准答案的位置，每位选手可自行编写对比脚本，参照上述【评分规则】进行判分练习：
result_CoLA.txt 的标准答案文件是data/questionBAnswer/result_CoLA_64.txt
result_QQP.txt 的标准答案文件是data/questionBAnswer/result_QQP_46.txt

## 考试环境

基础环境：
coda 版本 cuda:12.1.0-devel-ubuntu20.04
python 版本 3.9.19

资源详情：10 核 50G 内存 1 张 A10 卡

    numpy==1.24.4
    scikit-learn==1.3.0
    matplotlib==3.7.5
    transformers==4.38.1
    datasets==2.18.0
    accelerate==0.26.1
    evaluate==0.4.1
    bitsandbytes==0.42.0
    certifi==2024.6.2
    charset-normalizer==3.3.2
    colorama==0.4.6
    contourpy==1.1.1
    cycler==0.12.1
    filelock==3.15.4
    fonttools==4.53.0
    idna==3.7
    importlib_resources==6.4.0
    Jinja2==3.1.4
    kiwisolver==1.4.5
    MarkupSafe==2.1.5
    mpmath==1.3.0
    networkx==3.1
    onnx==1.16.1
    opencv-python==4.10.0.84
    packaging==23.1
    pandas==2.0.3
    pillow==9.5.0
    protobuf==3.20.3
    psutil==6.0.0
    py-cpuinfo==9.0.0
    pyparsing==3.1.2
    python-dateutil==2.9.0.post0
    pytz==2024.1
    PyYAML==6.0.1
    requests==2.32.3
    scipy==1.10.1
    seaborn==0.13.2
    setuptools==70.1.1
    six==1.16.0
    sympy==1.12.1
    torch==2.0.0
    torchaudio==2.0.1
    torchvision==0.15.1
    tqdm==4.66.4
    typing_extensions==4.12.2
    tzdata==2024.1
    ultralytics==8.2.45
    ultralytics-thop==2.0.0
    urllib3==2.2.2
    wheel==0.43.0
    zipp==3.19.2
    modelscope==1.15.0
    pycocotools==2.0.8
    peft==0.10.0
    sentencepiece==0.1.99
    streamlit==1.24.0
