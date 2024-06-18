# 基于 BERT 的文本情感分析系统

本项目使用BERT模型实现了一个文本情感分析系统，能够将文本分类为六种情感类别：中性、高兴、生气、伤心、恐惧和惊讶。

## 目录

- [基于 BERT 的文本情感分析系统](#基于-bert-的文本情感分析系统)
  - [目录](#目录)
  - [安装指南](#安装指南)
  - [使用说明](#使用说明)
    - [训练模型](#训练模型)
    - [评估模型](#评估模型)
    - [使用情感分析器](#使用情感分析器)
  - [项目结构](#项目结构)
  - [数据准备](#数据准备)
  - [模型保存与加载](#模型保存与加载)
  - [日志记录与监控](#日志记录与监控)
  - [许可证](#许可证)
  - [注意事项](#注意事项)

## 安装指南

首先克隆仓库并安装必要的依赖包：

```
git clone https://github.com/AuroraEchos/SentimentAnalysis.git
cd SentimentAnalysis
makedir bert_model_files
makedir saved_models
pip install -r requirements.txt
```

## 使用说明

### 训练模型
1. 准备训练集、验证集和测试集，格式为TSV文件。
2. 根据需要更新代码中的数据集路径。
3. 运行以下命令开始训练：
   ```
   python train.py
   ```
   该命令将按照指定轮数训练基于BERT模型的情感分析器，并根据验证准确率保存最佳模型。

### 评估模型
训练完成后，可以通过以下命令在测试集上评估模型性能：
```
python evaluate.py
```
该命令将输出测试集上的准确率。

### 使用情感分析器
你可以使用训练好的BERT模型来预测新文本的情感类别。以下是一个示例：
```
from sentiment_analyzer import SentimentAnalyzer

model_path = 'saved_models/best_model'
analyzer = SentimentAnalyzer(model_path)

text = "今天天气不错!"
predicted_label, confidence = analyzer.predict(text)
print(f"Predicted sentiment: {predicted_label} (Confidence: {confidence:.2f})")
```

## 项目结构
```
sentiment-analysis/
├── bert_model_files/
│   └── ...
├── data/
│   ├── cn_stopwords.txt
│   ├── data.csv
│   └── ...
├── saved_models/
│   └── bert_epoch_0/
│       ├── config.json
│       └── ...

├── evaluate.py
├── preprocess.py
├── README.md
├── sentiment_analyzer.py
├── train.py

```

## 数据准备
本项目使用的数据集为SMP微博情绪6分类数据集，来源于[SMP2020-EWECT](https://smp2020ewect.github.io/)。数据集应为TSV格式，包含两列：第一列为标签，第二列为文本。请确保数据集包含表头行。

示例 web_train.tsv：
```
label	text
0	这是一个中性句子。
1	我今天非常高兴！
2	我对你非常生气。
3	我对此感到非常伤心。
4	我现在真的很害怕。
5	太惊讶了！
```

## 模型保存与加载
预训练模型为 BERT，来源于[Hugging Face](https://huggingface.co/google-bert/bert-base-chinese)模型库。
模型和分词器在每个轮次结束时会根据验证准确率的提高情况进行保存，保存在 /saved_models 目录中。
加载保存的模型可以使用以下代码：
```
from transformers import BertTokenizer, BertForSequenceClassification

model_path = 'saved_models/bert_epoch_'
tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertForSequenceClassification.from_pretrained(model_path)
```

## 日志记录与监控
训练过程中，包含每个轮次的平均损失和验证准确率等信息会通过 logging 模块记录，并打印到控制台。

## 许可证
此项目基于MIT许可证。详情请参阅LICENSE文件。

## 注意事项

- 本项目仅供学习和研究使用，不保证在所有环境下均可正常运行。
- 如果遇到问题或有改进意见，请联系项目作者。
- 维护者：Wenhao Liu
