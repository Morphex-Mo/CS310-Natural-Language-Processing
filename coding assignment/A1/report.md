# CS310 NLP Assignment 1 Report

## 1. 实验设置

- 任务: 中文幽默检测二分类
- 数据来源: CLEVA 幽默检测数据集
- 代码文件: [coding assignment/A1/A1_nn.ipynb](coding%20assignment/A1/A1_nn.ipynb)
- 随机种子: 310
- 运行设备: CPU

数据划分:

- 训练集: 12677
- 验证集: 1267 (由训练集按 10% 划分)
- 测试集: 651

## 2. 模型参数

模型结构为 EmbeddingBag + 多层全连接分类器。

超参数:

- num_classes: 2
- embed_dim: 128
- hidden_dims: (128, 64)
- dropout: 0.25
- learning_rate: 0.001
- epochs: 10
- optimizer: Adam
- loss: CrossEntropyLoss

模型规模:

- 可训练参数总量: 965058

## 3. 分词与词表

你当前 notebook 中比较了两种 tokenizer:

1. basic_char
- 规则: 仅保留中文单字，丢弃英文、数字、标点
- 词表大小: 2687

2. advanced
- 规则: 正则拆分 + 中文短语词典最大匹配（phrase lexicon）
- 词表大小: 7345

## 4. 测试集结果

| tokenizer | accuracy | precision | recall | f1 | tp | tn | fp | fn |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| basic_char | 0.7312 | 0.4706 | 0.2353 | 0.3137 | 40 | 436 | 45 | 130 |
| advanced | 0.7066 | 0.4305 | 0.3824 | 0.4050 | 65 | 395 | 86 | 105 |

当前保存的最佳模型（按 F1）:

- best_experiment: advanced
- checkpoint: [coding assignment/A1/best_advanced_embeddingbag_model.pth](coding%20assignment/A1/best_advanced_embeddingbag_model.pth)

## 5. 结果分析（模型现在怎么样）

整体表现:

- 如果看 accuracy，basic_char 更高（0.7312）
- 如果看 F1，advanced 更好（0.4050）

这说明:

- basic_char 更偏向预测多数类，整体正确率高一些，但对正类召回偏低
- advanced 增强了对复杂模式的建模能力，正类召回提升，F1 更均衡

从作业要求角度看，你的 notebook 已满足:

- 使用了两种 tokenizer，并展示了词表大小变化
- 模型采用 EmbeddingBag
- 分类头包含至少 2 个隐藏层
- 报告了 accuracy / precision / recall / F1

## 6. 后续可改进方向（不改变作业核心框架）

- 对 advanced tokenizer 调整短语词典阈值（min_freq）与最大词长
- 在训练中加入轻量 class weight 以提高正类召回
- 进行 3~5 次不同随机种子重复实验，报告均值与方差，提升结果可信度
- 在不改模型大框架前提下，尝试更细的学习率调度和早停策略
