# CS310 NLP Assignment 1 实验报告

## 1. 实验任务与文件说明

本实验任务为中文幽默检测二分类，目标是判断句子是否具有幽默性（标签 0/1）。

本报告结合以下 3 个文件完成：

1. `A1_nn.ipynb`：完整实验代码（数据处理、建模、训练、评估）
2. `train.jsonl`：训练数据
3. `test.jsonl`：测试数据

实验环境与固定设置：

- 框架：PyTorch
- 随机种子：310
- 设备：CPU

数据规模：

- 训练集：12677
- 验证集：1267（由训练集按 10% 划分）
- 测试集：651

## 2. 实验过程

### 2.1 数据读取与预处理

- 从 `train.jsonl`、`test.jsonl` 逐行读取样本。
- 每条样本保留字段：`sentence`、`label`、`id`。
- 标签处理方式：取 `label[0]` 并转为整数。

### 2.2 两种分词方案设计

为满足作业中“比较至少两种 tokenizer”的要求，实验实现了两套方案：

1. `basic_char`
- 仅保留中文字符，按单字切分。
- 丢弃英文、数字和标点。

2. `advanced`
- 使用正则先切分出英文串、数字串、中文串和标点。
- 对中文串再使用“短语词典 + 最大匹配”做进一步分词。
- 短语词典来自训练集统计，设置 `min_freq=8`，`max_word_len=4`。

词表构建结果：

- `basic_char` 词表大小：2687
- `advanced` 词表大小：7345

### 2.3 模型结构

模型采用 `EmbeddingBag + MLP`：

- Embedding 层：`nn.EmbeddingBag(vocab_size, 128, mode="mean")`
- 分类头：两层隐藏层全连接网络
	- `128 -> 128 -> 64 -> 2`
	- 激活函数：ReLU
	- Dropout：0.25

训练参数：

- 损失函数：CrossEntropyLoss
- 优化器：Adam（学习率 1e-3）
- 训练轮数：10 epochs
- 学习率调度：ReduceLROnPlateau（依据验证集 F1）
- 梯度裁剪：`max_norm=1.0`

参数规模：

- 可训练参数总量：965058

### 2.4 训练与模型选择策略

- 在每个 epoch 结束后，在验证集计算 `accuracy / precision / recall / f1`。
- 使用验证集 F1 最高时的模型参数作为该 tokenizer 的最佳模型。
- 最终在测试集报告结果，并以测试集 F1 最高者作为本次实验最佳方案。

## 3. 实验结果

![report result](report%20result.png)

测试集指标如下：

| tokenizer | vocab_size | accuracy | precision | recall | f1 | loss | tp | tn | fp | fn |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| basic_char | 2687 | 0.7312 | 0.4706 | 0.2353 | 0.3137 | 0.5587 | 40 | 436 | 45 | 130 |
| advanced | 7345 | 0.7066 | 0.4305 | 0.3824 | 0.4050 | 0.6854 | 65 | 395 | 86 | 105 |

最佳模型（按 F1）：

- `advanced`（F1 = 0.4050）
- 模型权重保存为：`best_advanced_model.pth`

## 4. 结果分析

1. 从 Accuracy 看，`basic_char` 更高（0.7312 > 0.7066）。
2. 从 F1 看，`advanced` 明显更优（0.4050 > 0.3137）。
3. `advanced` 的 Recall 更高（0.3824 > 0.2353），说明其对正类（幽默句）识别更充分。
4. `basic_char` 的策略更保守，负类判定更稳定，因此准确率较高，但漏检正类较多（FN=130）。
5. 综合 precision 与 recall 的平衡需求，本实验最终选择 `advanced` 作为最佳方案。
