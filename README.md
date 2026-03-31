# CS310 自然语言处理课程资料

南方科技大学 CS310 自然语言处理课程的学习资料库。

## 📚 课程内容

本仓库包含以下内容：

### 📝 作业 (Coding Assignments)
- **A1**: 神经网络文本分类
  - 使用神经网络进行文本分类任务
  - 数据集：train.jsonl, test.jsonl
  - 产出：模型权重、实验报告与提交材料

- **A2**: Word2Vec词向量训练与调优
  - 基于莎士比亚语料训练词向量
  - 包含基础版与调优版notebook（`A2_w2v.ipynb`、`A2_w2v_tuned.ipynb`）
  - 数据与评估：`shakespeare.txt`、`questions-words-shakespeare.csv`、多组embeddings文件
  - 产出：实验报告与可视化结果

### 🧪 实验课 (Lab)
- **Lab 1**: Python基础文本处理
  - 文本清洗与预处理
  - 正则表达式应用
  - 中文文本分析（使用《三体3：死神永生》作为示例数据）
  - 词频统计与词汇表构建

- **Lab 2**: 神经网络文本分类
  - PyTorch深度学习框架
  - 使用torchtext进行文本处理
  - 情感分析任务（SST2数据集）
  - 神经网络模型训练与评估

- **Lab 3**: Word2Vec词向量训练与应用
  - 使用语料进行词向量训练
  - 词向量文件加载与相似度分析
  - 基于《论语》语料的分布式表示实验

- **Lab 4**: Transformer模型实验
  - Transformer核心结构实践
  - 序列建模与实验调参
  - 配套工具函数与实验notebook

- **Lab 5**: BPE与预训练实践
  - BPE子词切分实验（`lab5_bpe.ipynb`）
  - 预训练流程实验（`lab5_pretrain.ipynb`）
  - 使用 `the-verdict.txt` 进行文本实验
  - 配套工具函数与实验notebook

### 📖 课程讲义 (Lecture)
- 基础文本处理相关讲义
- `01-basic_text_proc.pdf`

## 🛠️ 技术栈

- **Python 3.x**
- **PyTorch** - 深度学习框架
- **torchtext** - 文本处理库
- **Gensim** - Word2Vec与词向量训练
- **Hugging Face datasets** - 数据集加载
- **正则表达式 (re)** - 文本处理
- **Jupyter Notebook** - 实验记录与结果展示

## 📋 环境配置

建议使用虚拟环境：

```bash
# 激活虚拟环境
.\venv\Scripts\Activate.ps1

# 安装必要的包
pip install torch torchtext datasets gensim
```

## 🧹 LaTeX 文档清理

`writing assignment` 目录提供了一个仓库内置脚本用于编译并清理中间文件，适合在保留 `.tex` 和 `.pdf` 的同时删除 `aux/log/out` 等临时产物。

推荐命令（在仓库根目录执行）：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File ".vscode/latex_build_clean.ps1" "E:/CS310-Natural-Language-Processing/writing assignment/A2" "A2_w2v_written_solution.tex" "A2_w2v_written_solution"
```

参数说明：
- 第一个参数：`.tex` 所在目录（建议使用绝对路径）
- 第二个参数：`.tex` 文件名（含扩展名）
- 第三个参数：文件主名（不含扩展名）

注意：第一个参数需精确到 `.tex` 所在子目录（本例为 `writing assignment/A2`）；如果使用相对路径，可能导致清理步骤无法命中目标文件。

## 📂 目录结构

```
CS310-Natural-Language-Processing/
├── coding assignment/
│   ├── A1/
│   │   ├── A1 report.md
│   │   ├── A1_nn.ipynb
│   │   ├── best_advanced_model.pth
│   │   ├── train.jsonl
│   │   ├── test.jsonl
│   │   └── submit_material/
│   │       └── A1_nn.ipynb
│   └── A2/
│       ├── A2_w2v.ipynb
│       ├── A2_w2v_tuned.ipynb
│       ├── report.md
│       ├── report.pdf
│       ├── shakespeare.txt
│       ├── questions-words-shakespeare.csv
│       ├── embeddings.txt
│       ├── embeddings_set1.txt
│       ├── embeddings_set2_ember.txt
│       ├── utils.py
│       └── *.png
├── lab/
│   ├── lab1/
│   │   ├── lab1_text_processing.ipynb
│   │   ├── text_classification_model.pth
│   │   └── 三体3死神永生-刘慈欣.txt
│   ├── lab2/
│   │   ├── data_utils.py
│   │   └── lab2_nn.ipynb
│   ├── lab3/
│   │   ├── lab3_w2v.ipynb
│   │   ├── embeddings.txt
│   │   ├── lunyu_20chapters.txt
│   │   └── utils.py
│   ├── lab4/
│       ├── lab4_trm.ipynb
│       └── utils.py
│   └── lab5/
│       ├── lab5_bpe.ipynb
│       ├── lab5_pretrain.ipynb
│       ├── the-verdict.txt
│       └── utils.py
├── lecture/
│   └── 01-basic_text_proc.pdf
├── writing assignment/
│   └── A2/
│       ├── A2_w2v_written_solution.tex
│       ├── A2_w2v_written_solution.pdf
│       └── A2_w2v_written.pdf
├── README_MAINTENANCE.md
└── README.md
```

## ⚠️ 学术诚信声明

**本仓库内容仅供学习参考，请勿直接抄袭用于作业提交。**

这些资料旨在帮助理解自然语言处理的基本概念和实现方法。如果你也在学习相关课程，请：

- ✅ 参考学习思路和方法
- ✅ 理解代码实现原理
- ✅ 在理解的基础上独立完成作业
- ❌ 直接复制代码提交作业
- ❌ 未经思考地照搬解决方案

学术诚信是每位学生的责任。抄袭不仅违反学校规定，更会影响自己的学习效果。

## 📅 更新日志

- **2026-03-31**: 根据维护规范补充Lab 5说明，更新目录结构（含writing assignment），同步README日期
- **2026-03-25**: 增加 LaTeX 清理说明，补充仓库脚本用法与参数注意事项
- **2026-03-24**: 根据维护规范更新README，补充A2与Lab4内容，修正目录结构与技术栈说明
- **2026-03-17**: 同步仓库最新结构，补充Lab 3内容，更新技术栈与目录说明
- **2026-03-10**: 初始化README，包含Lab 1, Lab 2和Assignment 1的内容

## 📄 License

本项目采用 [MIT License](LICENSE) 开源协议。

虽然代码开源，但请遵守学术诚信原则，不要将本仓库内容直接用于课程作业提交。

## 📧 联系方式

如有问题或建议，欢迎通过Issue进行讨论交流。

---

**SUSTech CS310 - Natural Language Processing**  
*Last Updated: 2026年3月31日*
