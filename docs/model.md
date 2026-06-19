# 机器学习模型

## 任务定义

本项目使用 `vinai/bertweet-base` 训练二分类文本分类器：

```text
输入：英文短文本 text
输出：label 0（非谣言）或 1（谣言）
```

模型只使用数据中的 `text` 列，不使用 `id`、`event` 等字段。数据主要来自社交媒体短文本、事实核查声明和新闻文本，目标是判断文本更接近真实信息还是虚假或谣言信息。

主数据集采用 PHEME 风格的谣言检测标签，描述一条信息在发布时是否未经证实，而不是最终真假：

- 主数据集：`0 -> 非谣言`、`1 -> 谣言`

额外数据集映射到同一分类方向：

- GossipCop：`R -> 0`、`F -> 1`
- FEVER/shared task：`SUPPORTS -> 0`、`REFUTES -> 1`

训练和验证 CSV 至少包含：

```text
text,label
```

## 为什么选择 BERTweet

BERTweet 是基于 Transformer Encoder、面向英文 Twitter 文本预训练的模型。其底层结构与 RoBERTa-base 类似，但预训练语料来自大规模英文推文。

### 文本风格匹配

原始数据包含 URL、hashtag、mention、缩写、口语表达和非正式语法。BERTweet 的预训练数据与这些表达形式更接近。

### 适合小样本

当前主训练集规模较小。从零训练 TextCNN、LSTM 或 Transformer 容易过拟合。BERTweet 已学习通用语言表示，微调时主要适配本任务的标签边界。

### 双向上下文建模

真假判断通常依赖完整上下文。Transformer Encoder 的 self-attention 允许每个 token 同时关注左右两侧的信息。

### 适合二分类微调

BERTweet 可以通过轻量分类头转换为二分类器。本项目采用端到端微调，encoder 和分类头都会更新。

## 整体模型结构

项目通过 `AutoModelForSequenceClassification` 加载 `vinai/bertweet-base` 并添加二分类头。

![BERTweet 二分类整体架构](assets/bertweet_architecture.svg)

```text
原始文本
  -> BERTweet tokenizer
  -> 预训练 BERTweet Transformer encoder
  -> dropout
  -> 线性分类头
  -> label 0 或 1 的 logits
```

训练时，encoder 与分类头一起微调。

### BERTweet Encoder

BERTweet base 采用 Transformer Encoder 架构。它不是自回归生成模型，而是双向编码模型。也就是说，模型在编码某个 token 时，可以同时利用该 token 左右两侧的上下文。

典型 base 级别结构为：

```text
Embedding Layer
  -> Transformer Encoder Layer 1
  -> Transformer Encoder Layer 2
  -> ...
  -> Transformer Encoder Layer N
  -> Contextual token representations
```

每一层 Transformer Encoder 都包含两个核心模块：

1. Multi-Head Self-Attention
2. Feed-Forward Network

并配合残差连接与 Layer Normalization。

### Transformer Encoder Block

单个 Transformer Encoder Block 的结构如下：

![Transformer Encoder Block](assets/transformer_encoder_block.svg)

#### Self-Attention

Self-attention 的作用是为每个 token 动态聚合上下文信息。对于一句文本中的每个 token，模型会计算它与其他 token 的相关性，然后根据相关性加权汇总上下文。

例如：

```text
"Breaking news: ..."
"rumor says ..."
"police confirms ..."
```

这些短语对真假判断的贡献不同。Self-attention 可以让模型自动学习哪些词、短语或上下文组合更重要。

#### Multi-Head Attention

Multi-head attention 会并行学习多组注意力模式。不同 attention head 可能关注不同信息：

- 实体关系
- 否定词
- 情绪化表达
- 来源可信度相关词
- 时间、地点、事件描述
- hashtag 或社交媒体表达

这比单一注意力头更适合处理复杂文本信号。

#### Feed-Forward Network

Self-attention 负责 token 之间的信息交互，Feed-Forward Network 负责对每个 token 的表示进行非线性变换。它增强了模型表达能力，使模型不仅能聚合上下文，还能学习更复杂的语义特征。

#### Residual、LayerNorm 与 Dropout

残差连接和 LayerNorm 让深层 Transformer 更容易训练：

- 残差连接保留原始信息，缓解梯度消失
- LayerNorm 稳定不同层之间的数值分布
- Dropout 减少过拟合

这些设计对小数据微调尤其重要。

### Tokenizer 与输入表示

BERTweet tokenizer 将原始文本拆分为子词 token，适合处理 hashtag、mention、URL、缩写、拼写变化和未登录词。

模型输入通常包含：

```text
input_ids
attention_mask
```

- `input_ids`：token 对应的词表编号
- `attention_mask`：区分真实 token 与 padding

当前配置使用：

```yaml
max_length: 128
```

该长度适合推文短文本。较长的 GossipCop 文本会被截断，因此数据清洗时将 `title` 和 `description` 放在正文前，以保留信息密度更高的内容。

### 分类头与预测

分类时，模型使用第一个特殊 token 的隐藏状态表示整段文本，再输入分类头：

```text
first-token hidden state
  -> dropout
  -> dense layer
  -> activation
  -> dropout
  -> linear layer
  -> logits for 2 classes
```

输出 logits 的形状为 `[batch_size, 2]`。训练使用 Cross Entropy Loss，预测通过 `argmax(logits)` 得到最终标签。

### 微调方式与数据适配

项目采用端到端微调：

```text
BERTweet encoder parameters: updated
classification head parameters: updated
```

这种方式不仅训练标签映射，还使 encoder 的表示适应事实性、表达方式、事件描述和社交媒体语境。

#### 原始推文数据

原始训练集包含 hashtag、URL 和事件关键词，与 BERTweet 的预训练语料风格接近。

#### 公开额外数据

额外数据包括 GossipCop 新闻或娱乐谣言数据，以及 shared task/FEVER 风格事实核查 claim 数据。它们能够扩展真假判断样本，但与原始推文存在分布差异，可能造成 domain shift。因此应根据验证集指标判断效果，并检查过拟合与类别偏向。

#### 小数据场景

```text
少量标注数据
  + 大规模预训练语言知识
  -> 更稳健的文本表示
```

这是项目选择 BERTweet 微调而非从零训练神经网络的主要原因。

## 项目文件与输出

```text
.
├── configs/
│   └── bertweet.yaml          # 模型与训练配置
├── datasets/
│   ├── train.csv              # 训练集
│   ├── val.csv                # 验证集
│   └── all_extra.parquet      # 合并后的可选扩展数据
├── checkpoints/
│   ├── base/                  # 缓存的 BERTweet 预训练权重
│   └── bertweet/              # 微调后的模型权重
├── outputs/
│   └── bertweet/              # 指标、日志和训练曲线
├── src/
│   ├── config.py              # 配置读取
│   ├── model/
│   │   ├── data.py            # 数据读取与清洗
│   │   ├── inference.py       # 模型推理封装
│   │   ├── metrics.py         # 分类指标
│   │   ├── pipeline.py        # 训练、评估和绘图逻辑
│   │   └── utils.py           # 模型工具函数
│   └── cli/                   # 统一入口调用的命令实现
├── tests/                     # CLI 与配置回归测试
├── main.py                    # 统一命令行入口
├── pyproject.toml             # 项目与依赖配置
└── environment.yml            # 可选 Conda 环境文件
```

模型与训练输出位置：

```text
checkpoints/base/
checkpoints/bertweet/best_model/
outputs/bertweet/history.json
outputs/bertweet/best_metrics.json
outputs/bertweet/cleaning_report.json
outputs/bertweet/evaluation_metrics.json
outputs/bertweet/test_metrics.json
outputs/bertweet/plots/
```

`configs/bertweet.yaml` 是数据路径、模型参数和训练参数的默认来源。`main.py` 的命令行参数可覆盖部分运行设置，例如设备、epoch 数、输出目录和 checkpoint 目录。

## 环境配置与训练

### 环境配置

推荐根据仓库提供的环境文件创建 Conda 环境。该文件会安装 Python 3.10，并以 editable 模式安装当前项目：

```bash
conda env create -f environment.yml
conda activate intro2ai
```

依赖发生变化后，可更新已有环境：

```bash
conda env update -f environment.yml --prune
```

也可以在虚拟环境中直接从 `pyproject.toml` 安装：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

Windows PowerShell 使用 `.\.venv\Scripts\Activate.ps1` 激活虚拟环境。

检查环境：

```bash
python -m pip check
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

### 训练

在项目根目录运行：

```bash
python main.py train --config configs/bertweet.yaml
```

首次运行可能从 Hugging Face 下载 `vinai/bertweet-base` 并缓存至 `checkpoints/base/`，后续训练优先使用本地缓存。

默认训练参数来自 `configs/bertweet.yaml`。例如，可临时覆盖训练轮数和设备：

```bash
python main.py train \
  --config configs/bertweet.yaml \
  --epochs 8 \
  --device cuda
```

最佳模型根据验证集 `macro_f1` 选择并保存至 `checkpoints/bertweet/best_model/`。当该指标连续若干轮没有改善时，训练按照 `early_stopping_patience` 提前停止。

后续评估、预测和训练曲线绘制应继续传入训练时使用的同一个配置文件。当前实现不会根据 checkpoint 自动切换到对应配置；省略 `--config` 时会回退到 `configs/bertweet.yaml`。

### 评估

评估最佳 checkpoint：

```bash
python main.py evaluate --config configs/bertweet.yaml
```

该命令默认评估配置中的验证集，并将结果写入 `outputs/bertweet/evaluation_metrics.json`。

评估指定 checkpoint：

```bash
python main.py evaluate \
  --config configs/bertweet.yaml \
  --checkpoint checkpoints/bertweet/best_model
```

评估包含 `text,label` 列的自定义测试集：

```bash
python main.py evaluate \
  --config configs/bertweet.yaml \
  --test datasets/my_test.csv \
  --output outputs/bertweet/my_test_metrics.json
```

自定义测试集未指定 `--output` 时，结果默认保存为 `outputs/bertweet/test_metrics.json`。评估输出包含 macro-F1、accuracy 和 confusion matrix。

### 单条预测

```bash
python main.py predict \
  --config configs/bertweet.yaml \
  "Example tweet text"
```

使用 `--json` 可输出便于程序处理的 JSON：

```bash
python main.py predict \
  --config configs/bertweet.yaml \
  --json \
  "Example tweet text"
```

### 绘制训练曲线

根据 `outputs/bertweet/history.json` 重新生成曲线：

```bash
python main.py plot-history --config configs/bertweet.yaml
```

该命令默认读取 `outputs/bertweet/history.json`。也可以显式指定输入历史文件和绘图输出目录：

```bash
python main.py plot-history \
  --config configs/bertweet.yaml \
  --history outputs/bertweet/history.json \
  --output-dir outputs/bertweet
```

生成文件包括：

```text
outputs/bertweet/plots/loss_curve.png
outputs/bertweet/plots/accuracy_curve.png
outputs/bertweet/plots/macro_f1_curve.png
outputs/bertweet/plots/grad_norm_curve.png
```
