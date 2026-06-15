# BERTweet Binary Text Classifier

## Overview

This project fine-tunes `vinai/bertweet-base` for binary text classification.
The input is a tweet-like English text, and the output label is `0` or `1`.

The model uses only the `text` column as input. Columns such as `id` and `event`
are not used by the classifier.

## Model

The model architecture is:

```text
raw text
  -> BERTweet tokenizer
  -> pretrained BERTweet base encoder
  -> dropout
  -> linear classification head
  -> label 0 or 1
```

The training script uses `AutoModelForSequenceClassification`. It loads the
pretrained BERTweet encoder and adds a binary classification head. During
training, both the encoder and the classification head are fine-tuned.

The original BERTweet base checkpoint is cached under:

```text
checkpoints/base/
```

The fine-tuned best model is saved under:

```text
checkpoints/bertweet/best_model/
```

Training logs, metrics, and plots are saved under:

```text
outputs/bertweet/
```

## Project Structure

```text
.
├── configs/
│   └── bertweet.yaml          # training and model configuration
├── datasets/
│   ├── train.csv              # training set
│   ├── val.csv                # validation set
│   └── extra/                 # converted extra training CSV files
├── checkpoints/
│   ├── base/                  # cached pretrained BERTweet checkpoint
│   └── bertweet/              # fine-tuned model checkpoints
├── outputs/
│   └── bertweet/              # metrics, logs, and training curves
├── src/
│   ├── config.py              # config loading
│   ├── data.py                # dataset loading and cleaning
│   ├── metrics.py             # classification metrics
│   ├── training.py            # reusable training and evaluation logic
│   └── utils.py               # shared utilities
├── scripts/
│   ├── evaluate.py            # evaluation entry
│   ├── plot_history.py        # plot curves from history.json
│   ├── predict.py             # single-text prediction entry
│   ├── prepare_extra_datasets.py
│   └── train.py               # training entry
├── pyproject.toml             # editable package and dependencies
└── environment.yml            # optional Conda environment file
```

The CSV files must contain at least:

```text
text,label
```

## Environment Setup

Create the Conda environment:

```bash
conda create -n intro2ai python=3.10 -y
conda activate intro2ai
```

Prevent user-site packages from leaking into the Conda environment:

```bash
conda env config vars set PYTHONNOUSERSITE=1 -n intro2ai
conda deactivate
conda activate intro2ai
```

Install this project and its dependencies in editable mode:

```bash
cd /path/to/introuduction_to_ai
python -m pip install -e .
```

Verify the environment:

```bash
python -m pip check
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

## Prepare Extra Datasets

Convert public datasets to trainable CSV files:

```bash
python scripts/prepare_extra_datasets.py
```

The script reads `datasets/gossipcop_*.parquet` and
`datasets/shared_task_dev.jsonl`, then writes:

```text
datasets/extra/gossipcop_extra.csv
datasets/extra/shared_task_extra.csv
datasets/extra/all_extra.csv
datasets/extra/extra_datasets_report.json
```

Label mapping:

```text
GossipCop: R -> 0, F -> 1, H/M kept as source metadata
Shared task: SUPPORTS -> 0, REFUTES -> 1, NOT ENOUGH INFO skipped
```

Extra training data is controlled in `configs/bertweet.yaml`:

```yaml
data:
  extra_datasets:
    enabled: true
    paths:
      - datasets/extra/gossipcop_extra.csv
      - datasets/extra/shared_task_extra.csv
```

## Train

Run training from the project root:

```bash
python scripts/train.py --config configs/bertweet.yaml
```

The first run may download `vinai/bertweet-base` from Hugging Face and save it
to `checkpoints/base/`. Later runs will prefer the local cached checkpoint.

Important output files:

```text
checkpoints/bertweet/best_model/
outputs/bertweet/history.json
outputs/bertweet/best_metrics.json
outputs/bertweet/cleaning_report.json
outputs/bertweet/plots/
```

## Evaluate

Evaluate the best checkpoint:

```bash
python scripts/evaluate.py --config configs/bertweet.yaml
```

To evaluate a specific checkpoint:

```bash
python scripts/evaluate.py \
  --config configs/bertweet.yaml \
  --checkpoint checkpoints/bertweet/best_model
```

## Predict

Predict one text:

```bash
python scripts/predict.py \
  --config configs/bertweet.yaml \
  --text "Example tweet text"
```

## Plot Curves

Regenerate training curves from `outputs/bertweet/history.json`:

```bash
python scripts/plot_history.py --config configs/bertweet.yaml
```

Generated plots:

```text
outputs/bertweet/plots/loss_curve.png
outputs/bertweet/plots/accuracy_curve.png
outputs/bertweet/plots/macro_f1_curve.png
outputs/bertweet/plots/grad_norm_curve.png
```

---

# BERTweet 二分类文本分类器

## 项目简介

本项目使用 `vinai/bertweet-base` 进行二分类文本分类微调。模型输入是一段英文短文本，输出标签为 `0` 或 `1`。

模型只使用数据中的 `text` 列作为输入，不使用 `id`、`event` 等字段。

## 模型结构

模型结构如下：

```text
原始文本
  -> BERTweet tokenizer
  -> 预训练 BERTweet base encoder
  -> dropout
  -> 线性分类头
  -> label 0 或 1
```

训练代码使用 `AutoModelForSequenceClassification`。它会加载预训练的 BERTweet encoder，并添加一个二分类分类头。训练时，encoder 和分类头都会一起微调。

原始 BERTweet base checkpoint 缓存在：

```text
checkpoints/base/
```

微调得到的最佳模型保存在：

```text
checkpoints/bertweet/best_model/
```

训练日志、指标和曲线保存在：

```text
outputs/bertweet/
```

## 项目结构

```text
.
├── configs/
│   └── bertweet.yaml          # 模型与训练配置
├── datasets/
│   ├── train.csv              # 训练集
│   ├── val.csv                # 验证集
│   └── extra/                 # 转换后的额外训练 CSV
├── checkpoints/
│   ├── base/                  # 缓存的 BERTweet 预训练权重
│   └── bertweet/              # 微调后的模型权重
├── outputs/
│   └── bertweet/              # 指标、日志和训练曲线
├── src/
│   ├── config.py              # 配置读取
│   ├── data.py                # 数据读取与清洗
│   ├── metrics.py             # 分类指标
│   ├── training.py            # 可复用训练与评估逻辑
│   └── utils.py               # 通用工具
├── scripts/
│   ├── evaluate.py            # 评估入口
│   ├── plot_history.py        # 根据 history.json 绘制曲线
│   ├── predict.py             # 单条文本预测入口
│   ├── prepare_extra_datasets.py
│   └── train.py               # 训练入口
├── pyproject.toml             # editable package 与依赖配置
└── environment.yml            # 可选 Conda 环境配置
```

CSV 文件至少需要包含：

```text
text,label
```

## 环境配置

创建 Conda 环境：

```bash
conda create -n intro2ai python=3.10 -y
conda activate intro2ai
```

防止用户目录中的 Python 包污染当前 Conda 环境：

```bash
conda env config vars set PYTHONNOUSERSITE=1 -n intro2ai
conda deactivate
conda activate intro2ai
```

以 editable 模式安装项目和依赖：

```bash
cd /path/to/introuduction_to_ai
python -m pip install -e .
```

检查环境：

```bash
python -m pip check
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

## 准备额外数据集

将公开数据集转换为可训练的 CSV：

```bash
python scripts/prepare_extra_datasets.py
```

脚本读取 `datasets/gossipcop_*.parquet` 和 `datasets/shared_task_dev.jsonl`，并输出：

```text
datasets/extra/gossipcop_extra.csv
datasets/extra/shared_task_extra.csv
datasets/extra/all_extra.csv
datasets/extra/extra_datasets_report.json
```

标签映射：

```text
GossipCop: R -> 0, F -> 1，H/M 作为 source 元数据保留
Shared task: SUPPORTS -> 0, REFUTES -> 1，NOT ENOUGH INFO 丢弃
```

是否加入额外训练数据由 `configs/bertweet.yaml` 控制：

```yaml
data:
  extra_datasets:
    enabled: true
    paths:
      - datasets/extra/gossipcop_extra.csv
      - datasets/extra/shared_task_extra.csv
```

## 训练

在项目根目录下运行：

```bash
python scripts/train.py --config configs/bertweet.yaml
```

首次运行可能会从 Hugging Face 下载 `vinai/bertweet-base`，并保存到 `checkpoints/base/`。之后训练会优先使用本地缓存。

主要输出文件：

```text
checkpoints/bertweet/best_model/
outputs/bertweet/history.json
outputs/bertweet/best_metrics.json
outputs/bertweet/cleaning_report.json
outputs/bertweet/plots/
```

## 评估

评估最佳 checkpoint：

```bash
python scripts/evaluate.py --config configs/bertweet.yaml
```

评估指定 checkpoint：

```bash
python scripts/evaluate.py \
  --config configs/bertweet.yaml \
  --checkpoint checkpoints/bertweet/best_model
```

## 预测

预测单条文本：

```bash
python scripts/predict.py \
  --config configs/bertweet.yaml \
  --text "Example tweet text"
```

## 绘制训练曲线

根据 `outputs/bertweet/history.json` 重新生成训练曲线：

```bash
python scripts/plot_history.py --config configs/bertweet.yaml
```

生成的图像包括：

```text
outputs/bertweet/plots/loss_curve.png
outputs/bertweet/plots/accuracy_curve.png
outputs/bertweet/plots/macro_f1_curve.png
outputs/bertweet/plots/grad_norm_curve.png
```
