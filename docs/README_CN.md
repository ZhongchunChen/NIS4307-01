# 谣言检测系统

**文档**：[English README](../README.md)

**最终报告**：$\LaTeX$ 源文件 `report.tex` 与编译后的 `report.pdf` 位于 `/report` 目录。请以 main 分支的最新提交为准。

**目录**：
1. [项目简介](#项目简介)
2. [环境安装](#环境安装)
3. [准备模型](#准备模型)
4. [可选的 LLM 与 RAG 配置](#可选的-llm-与-rag-配置)
5. [运行 Web 演示](#运行-web-演示)
6. [使用自定义测试集评估](#使用自定义测试集评估)
7. [项目结构](#项目结构)
8. [命令概览](#命令概览)

## 项目简介

本项目使用微调后的 BERTweet 模型对英文语句进行谣言二分类。Web 界面还可以选择从 ChromaDB 检索相关证据，并调用兼容 OpenAI API 的大语言模型解释预测结果、比较 ML 与 LLM 的判断，以及分析两者一致或分歧的原因。

各组件在可行范围内保持独立：模型训练与评估只依赖 ML 模型；LLM 与 RAG 可用于完整的可解释演示。模型设计、实验结果和技术分析请参阅[最终报告](../report/report.pdf)及 `docs/` 下的模块文档。

## 环境安装

推荐使用 Conda 创建 Python 3.10 环境：

```bash
conda env create -f environment.yml
conda activate intro2ai
```

依赖发生变化后，可更新已有环境：

```bash
conda env update -f environment.yml --prune
```

也可以使用 Python 虚拟环境安装：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

在 Windows PowerShell 中，使用 `\.venv\Scripts\Activate.ps1` 激活环境。

## 准备模型

真实推理与评估优先使用以下路径中本地训练得到的检查点：

```text
checkpoints/bertweet/best_model/
```

如果该目录不存在，项目会下载 `model.huggingface_checkpoint` 配置的微调检查点。本地训练成功后生成的检查点会自动优先使用。训练数据至少需要包含 `text` 和 `label` 两列，其中 `0` 表示非谣言，`1` 表示谣言。

```csv
text,label
"This claim is confirmed by an official source.",0
"A celebrity secretly died yesterday.",1
```

数据路径采用本地优先策略。如果配置的文件不存在，项目会根据 `configs/bertweet.yaml` 中的 `data.huggingface` 映射，从指定数据集仓库下载对应的 Parquet 文件，并复用 Hugging Face 缓存。

使用 `configs/bertweet.yaml` 中的默认配置训练：

```bash
python main.py train
```

预训练模型 `vinai/bertweet-base` 会缓存在 `checkpoints/base/`。系统按验证集 macro-F1 选择最佳模型，并保存至 `checkpoints/bertweet/best_model/`；指标、训练历史和曲线保存在 `outputs/bertweet/`。

## 可选的 LLM 与 RAG 配置

本节只适用于完整的可解释演示。ML-only 演示与模型评估不需要 API 密钥或 RAG 数据库。

### 配置 LLM

复制环境变量模板：

```bash
cp .example.env .env
```

Windows PowerShell 使用 `Copy-Item .example.env .env`。

在 `.env` 中填写兼容 OpenAI API 的配置：

```text
API=https://models.sjtu.edu.cn/api/v1
API_SECRET=your-api-key
API_MODEL=deepseek-reasoner
```

### 启用 RAG

项目首先在以下位置查找数据库：

```text
datasets/ChromaDB_data_populate/DataBase/data
```

如果本地数据库不存在，系统会从配置的 Hugging Face 仓库下载 `ChromaDB_data_populate.zip`（默认仓库为 `MingchenDai/NIS4307-ChromaDB_data_populate`），安全解压到上述位置，并在后续运行中复用。可通过 `RAG_CHROMA_HF_REPO_ID`、`RAG_CHROMA_HF_REVISION`、`RAG_CHROMA_HF_REPO_TYPE` 和 `RAG_CHROMA_HF_ARCHIVE` 修改仓库及文件配置。

RAG 为可选功能。数据库无法下载或加载时，Web 应用仍可在没有检索证据的情况下运行。

## 运行 Web 演示

启动前请确认环境已激活。如果需要查看真实的 ML 预测，还必须准备好微调后的检查点。

运行完整的 ML、LLM 与可用 RAG 流程：

```bash
python main.py serve
```

浏览器访问 `http://localhost:8000`。

无需配置 LLM 和 RAG 的 BERTweet-only 演示：

```bash
python main.py serve --no-llm --no-rag
```

> **检查点规则**：系统优先使用本地训练的 `checkpoints/bertweet/best_model/`。如果该目录不存在，则下载并缓存配置的 Hugging Face 检查点。只有本地和远程检查点都无法解析时，Web 界面才会使用 mock 分类器。

## 使用自定义测试集评估

该流程用于按照课程评分要求，在自行构造的数据集上评估模型。评估只使用训练后的 BERTweet 检查点，不调用 LLM API 或 RAG 模块。

创建至少包含 `text` 和 `label` 两列的 CSV：

```csv
text,label
"This report cites a verifiable government announcement.",0
"Scientists secretly confirmed an impossible cure overnight.",1
```

CSV 可以包含额外列。缺少文本、缺少标签或标签不属于 `0`、`1` 的行会在评估时被排除。

使用配置中的最佳检查点评估：

```bash
python main.py evaluate --test datasets/my_test.csv
```

如果本地路径不存在，`--test` 也会使用配置的 Hugging Face 映射。例如，`--test datasets/val.csv` 会获取远程 validation Parquet 文件；已存在的本地文件始终优先。若需使用其他远程测试文件，请先在 `data.huggingface.files` 中添加映射。

也可以指定检查点和输出文件：

```bash
python main.py evaluate \
  --test datasets/my_test.csv \
  --checkpoint checkpoints/bertweet/best_model \
  --output outputs/bertweet/my_test_metrics.json
```

命令会输出 macro-F1、accuracy 和 confusion matrix。未指定 `--output` 时，自定义测试结果保存在 `outputs/bertweet/test_metrics.json`。

## 项目结构

```text
configs/        模型与训练配置
datasets/       本地训练、测试及 RAG 数据
docs/           模块文档与架构图片
report/         技术报告源文件与编译结果
src/model/      数据处理、训练、评估与推理
src/rag/        可选的 ChromaDB 检索与 RAG 工具
src/web/        FastAPI 应用与 Web 模板
src/cli/        命令实现与参数校验
tests/          CLI 与配置回归测试
main.py         统一命令行入口
```

## 命令概览

| 命令 | 用途 |
| --- | --- |
| `python main.py serve` | 启动 Web 演示 |
| `python main.py train` | 训练 BERTweet 模型 |
| `python main.py evaluate --test FILE.csv` | 评估带标签的测试集 |
| `python main.py predict "TEXT"` | 对单条语句分类 |
| `python main.py plot-history` | 重新生成训练曲线 |
| `python main.py rag query "TEXT"` | 运行独立 RAG 分析 |

查看完整参数：

```bash
python main.py --help
python main.py COMMAND --help
```

## 技术文档

- [机器学习模型](model.md)：BERTweet 架构、数据准备、训练、评估、预测与输出文件
- [前端](frontend.md)：FastAPI/Jinja 架构、分析流程、环境配置与 ML 接口约定
- [技术报告](../report/)：详细的模型设计、实验、结果与分析
- [英文 README](../README.md)：英文项目安装与使用说明
