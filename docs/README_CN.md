# 谣言检测系统

上海交通大学 NIS4307 *人工智能导论* 课程项目。

[English](https://github.com/ZhongchunChen/NIS4307-01/blob/main/README.md)

## 项目简介

本项目结合 **BERTweet 微调模型**、**RAG 检索增强模块**与 **LLM 多阶段分析**，构建一个可解释的谣言检测系统。用户输入一段英文文本后，系统首先调用机器学习模型给出 0/1 二分类结果和置信度，同时调用 RAG 模块从 ChromaDB 知识库中检索相关证据，再由 LLM 进行独立判断、比较 ML 与 LLM 的结果，并分析两者一致或分歧的原因。最终系统通过可视化界面展示 ML 判语、LLM 判语、置信度、判断依据、比较分析以及 RAG 检索证据。

## 项目结构

```
NIS4307-01/
├── RAG/                         # RAG 模块与 ChromaDB 检索代码
│   ├── chroma_retriever.py       # ChromaDB 检索器
│   ├── config.py                 # RAG 配置
│   ├── em.py                     # 向量化 / embedding 相关代码
│   ├── llm_judge.py              # RAG 独立 LLM 判断入口
│   ├── main.py                   # RAG 独立运行入口
│   ├── requirements.txt          # RAG 模块依赖
│   └── train.csv                 # RAG 示例 / 辅助数据
├── configs/
│   └── bertweet.yaml             # 模型与训练配置
├── docs/
│   ├── frontend.md               # 前端架构说明
│   ├── model.md                  # 模型训练文档
│   └── README_CN.md              # 中文 README
├── outputs/bertweet/             # 训练指标、曲线和评估结果
├── src/
│   ├── config.py                 # 配置加载（.env + YAML）
│   ├── model.py                  # BERTweet 模型封装
│   ├── api_service.py            # LLM 多阶段分析服务
│   ├── rag_service.py            # 主系统调用 RAG 检索证据
│   ├── training/
│   │   ├── data.py               # 数据加载与清洗
│   │   ├── metrics.py            # 分类指标
│   │   ├── pipeline.py           # 训练、评估和绘图逻辑
│   │   └── utils.py              # 训练工具函数
│   └── web/
│       ├── app.py                # FastAPI 应用工厂
│       └── templates/
│           └── index.html        # Jinja2 前端页面
├── scripts/                       # 训练、评估、预测和绘图 CLI 包装
├── .example.env                  # 环境变量模板
├── .gitignore                    # Git 忽略规则
├── README.md                     # 项目说明
├── environment.yml               # Conda 环境配置
├── main.py                       # 统一入口
├── pyproject.toml                # 项目配置
├── requirements.txt              # Python 依赖
└── report.pdf                    # 课程报告
```

## 架构概览

### 整体流程

```
用户输入语句
    │
    ▼
┌──────────────────────────────────────────────────────┐
│  Step 1   classify_statement()     ← BERTweet 模型   │
│           返回 {is_rumor, confidence}                │
├──────────────────────────────────────────────────────┤
│  Step 2   retrieve_rag_evidence()  ← RAG / ChromaDB  │
│           检索相关证据 evidence                      │
├──────────────────────────────────────────────────────┤
│  Step 3   judge_statement()        ← LLM Stage 1     │
│           LLM 结合文本和 RAG 证据进行独立判断         │
│           返回 {is_rumor, label, reasoning,          │
│                 supporting_indicators}               │
├──────────────────────────────────────────────────────┤
│  Step 4   compare_results()        ← LLM Stage 2     │
│           LLM 比较 ML 与自身结果                     │
│           返回 {agreement, comparison_summary}       │
├──────────────────────────────────────────────────────┤
│  Step 5   analyze_root_cause()     ← LLM Stage 3     │
│           一致 → 综合解释                             │
│           分歧 → 分析原因                             │
│           返回 {root_cause_analysis,                  │
│                 key_indicators}                       │
└──────────────────────────────────────────────────────┘
    │
    ▼
渲染 HTML → 返回用户
```

### ML 模型

基于 `vinai/bertweet-base` 微调的二分类器，架构如下：

```
原始文本 → BERTweet tokenizer → BERTweet encoder → dropout → 线性分类头 → [rumor / not rumor]
```

模型输出两个类别的 softmax 概率，预测标签为 `0`（非谣言）或 `1`（谣言），置信度取预测类别的概率值。

### RAG 检索增强模块

RAG 模块使用 ChromaDB 向量数据库进行相似证据检索。用户输入文本后，系统会调用 `src/rag_service.py`，从 `RAG/` 目录下的 ChromaDB 数据库中检索若干条相关证据。检索结果会被传入 LLM prompt，作为生成判断依据和分歧分析时的参考信息。

RAG 模块不直接改变 BERTweet 模型输出的 0/1 分类结果，而是作为解释增强模块，为 LLM 提供外部证据或相似样本。若 RAG 数据库、依赖或本地路径不可用，系统会自动返回空 evidence，主流程仍可退化为原有的 BERTweet + LLM 分析流程。

### LLM 分析（三阶段）

| 阶段 | 函数 | 作用 |
|------|------|------|
| Stage 1 | `judge_statement()` | LLM 结合输入文本与 RAG 检索证据，从可验证性、信源可信度、逻辑连贯性、情绪化语言、具体性等维度分析 |
| Stage 2 | `compare_results()` | 将 ML 结果与 LLM 独立判断提交给 LLM 比较，判定一致或分歧 |
| Stage 3 | `analyze_root_cause()` | 一致时综合解释，分歧时分析误导原因 |

### 前端展示

- 三列判语栏：ML Verdict | LLM Verdict（分歧时显示黄色 ⚠） | Confidence
- 双列详情：LLM Analysis（判据 + 支撑线索）| Comparison（比较摘要 + 根因分析 + 关键指标）
- RAG Retrieved Evidence：展示 RAG 检索到的证据来源、标签、距离和文本内容

## 训练

### 数据准备

在 `datasets/` 下放置 `train.csv` 和 `val.csv`，需包含 `text` 和 `label` 两列：

```csv
text,label
"This appears to be a rumor",1
"This is verified news",0
```

### 配置文件

编辑 `configs/bertweet.yaml`，可调整超参数、路径等。关键配置项：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `data.train_path` | `datasets/train.csv` | 训练集路径 |
| `data.val_path` | `datasets/val.csv` | 验证集路径 |
| `training.epochs` | 16 | 训练轮数 |
| `training.learning_rate` | 1e-5 | 学习率 |
| `training.early_stopping_metric` | `macro_f1` | 早停监控指标 |
| `training.early_stopping_patience` | 2 | 早停 patience |

### 开始训练

```bash
python main.py --train
```

首次运行会自动从 Hugging Face 下载 `vinai/bertweet-base` 并缓存到 `checkpoints/base/`。训练完成后，最佳模型保存在 `checkpoints/bertweet/best_model/`，训练曲线和指标保存在 `outputs/bertweet/`。

### 评估模型

```bash
python -m src.evaluate --config configs/bertweet.yaml
```

### CLI 预测

```bash
python -m src.predict --config configs/bertweet.yaml --text "示例文本"
```

## RAG 模块运行说明

RAG 代码位于 `RAG/` 目录下，主系统已经通过 `src/rag_service.py` 将其作为可选辅助模块接入 LLM 分析流程。

由于 ChromaDB 向量数据库文件较大，未直接提交到仓库中，而是通过 GitHub Release 提供。运行 RAG 功能前，需要在 Releases 页面下载：

```
ChromaDB_data_populate.zip
```

下载后将其解压到 `RAG/` 目录下。解压后的目录结构应类似：

```
RAG/
├── chroma_retriever.py
├── config.py
├── main.py
├── requirements.txt
├── ...
└── ChromaDB_data_populate/
    └── DataBase/
        └── data/
```

安装 RAG 依赖：

```bash
cd RAG
pip install -r requirements.txt
```

独立运行 RAG 模块：

```bash
python main.py
```

在主系统中启用 RAG 时，只需确保 `RAG/ChromaDB_data_populate/DataBase/data` 路径存在。若该数据库不存在，主系统不会崩溃，但页面中不会展示 RAG 检索证据。

## 环境配置

### 安装依赖

Conda（推荐）：

```bash
conda env create -f environment.yml
conda activate intro2ai
```

或 pip：

```bash
pip install -r requirements.txt
```

如果需要独立运行 RAG 模块，还需进入 `RAG/` 目录安装其依赖：

```bash
cd RAG
pip install -r requirements.txt
```

### 配置 LLM API

复制 `.example.env` 为 `.env`，填入实际值：

```
API=https://models.sjtu.edu.cn/api/v1
API_SECRET=your-api-key
API_MODEL=deepseek-reasoner
```

配置指引见 https://claw.sjtu.edu.cn/guide/sjtu-api/

注意：`.env` 仅用于本地运行，请勿提交到 GitHub。

### 启动前端

```bash
python main.py
```

默认监听 `0.0.0.0:8000`，浏览器访问 `http://localhost:8000`。

若已正确解压 RAG 数据库，前端分析结果中会显示 **RAG Retrieved Evidence** 区域；若未配置 RAG 数据库，系统仍可正常完成 BERTweet + LLM 分析。

### CLI 选项

```
python main.py            默认启动前端
python main.py --display  启动前端
python main.py --train    训练模型
python main.py --help     查看帮助
```
