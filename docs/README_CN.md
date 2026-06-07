# 谣言检测系统

上海交通大学 NIS4307 *人工智能导论* 课程项目。

[English](https://github.com/ZhongchunChen/NIS4307-01/blob/frontend/README.md)

## 项目简介

本项目结合 **BERTweet 微调模型**与 **LLM 多阶段分析**，构建一个可解释的谣言检测系统。用户输入一段英文文本后，系统同时调用机器学习模型和 LLM 进行独立判断，比较两者结果，分析分歧或一致原因，最后以可视化界面呈现完整分析。

## 项目结构

```
NIS4307-01/
├── main.py                       # 统一入口（训练 / 前端）
├── requirements.txt              # Python 依赖
├── environment.yml               # Conda 环境配置
├── .env                          # 环境变量（自行创建）
├── .example.env                  # 环境变量模板
├── configs/
│   └── bertweet.yaml             # 模型与训练配置
├── checkpoints/
│   ├── base/                     # 缓存的 BERTweet 预训练权重
│   └── bertweet/best_model/      # 微调后的最佳模型
├── outputs/bertweet/             # 训练指标、曲线和报告
├── datasets/
│   ├── train.csv                 # 训练集
│   └── val.csv                   # 验证集
├── docs/
│   ├── frontend.md               # 前端架构说明（中文）
│   ├── model.md                  # 模型训练文档
│   └── README_CN.md              # 中文 README
└── src/
    ├── app.py                    # FastAPI 应用工厂
    ├── config.py                 # 配置加载（.env + YAML）
    ├── model.py                  # 模型封装（lazy-load BERTweet）
    ├── api_service.py            # LLM 三阶段分析服务
    ├── data.py                   # 数据加载与清洗
    ├── train.py                  # 训练脚本
    ├── evaluate.py               # 评估脚本
    ├── predict.py                # 单文本预测 CLI
    ├── metrics.py                # 分类指标
    ├── plot_history.py           # 训练曲线绘制
    ├── utils.py                  # 通用工具
    └── templates/
        └── index.html            # Jinja2 前端页面
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
│  Step 2   judge_statement()        ← LLM Stage 1     │
│           LLM 独立判断（未看到 ML 结果）              │
│           返回 {is_rumor, label, reasoning,          │
│                 supporting_indicators}               │
├──────────────────────────────────────────────────────┤
│  Step 3   compare_results()        ← LLM Stage 2     │
│           LLM 比较 ML 与自身结果                     │
│           返回 {agreement, comparison_summary}       │
├──────────────────────────────────────────────────────┤
│  Step 4   analyze_root_cause()     ← LLM Stage 3     │
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

### LLM 分析（三阶段）

| 阶段 | 函数 | 作用 |
|------|------|------|
| Stage 1 | `judge_statement()` | LLM 独立判断，从可验证性、信源可信度、逻辑连贯性、情绪化语言、具体性五个维度分析 |
| Stage 2 | `compare_results()` | 将 ML 结果与 LLM 独立判断提交给 LLM 比较，判定一致或分歧 |
| Stage 3 | `analyze_root_cause()` | 一致时综合解释，分歧时分析误导原因 |

### 前端展示

- 三列判语栏：ML Verdict | LLM Verdict（分歧时显示黄色 ⚠） | Confidence
- 双列详情：LLM Analysis（判据 + 支撑线索）| Comparison（比较摘要 + 根因分析 + 关键指标）

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

### 配置 LLM API

复制 `.example.env` 为 `.env`，填入实际值：

```
API=https://models.sjtu.edu.cn/api/v1
API_SECRET=your-api-key
API_MODEL=deepseek-reasoner
```

配置指引见 https://claw.sjtu.edu.cn/guide/sjtu-api/

### 启动前端

```bash
python main.py
```

默认监听 `0.0.0.0:8000`，浏览器访问 `http://localhost:8000`。

### CLI 选项

```
python main.py            默认启动前端
python main.py --display  启动前端
python main.py --train    训练模型
python main.py --help     查看帮助
```
