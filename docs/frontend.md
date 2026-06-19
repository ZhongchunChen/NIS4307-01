# 前端

## 项目结构

```text
NIS4307-01/
├── main.py                  # 入口：创建应用并启动 uvicorn
├── requirements.txt         # Python 依赖
├── .env                     # 环境变量（需自行创建）
├── .example.env             # 环境变量模板
└── src/
    ├── config.py            # 配置加载（从 .env 读取）
    ├── model/
    │   └── inference.py     # ML 模型推理封装
    └── web/
        ├── app.py           # FastAPI 应用工厂，定义路由
        ├── llm_service.py   # LLM 三阶段分析服务
        └── templates/
            └── index.html   # Jinja2 前端页面
```

## 启动方式

### 安装依赖

使用 Conda 安装：

```bash
conda env create -f environment.yml
conda activate intro2ai
```

或直接使用 pip：

```bash
pip install -r requirements.txt
```

### 配置环境变量

复制 `.example.env` 为 `.env`，填入实际值：

```dotenv
API=https://models.sjtu.edu.cn/api/v1
API_SECRET=your-api-key
API_MODEL=deepseek-reasoner
```

具体配置见 `https://claw.sjtu.edu.cn/guide/sjtu-api/`。

### 启动应用

```bash
python main.py
```

默认监听 `0.0.0.0:8000`，浏览器访问 `http://localhost:8000`。

## 架构概览

应用采用 **FastAPI + Jinja2** 的 SSR（服务端渲染）架构：浏览器请求由 FastAPI 路由处理，后端调用 ML 模型和 LLM API，再将结果渲染为 HTML 返回。

用户提交待检测语句后，后端依次完成四个步骤。

### ML 模型分类

调用 `classify_statement()`，由机器学习模型对语句进行统计分类，输出 `is_rumor`（`0` 或 `1`）和 `confidence`（`0` 到 `1`）。这一步仅依赖模型从训练数据中学习的模式。

### LLM 独立评判（Stage 1）

在未向 LLM 展示 ML 结果的情况下，让 LLM 独立分析语句。评估维度包括：

- 可验证性
- 信源可信度
- 逻辑连贯性
- 情绪化语言
- 信息具体性

该阶段输出 LLM 判断 `is_rumor`、文本标签 `label`、判断依据 `reasoning` 和支撑线索 `supporting_indicators`。

### 结果比较（Stage 2）

将 ML 与 LLM 的独立判断一同提交给 LLM，比较两者是否一致。该阶段输出 `agreement` 和 `comparison_summary`。

### 根因分析（Stage 3）

- 当 ML 与 LLM 一致时，LLM 综合统计信号和语言学证据解释判断。
- 当 ML 与 LLM 不一致时，LLM 分析文本中可能误导统计模型的特征、双方遗漏的信息，以及哪一种判断更可能正确。

最终页面以三列展示 ML 判断、LLM 判断和置信度；结果不一致时显示警告。下方展示 LLM 判据、支撑线索、比较摘要、根因分析和关键指标。

## 请求处理流程

核心处理流程位于 `POST /analyze`：

```text
用户输入语句
    │
    ▼
Step 1: classify_statement(statement)          ← ML 模型
    返回 {"is_rumor": 0/1, "confidence": 0.0~1.0}
    │
    ▼
Step 2: judge_statement(statement)             ← LLM Stage 1
    返回 {"is_rumor", "label", "reasoning", "supporting_indicators"}
    │
    ▼
Step 3: compare_results(statement, ml, llm)    ← LLM Stage 2
    返回 {"agreement": true/false, "comparison_summary"}
    │
    ▼
Step 4: analyze_root_cause(statement, ml, llm, agreement)  ← LLM Stage 3
    返回 {"root_cause_analysis", "key_indicators"}
    │
    ▼
渲染 index.html → 返回给用户
```

## ML 模型接口

模型接口供 Web 模块调用，位于 `src/model/inference.py`。

### 函数签名

```python
def classify_statement(statement: str) -> dict[str, int | float]:
```

### 输入与输出

输入 `statement` 是待检测文本。返回字典包含：

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `is_rumor` | `int` | `0` = 非谣言，`1` = 谣言 |
| `confidence` | `float` | 可选置信度，范围为 `0.0` 到 `1.0` |

示例：

```python
{
    "is_rumor": 1,
    "confidence": 0.87,
}
```

若不使用 `confidence`，前端可以忽略该字段。

### 接口测试

```python
from src.model import classify_statement

result = classify_statement("某测试语句")
print(result)
```
