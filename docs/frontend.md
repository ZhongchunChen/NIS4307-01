# 前端架构说明

## 项目结构

```
NIS4307-01/
├── main.py                  # 入口：创建应用并启动 uvicorn
├── requirements.txt         # Python 依赖
├── .env                     # 环境变量（需自行创建）
├── .example.env             # 环境变量模板
└── src/
    ├── app.py               # FastAPI 应用工厂，定义路由
    ├── config.py            # 配置加载（从 .env 读取）
    ├── model.py             # ★ ML 模型（待协作者实现）
    ├── api_service.py       # LLM 三阶段分析服务
    └── templates/
        └── index.html       # Jinja2 前端页面
```

## 架构概览

当用户在浏览器提交待检测语句后，后端分四个步骤依次处理：

**Step 1 — ML 模型分类**

调用 `classify_statement()`，由机器学习模型对语句进行统计层面的分类，输出两个数值：`is_rumor`（0 或 1）和 `confidence`（0~1 的置信度）。这一步仅依赖模型对训练数据的学习，不涉及语义理解。

**Step 2 — LLM 独立评判（Stage 1）**

调用 LLM API，让大语言模型在**未看到 ML 结果**的情况下，独立对语句进行分析。LLM 从以下维度评估：可验证性（能否查证）、信源可信度（是否引用专家或机构）、逻辑连贯性（是否存在谬误）、情绪化语言（是否使用恐惧/愤怒/煽情措辞）、具体性（是否给出具体数字、日期或名称）。输出 LLM 自身的判断（`is_rumor`、`label`）、判断依据（`reasoning`）以及文本中的支撑线索（`supporting_indicators`）。

**Step 3 — 结果比较（Stage 2）**

将 ML 模型的结果与 LLM 的独立评判一同提交给 LLM，要求 LLM 比较两者是否一致。输出 `agreement`（true = 一致 / false = 分歧）和比较摘要（`comparison_summary`）。

**Step 4 — 根因分析（Stage 3）**

根据比较结果分支处理：
- **若 ML 与 LLM 一致**：LLM 综合统计信号和语言学证据，统一解释该语句为什么可能为真或为假；
- **若 ML 与 LLM 不一致**：LLM 分析分歧原因 —— 文本中哪些特征可能误导了统计模型？LLM 捕捉到了什么模型遗漏的信息（或反之）？哪种判断更可能正确？

最后将以上所有结果渲染到 HTML 页面返回给用户。前端以三列并排展示 ML 判语、LLM 判语和置信度，若两者不一致则在 LLM 判语旁显示黄色警告标识；下方以双列布局展示 LLM 分析（判据 + 支撑线索）和比较结果（比较摘要 + 根因分析 + 关键指标）。

应用采用 **FastAPI + Jinja2** 的 SSR（服务端渲染）架构。用户通过浏览器访问 → FastAPI 路由处理 → 调用 ML 模型和 LLM API → 渲染 HTML 返回。

核心处理流程（`POST /analyze`）：

```
用户输入语句
    │
    ▼
Step 1: classify_statement(statement)          ← ML 模型
    返回 {"is_rumor": 0/1, "confidence": 0.0~1.0} # Confidence可选
    │
    ▼
Step 2: judge_statement(statement)             ← LLM Stage 1（独立判断）
    返回 {"is_rumor", "label", "reasoning", "supporting_indicators"}
    │
    ▼
Step 3: compare_results(statement, ml, llm)    ← LLM Stage 2（比较）
    返回 {"agreement": true/false, "comparison_summary"}
    │
    ▼
Step 4: analyze_root_cause(statement, ml, llm, agreement)  ← LLM Stage 3（根因分析）
    返回 {"root_cause_analysis", "key_indicators"}
    │
    ▼
渲染 index.html → 返回给用户
```

## 启动方式

### 1. 安装依赖

使用 Conda 安装：

```bash
conda env create -f environment.yml
conda activate intro2ai
```

或直接使用 pip：

```bash
pip install -r requirements.txt
```

### 2. 配置环境变量

复制 `.example.env` 为 `.env`，填入实际值：

```
API=https://models.sjtu.edu.cn/api/v1    # LLM API 地址
API_SECRET=your-api-key                   # API 密钥
API_MODEL=deepseek-reasoner               # 模型名称
```

具体配置见`https://claw.sjtu.edu.cn/guide/sjtu-api/`。

### 3. 启动

```bash
python main.py
```

默认监听 `0.0.0.0:8000`，浏览器访问 `http://localhost:8000`。

## ML 模型接口（供模型组实现）

**文件**：`src/model.py`

**函数签名**：

```python
def classify_statement(statement: str) -> dict[str, int | float]:
```

**输入**：`statement` — 待检测的文本语句（字符串）

**输出**：一个字典，必须包含以下两个字段：

| 字段 | 类型 | 说明 |
|------|------|------|
| `is_rumor` | `int` | `0` = 非谣言，`1` = 谣言 |
| `confidence` | `float` | 置信度，范围 `0.0` ~ `1.0` |

`confidence`可选；不用的话后续前端直接弃用该字段。

**示例返回值**：

```python
{
    "is_rumor": 1,
    "confidence": 0.87
}
```

**测试方式**：

```python
from src.model import classify_statement
result = classify_statement("某测试语句")
print(result)  # {"is_rumor": 1, "confidence": 0.87}
```
