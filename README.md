# Rumor Detection System

Course Project of NIS4307 *Introduction to Artificial Intelligence*, Shanghai Jiao Tong University.

[中文文档](docs/README_CN.md)

## Overview

This project combines a **fine-tuned BERTweet model**, **multi-stage LLM analysis**, and an optional **RAG evidence retrieval module** to build an explainable rumor detection system. Given an English statement, the system classifies it with a machine learning model, retrieves related evidence from a ChromaDB knowledge base when available, asks an LLM to analyze the statement, compares the ML and LLM verdicts, analyzes agreement or divergence, and presents the full analysis in a visual interface.

## Project Structure

```text
NIS4307-01/
├── RAG/
│   └── requirements.txt          # Wrapper for unified project dependencies
├── configs/
│   └── bertweet.yaml          # training and model configuration
├── datasets/
│   ├── train.csv              # training set
│   ├── val.csv                # validation set
│   ├── gossipcop_extra.csv     # converted GossipCop training CSV
│   ├── shared_task_extra.csv   # converted shared-task training CSV
│   └── raw_data/               # original public dataset files
├── checkpoints/
│   ├── base/                  # cached pretrained BERTweet checkpoint
│   └── bertweet/              # fine-tuned model checkpoints
├── outputs/
│   └── bertweet/              # metrics, logs, and training curves
│   └── bertweet.yaml             # Model and training configuration
├── docs/
│   ├── frontend.md               # Frontend architecture documentation
│   ├── model.md                  # Model training documentation
│   └── README_CN.md              # Chinese README
├── outputs/bertweet/             # Training metrics, plots, and evaluation outputs
├── src/
│   ├── config.py                 # Configuration loader (.env + YAML)
│   ├── model.py                  # BERTweet model wrapper
│   ├── api_service.py            # LLM multi-stage analysis service
│   ├── rag/
│   │   ├── retriever.py          # ChromaDB evidence retriever
│   │   ├── service.py            # Main-system RAG evidence wrapper
│   │   ├── llm_judge.py          # Standalone RAG + LLM judge
│   │   ├── cli.py                # Standalone RAG CLI
│   │   └── config.py             # RAG configuration
│   ├── training/
│   │   ├── data.py               # Dataset loading and cleaning
│   │   ├── metrics.py            # Classification metrics
│   │   ├── pipeline.py           # Training, evaluation, and plotting logic
│   │   └── utils.py              # Training utilities
│   └── web/
│       ├── app.py                # FastAPI application factory
│       └── templates/
│           └── index.html        # Jinja2 frontend page
│   └── cli/                       # CLI wrappers for train/evaluate/predict/plot
├── .example.env                  # Environment variable template
├── .gitignore                    # Git ignore rules
├── README.md                     # Project README
├── environment.yml               # Conda environment configuration
├── main.py                       # Unified entry point
├── pyproject.toml                # Project package configuration
├── requirements.txt              # Python dependencies
└── report.pdf                    # Final course report
```

## Architecture

### Overall Pipeline

```
User submits a statement
    │
    ▼
┌──────────────────────────────────────────────────────┐
│  Step 1   classify_statement()     ← BERTweet model  │
│           Returns {is_rumor, confidence}             │
├──────────────────────────────────────────────────────┤
│  Step 2   retrieve_rag_evidence()  ← Optional RAG    │
│           Retrieves top-k evidence from ChromaDB     │
├──────────────────────────────────────────────────────┤
│  Step 3   judge_statement()        ← LLM Stage 1     │
│           LLM judges with optional RAG evidence      │
│           Returns {is_rumor, label, reasoning,       │
│                    supporting_indicators}            │
├──────────────────────────────────────────────────────┤
│  Step 4   compare_results()        ← LLM Stage 2     │
│           LLM compares ML and own verdicts           │
│           Returns {agreement, comparison_summary}    │
├──────────────────────────────────────────────────────┤
│  Step 5   analyze_root_cause()     ← LLM Stage 3     │
│           Convergent → unified explanation           │
│           Divergent → root cause analysis            │
│           Returns {root_cause_analysis,              │
│                    key_indicators}                   │
└──────────────────────────────────────────────────────┘
    │
    ▼
Render HTML → returned to user
```

### ML Model

A binary classifier fine-tuned from `vinai/bertweet-base`:

```
raw text → BERTweet tokenizer → BERTweet encoder → dropout → linear head → [rumor / not rumor]
```

Outputs softmax probabilities for two classes. Confidence is the probability of the predicted class.

### RAG Evidence Retrieval

The optional RAG module retrieves top-k related evidence from a local ChromaDB knowledge base and passes the retrieved evidence to the LLM prompt. RAG does **not** modify the BERTweet model's binary prediction. It is used only as additional context for explanation and comparison.

Runtime behavior:

- If the RAG database and dependencies are available, the main app calls `retrieve_rag_evidence()` before LLM Stage 1.
- Retrieved evidence is shown in the frontend under **RAG Retrieved Evidence**.
- If RAG is unavailable, the app falls back to the original BERTweet + LLM pipeline and continues to run.

### LLM Analysis (Three Stages)

| Stage | Function | Purpose |
|-------|----------|---------|
| Stage 1 | `judge_statement()` | LLM judges the text across five dimensions: verifiability, source credibility, logical coherence, emotional language, and specificity, with optional RAG evidence as reference |
| Stage 2 | `compare_results()` | LLM compares the ML verdict with its own, determining agreement or divergence |
| Stage 3 | `analyze_root_cause()` | If convergent: synthesizes a unified explanation. If divergent: analyzes what misled which system |

### Display

- Three-column verdict header: ML Verdict | LLM Verdict (with yellow ⚠ on divergence) | Confidence
- Two-column detail: LLM Analysis (reasoning + indicators) | Comparison (summary + root cause + key indicators)
- RAG Retrieved Evidence: top retrieved evidence items with source, label, distance, and text when RAG is available

## Training

### Data Preparation

Place `train.csv` and `val.csv` under `datasets/` with `text` and `label` columns:

```csv
text,label
"This appears to be a rumor",1
"This is verified news",0
```

### Configuration

Edit `configs/bertweet.yaml` to adjust hyperparameters and paths. Key options:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `data.train_path` | `datasets/train.csv` | Training set path |
| `data.val_path` | `datasets/val.csv` | Validation set path |
| `training.epochs` | 16 | Number of epochs |
| `training.learning_rate` | 1e-5 | Learning rate |
| `training.early_stopping_metric` | `macro_f1` | Early stopping metric |
| `training.early_stopping_patience` | 2 | Early stopping patience |

### Run Training

```bash
python main.py --train
```

The first run downloads `vinai/bertweet-base` from Hugging Face and caches it at `checkpoints/base/`. The best model is saved to `checkpoints/bertweet/best_model/`, with plots and metrics in `outputs/bertweet/`.

### Evaluate

```bash
python -m src.evaluate --config configs/bertweet.yaml
```

### Predict via CLI

```bash
python -m src.predict --config configs/bertweet.yaml --text "Example text"
```

## Setup

### Install Dependencies

Conda is the recommended setup path. It creates a Python 3.10 environment and installs this repository in editable mode from `pyproject.toml`, including the frontend, BERTweet model, LLM API client, and RAG dependencies.

```bash
conda env create -f environment.yml
conda activate intro2ai
```

If the environment already exists, update it after dependency changes:

```bash
conda env update -f environment.yml --prune
conda activate intro2ai
```

Pip-only setup is also supported for users who do not use Conda:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

`pyproject.toml` is the canonical dependency list. `requirements.txt` and `RAG/requirements.txt` are compatibility wrappers, so do not maintain separate frontend/model/RAG environments.

### Configure LLM API

Copy `.example.env` to `.env` and fill in your credentials:

```
API=https://models.sjtu.edu.cn/api/v1
API_SECRET=your-api-key
API_MODEL=deepseek-reasoner
```

See https://claw.sjtu.edu.cn/guide/sjtu-api/ for setup instructions.

### Optional: Enable RAG Evidence Retrieval

RAG Python code is packaged under `src/rag/`. The ChromaDB database is large and is not stored directly in the Git repository. To enable RAG in the frontend:

1. Open the GitHub **Releases** page.
2. Download `ChromaDB_data_populate.zip` from the release named **ChromaDB database for RAG**.
3. Extract the zip file into the `datasets/` directory.

After extraction, the path should look like:

```text
datasets/ChromaDB_data_populate/DataBase/data
```

When RAG is correctly configured, the frontend will display a **RAG Retrieved Evidence** section after analysis. If the database or dependencies are missing, the main app still runs without RAG evidence.

You can also run the standalone RAG demo:

```bash
python -m src.rag.cli
```

### Launch the Frontend

```bash
python main.py
```

Listens on `0.0.0.0:8000`. Open `http://localhost:8000` in a browser.

### CLI Options

```
python main.py              Start the frontend (default)
python main.py --display    Start the frontend
python main.py --train      Train the model
python main.py --help       Show help
```
