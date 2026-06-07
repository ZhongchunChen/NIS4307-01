# Rumor Detection System

Course Project of NIS4307 *Introduction to Artificial Intelligence*, Shanghai Jiao Tong University.

[中文文档](docs/README_CN.md)

## Overview

This project combines a **fine-tuned BERTweet model** with **multi-stage LLM analysis** to build an explainable rumor detection system. Given an English statement, the system independently classifies it with both a machine learning model and an LLM, compares the two verdicts, analyzes agreement or divergence, and presents the full analysis in a visual interface.

## Project Structure

```
NIS4307-01/
├── main.py                       # Unified entry point (train / serve)
├── requirements.txt              # Python dependencies
├── environment.yml               # Conda environment config
├── .env                          # Environment variables (create locally)
├── .example.env                  # Environment variable template
├── configs/
│   └── bertweet.yaml             # Model and training configuration
├── checkpoints/
│   ├── base/                     # Cached pretrained BERTweet weights
│   └── bertweet/best_model/      # Fine-tuned best model
├── outputs/bertweet/             # Training metrics, plots, and reports
├── datasets/
│   ├── train.csv                 # Training set
│   └── val.csv                   # Validation set
├── docs/
│   ├── frontend.md               # Frontend architecture
│   ├── model.md                  # Model training guide
│   └── README_CN.md              # Chinese README
└── src/
    ├── app.py                    # FastAPI application factory
    ├── config.py                 # Configuration loader (.env + YAML)
    ├── model.py                  # Model wrapper (lazy-loads BERTweet)
    ├── api_service.py            # LLM three-stage analysis service
    ├── data.py                   # Dataset loading and cleaning
    ├── train.py                  # Training script
    ├── evaluate.py               # Evaluation script
    ├── predict.py                # Single-text prediction CLI
    ├── metrics.py                # Classification metrics
    ├── plot_history.py           # Training curve plotting
    ├── utils.py                  # Shared utilities
    └── templates/
        └── index.html            # Jinja2 frontend page
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
│  Step 2   judge_statement()        ← LLM Stage 1     │
│           LLM judges independently (no ML result)    │
│           Returns {is_rumor, label, reasoning,       │
│                    supporting_indicators}            │
├──────────────────────────────────────────────────────┤
│  Step 3   compare_results()        ← LLM Stage 2     │
│           LLM compares ML and own verdicts           │
│           Returns {agreement, comparison_summary}    │
├──────────────────────────────────────────────────────┤
│  Step 4   analyze_root_cause()     ← LLM Stage 3     │
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

### LLM Analysis (Three Stages)

| Stage | Function | Purpose |
|-------|----------|---------|
| Stage 1 | `judge_statement()` | LLM independently judges the text across five dimensions: verifiability, source credibility, logical coherence, emotional language, and specificity |
| Stage 2 | `compare_results()` | LLM compares the ML verdict with its own, determining agreement or divergence |
| Stage 3 | `analyze_root_cause()` | If convergent: synthesizes a unified explanation. If divergent: analyzes what misled which system |

### Display

- Three-column verdict header: ML Verdict | LLM Verdict (with yellow ⚠ on divergence) | Confidence
- Two-column detail: LLM Analysis (reasoning + indicators) | Comparison (summary + root cause + key indicators)

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

Conda (recommended):

```bash
conda env create -f environment.yml
conda activate intro2ai
```

Or pip:

```bash
pip install -r requirements.txt
```

### Configure LLM API

Copy `.example.env` to `.env` and fill in your credentials:

```
API=https://models.sjtu.edu.cn/api/v1
API_SECRET=your-api-key
API_MODEL=deepseek-reasoner
```

See https://claw.sjtu.edu.cn/guide/sjtu-api/ for setup instructions.

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
