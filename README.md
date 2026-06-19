# Rumor Detection System

> **NIS4307: Introduction to Artificial Intelligence** — Course Project by Group 4, Shanghai Jiao Tong University.

**Authors**:
* Zhongchun Chen [@ZhongchunChen](https://github.com/ZhongchunChen)
* Runze Shen [@RanceChen06](https://github.com/RanceChen06)
* Mingchen Dai [@MingchenDai](https://github.com/MingchenDai)
* Zihao Xie [@Zihao-Xie090](https://github.com/Zihao-Xie090)

**Docs**: For Chinese docs, go to [中文文档](docs/README_CN.md).

**Final Report**: The $\LaTeX$ source code (`report.tex`) and the compiled document (`report.pdf`) are located in the `/report` directory. Please ensure you are referencing the **latest** commit on the main branch for the most up-to-date version.

**Table of Contents**:
1. [Overview](#overview)
2. [Setup](#setup)
3. [Prepare the Model](#prepare-the-model)
4. [Optional LLM and RAG Configuration](#optional-llm-and-rag-configuration)
5. [Run the Web Demo](#run-the-web-demo)
6. [Evaluate a Custom Test Dataset](#evaluate-a-custom-test-dataset)
7. [Project Structure](#project-structure)
8. [Command Summary](#command-summary)

## Overview

The Rumor Detection System classifies English statements using a fine-tuned BERTweet model. Its web interface can optionally retrieve related evidence from ChromaDB and use an OpenAI-compatible LLM to explain the prediction, compare ML and LLM verdicts, and analyze agreement or divergence.

The components remain independent where practical. Training and evaluation require only the ML model, while LLM and RAG support can be enabled for the complete interactive demo. Detailed architecture, experiments, and technical analysis are provided in the [final report](report/report.pdf) and documents under `docs/`.

## Setup

Conda with Python 3.10 is the recommended environment:

```bash
conda env create -f environment.yml
conda activate intro2ai
```

Update an existing environment after dependency changes with:

```bash
conda env update -f environment.yml --prune
```

Alternatively, install the project in a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

On Windows PowerShell, activate it with `\.venv\Scripts\Activate.ps1`.

## Prepare the Model

Real inference and evaluation prefer a locally trained checkpoint at:

```text
checkpoints/bertweet/best_model/
```

If that directory is absent, the project downloads the fine-tuned checkpoint configured under `model.huggingface_checkpoint`. A successful local training run takes precedence automatically. To train, the datasets must contain `text` and `label` columns, where `0` means not rumor and `1` means rumor.

```csv
text,label
"This claim is confirmed by an official source.",0
"A celebrity secretly died yesterday.",1
```

Dataset paths are local-first. If a configured file is absent, the project downloads its mapped Parquet file from the Hugging Face dataset repository configured under `data.huggingface` in `configs/bertweet.yaml` and reuses the Hugging Face cache.

Train with the defaults in `configs/bertweet.yaml`:

```bash
python main.py train
```

The pretrained `vinai/bertweet-base` model is cached under `checkpoints/base/`. The best checkpoint is selected by validation macro-F1 and saved under `checkpoints/bertweet/best_model/`. Metrics, training history, and plots are written to `outputs/bertweet/`.

## Optional LLM and RAG Configuration

This section is required only for the complete explanatory demo. ML-only display and evaluation do not require an API key or RAG database.

### Configure the LLM

Copy the environment template:

```bash
cp .example.env .env
```

On Windows PowerShell, use `Copy-Item .example.env .env`.

Set the OpenAI-compatible API values in `.env`:

```text
API=https://models.sjtu.edu.cn/api/v1
API_SECRET=your-api-key
API_MODEL=deepseek-reasoner
```

### Enable RAG

Download `ChromaDB_data_populate.zip` from the project Releases page and extract it so the database is located at:

```text
datasets/ChromaDB_data_populate/DataBase/data
```

RAG is optional. If the database is unavailable, the web application can continue without retrieved evidence.

## Run the Web Demo

Before starting, ensure the environment is active and a real checkpoint is available if you intend to inspect meaningful ML predictions.

Run the complete ML, LLM, and available RAG workflow:

```bash
python main.py serve
```

Open `http://localhost:8000` in a browser.

Run the BERTweet-only display without LLM or RAG configuration:

```bash
python main.py serve --no-llm --no-rag
```

> **Checkpoint behavior:** The locally trained `checkpoints/bertweet/best_model/` is preferred. If it is missing, the configured Hugging Face checkpoint is downloaded and cached. The web interface uses its mock classifier only when neither checkpoint can be resolved.

## Evaluate a Custom Test Dataset

This workflow supports evaluation on self-created data for the course grading policy. It uses only the trained BERTweet checkpoint and does not call the LLM API or RAG module.

Create a CSV containing at least `text` and `label` columns:

```csv
text,label
"This report cites a verifiable government announcement.",0
"Scientists secretly confirmed an impossible cure overnight.",1
```

Additional columns are allowed. Rows with missing text, missing labels, or labels outside `0` and `1` are excluded.

Evaluate the configured best checkpoint:

```bash
python main.py evaluate --test datasets/my_test.csv
```

If the local path does not exist, `--test` also uses the configured Hugging Face mapping. For example, `--test datasets/val.csv` retrieves the remote validation Parquet file; an existing local file always takes precedence. Add custom remote test files to `data.huggingface.files` before referencing them with `--test`.

Or specify a checkpoint and output file:

```bash
python main.py evaluate \
  --test datasets/my_test.csv \
  --checkpoint checkpoints/bertweet/best_model \
  --output outputs/bertweet/my_test_metrics.json
```

The command prints macro-F1, accuracy, and the confusion matrix. Without `--output`, custom-test results are saved to `outputs/bertweet/test_metrics.json`.

## Project Structure

```text
configs/        Model and training configuration
datasets/       Local training, test, and RAG data
docs/           Module documentation and architecture assets
report/         Technical report source and compiled document
src/model/      Data processing, training, evaluation, and inference
src/rag/        Optional ChromaDB retrieval and RAG utilities
src/web/        FastAPI application and web template
src/cli/        Command implementations and argument validation
tests/          CLI and configuration regression tests
main.py         Unified command-line entry point
```

## Command Summary

| Command | Purpose |
| --- | --- |
| `python main.py serve` | Start the web demo |
| `python main.py train` | Train the BERTweet model |
| `python main.py evaluate --test FILE.csv` | Evaluate a labeled test dataset |
| `python main.py predict "TEXT"` | Classify one statement |
| `python main.py plot-history` | Regenerate training plots |
| `python main.py rag query "TEXT"` | Run standalone RAG analysis |

For complete options:

```bash
python main.py --help
python main.py COMMAND --help
```

## Technical Documentation

- [Machine learning model](docs/model.md): BERTweet architecture, data preparation, training, evaluation, prediction, and generated artifacts
- [Frontend](docs/frontend.md): FastAPI/Jinja architecture, analysis workflow, environment setup, and ML interface contract
- [Technical report](report/): detailed model design, experiments, results, and analysis
- [Chinese README](docs/README_CN.md): Chinese project setup and usage guide
