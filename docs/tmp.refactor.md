**1. Environment Unification**

Use one environment definition as the source of truth. Right now you have:

- Root `requirements.txt`: clean, modern app/model deps.
- `pyproject.toml`: missing FastAPI/OpenAI/RAG deps.
- `environment.yml`: wraps root `requirements.txt`.
- `RAG/requirements.txt`: UTF-16 encoded, heavily pinned, and conflicts with root versions.

Best direction:

```text
pyproject.toml          # primary dependency source
requirements.txt        # optional export for pip users
environment.yml         # conda wrapper that installs the project
```

Use optional dependency groups:

```toml
[project.optional-dependencies]
web = ["fastapi", "uvicorn", "jinja2", "openai", "python-dotenv"]
rag = ["chromadb", "sentence-transformers"]
dev = ["pytest", "ruff"]
```

Then users can install:

```bash
pip install -e ".[web,rag]"
```

or with conda:

```bash
conda env create -f environment.yml
conda activate intro2ai
pip install -e ".[web,rag]"
```

I would remove `RAG/requirements.txt` after merging its needed packages into `pyproject.toml`.

**2. Documentation Hierarchy**

Make root `README.md` the user guide only. Suggested root README sections:

```text
# Rumor Detection System
## Overview
## Project Structure
## Installation
  - Windows
  - macOS
  - Linux
  - Conda
  - pip/venv
## Configuration
## Train the ML Model
## Evaluate the Model
## Run Prediction
## Run the Demo Web App
## main.py Arguments
## Optional RAG Setup
## Troubleshooting
```

Move module explanation into:

```text
docs/model.md
docs/frontend.md
docs/rag.md
docs/configuration.md      # optional but useful
docs/development.md        # optional contributor/dev notes
docs/README_CN.md          # Chinese user guide if needed
```

The current `docs/model.md` is too long and bilingual; split English/Chinese or keep one language per file. Add `docs/rag.md`; RAG is currently undocumented except inside the root README.

**3. Module Organization**

Do not keep a flat `src/` forever, and do not keep `RAG/` as a separate pseudo-project. Use one package with discrete modules:

```text
src/rumor_detection/
  __init__.py
  cli.py
  config.py

  ml/
    data.py
    model.py
    train.py
    evaluate.py
    predict.py
    metrics.py
    plot_history.py

  web/
    app.py
    templates/index.html

  llm/
    service.py
    prompts.py

  rag/
    retriever.py
    service.py
    config.py
```

This gives you one environment, one import system, and clean boundaries. The modules remain discrete, but integration is easier and you avoid the current `sys.path.insert()` in `src/rag_service.py`.

**Additional Improvement Suggestions**

- Replace hard-coded RAG config/API keys in `RAG/config.py` with `.env` variables.
- Standardize env names: use one scheme such as `LLM_API_BASE`, `LLM_API_KEY`, `LLM_MODEL`, `RAG_DB_PATH`, `RAG_TOP_K`.
- Expand `main.py` into subcommands: `train`, `evaluate`, `predict`, `serve`, maybe `rag-query`.
- Add `--config`, `--host`, `--port`, `--checkpoint`, and `--no-rag` options.
- Add a small pytest suite for config loading, data cleaning, metrics, JSON parsing, and RAG-disabled fallback.
- Remove generated files from version control unless needed: `outputs/`, `checkpoints/`, LaTeX artifacts in `report/`.
- Add `ruff` for formatting/linting.
- Rename the Python package away from `src`; importing from `src.model` works, but `rumor_detection.ml.model` is much clearer.

Best next step: first unify dependencies and package layout, then rewrite the README/docs around the new structure.