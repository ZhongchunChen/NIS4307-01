"""
Rumor detection model.

Wraps the BERTweet binary classifier for use by the FastAPI frontend.
On first call, lazy-loads the trained model checkpoint. If no checkpoint
is found, falls back to a mock classifier for frontend testing.
"""

import html
import logging
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.config import load_config
from src.utils import get_device

_logger = logging.getLogger(__name__)

_tokenizer: AutoTokenizer | None = None
_model: AutoModelForSequenceClassification | None = None
_device: torch.device | None = None
_max_length: int = 128


def _ensure_model_loaded() -> None:
    global _tokenizer, _model, _device, _max_length

    if _model is not None:
        return

    config = load_config("configs/bertweet.yaml")
    training = config["training"]
    checkpoint = Path(training["checkpoint_dir"]) / "best_model"

    if not checkpoint.exists():
        raise FileNotFoundError(
            f"Model checkpoint not found at {checkpoint}. "
            "Train the model first or provide the checkpoint directory."
        )

    _device = get_device(training.get("device", "auto"))
    _max_length = config["model"]["max_length"]
    _tokenizer = AutoTokenizer.from_pretrained(str(checkpoint))
    _model = AutoModelForSequenceClassification.from_pretrained(str(checkpoint)).to(_device)
    _model.eval()
    _logger.info("Model loaded from %s on %s", checkpoint, _device)


def classify_statement(statement: str) -> dict[str, int | float]:
    """Classify a statement as rumor (1) or not rumor (0).

    Returns:
        {"is_rumor": int, "confidence": float}
    """
    try:
        _ensure_model_loaded()
    except FileNotFoundError:
        _logger.warning("Checkpoint not found, using mock classifier.")
        return _mock_classify(statement)

    cleaned = " ".join(html.unescape(statement).split())
    encoded = _tokenizer(
        cleaned,
        max_length=_max_length,
        truncation=True,
        return_tensors="pt",
    )
    encoded = {key: value.to(_device) for key, value in encoded.items()}

    with torch.no_grad():
        probabilities = _model(**encoded).logits.softmax(dim=-1).squeeze(0).cpu()

    is_rumor = int(probabilities.argmax())
    confidence = float(probabilities[is_rumor])

    return {"is_rumor": is_rumor, "confidence": round(confidence, 4)}


def _mock_classify(statement: str) -> dict[str, int | float]:
    import random
    import time

    time.sleep(0.5 + random.random() * 1.0)
    return {
        "is_rumor": random.randint(0, 1),
        "confidence": round(random.uniform(0.55, 0.95), 2),
    }
