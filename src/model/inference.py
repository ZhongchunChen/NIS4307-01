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

from src.config import load_config, resolve_checkpoint_path
from src.model.utils import get_device

_logger = logging.getLogger(__name__)

_tokenizer: AutoTokenizer | None = None
_model: AutoModelForSequenceClassification | None = None
_device: torch.device | None = None
_max_length: int = 128
_config_path: str | Path = "configs/bertweet.yaml"
_checkpoint_path: str | Path | None = None
_device_name: str | None = None
_force_mock: bool = False


def configure_inference(
    config_path: str | Path | None = None,
    checkpoint: str | Path | None = None,
    device: str | None = None,
    force_mock: bool = False,
) -> None:
    """Configure lazy-loaded inference for the web app or CLI callers."""
    global _tokenizer, _model, _device, _max_length
    global _config_path, _checkpoint_path, _device_name, _force_mock

    if config_path is not None:
        _config_path = config_path
    _checkpoint_path = checkpoint
    _device_name = device
    _force_mock = force_mock

    _tokenizer = None
    _model = None
    _device = None
    _max_length = 128


def _ensure_model_loaded() -> None:
    global _tokenizer, _model, _device, _max_length

    if _model is not None:
        return

    config = load_config(_config_path)
    training = config["training"]
    checkpoint = resolve_checkpoint_path(config, _checkpoint_path)

    _device = get_device(_device_name or training.get("device", "auto"))
    _max_length = config["model"]["max_length"]
    _tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    _model = AutoModelForSequenceClassification.from_pretrained(checkpoint).to(_device)
    _model.eval()
    _logger.info("Model loaded from %s on %s", checkpoint, _device)


def classify_statement(statement: str) -> dict[str, int | float]:
    """Classify a statement as rumor (1) or not rumor (0).

    Returns:
        {"is_rumor": int, "confidence": float}
    """
    if _force_mock:
        return _mock_classify(statement)

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
