"""
API service for enriching raw model output with text explanations.

Takes the ML model's raw classification ({is_rumor, confidence}) and calls
the LLM API to generate human-readable labels, reasoning, key indicators,
and a detailed explanation.
"""

import json
import re

from openai import OpenAI

from .config import API_BASE_URL, API_MODEL, API_SECRET

client = OpenAI(api_key=API_SECRET, base_url=API_BASE_URL)

ENRICH_PROMPT = """You are a rumor detection analyst. Given a statement and a machine learning model's raw classification result, produce a structured analysis.

The ML model classified the statement as:
- is_rumor: {is_rumor} (0 = not rumor, 1 = rumor)
- confidence: {confidence} (0.0 to 1.0)

Analyze the statement and respond with ONLY a JSON object (no markdown, no extra text):
{{
  "label": "RUMOR" or "NOT RUMOR",
  "reasoning": "step-by-step analysis of the statement considering verifiability, source credibility, logical coherence, emotional language, and specificity",
  "key_indicators": ["specific clue 1", "specific clue 2", ...]
}}"""

EXPLAIN_PROMPT = """You are an AI explanation system. Given a rumor classification result, provide a clear, educational explanation that a non-expert can understand.

Explain:
1. Why the model reached this conclusion (in simple terms)
2. What specific clues in the text support the classification
3. How a reader can verify this claim themselves (fact-checking tips)

Keep the explanation concise and actionable."""


def enrich_classification(
    statement: str, raw_result: dict[str, int | float]
) -> dict[str, object]:
    """Call the LLM to generate label, reasoning, and key indicators.

    Args:
        statement: The original statement being analyzed.
        raw_result: Dict with is_rumor (int 0/1) and confidence (float).

    Returns:
        Dict with keys: label, reasoning, key_indicators.
    """
    prompt = ENRICH_PROMPT.format(
        is_rumor=raw_result["is_rumor"],
        confidence=raw_result["confidence"],
    )

    response = client.chat.completions.create(
        model=API_MODEL,
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": statement},
        ],
        temperature=0.3,
    )

    message = response.choices[0].message
    raw = message.content or ""
    reasoning_content = getattr(message, "reasoning_content", None)

    result = _parse_json(raw)
    if reasoning_content:
        result["reasoning_content"] = reasoning_content

    return result


def explain_result(statement: str, enriched: dict[str, object]) -> str:
    """Call the LLM to generate a user-friendly explanation.

    Args:
        statement: The original statement.
        enriched: Dict with at least label, confidence, reasoning, key_indicators.

    Returns:
        Explanation string (markdown).
    """
    response = client.chat.completions.create(
        model=API_MODEL,
        messages=[
            {"role": "system", "content": EXPLAIN_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Statement: {statement}\n\n"
                    f"Classification: {enriched.get('label')}\n"
                    f"Confidence: {enriched.get('confidence')}\n"
                    f"Reasoning: {enriched.get('reasoning')}\n"
                    f"Key indicators: {enriched.get('key_indicators')}\n\n"
                    "Please explain this result."
                ),
            },
        ],
        temperature=0.3,
    )

    return response.choices[0].message.content or ""


def _parse_json(raw: str) -> dict[str, object]:
    """Parse JSON from model output, with fallback for markdown-wrapped JSON."""
    raw = raw.strip()

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    match = re.search(r"```(?:json)?\s*([\s\S]*?)```", raw)
    if match:
        try:
            return json.loads(match.group(1).strip())
        except json.JSONDecodeError:
            pass

    match = re.search(r"\{[\s\S]*\}", raw)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    return {}
