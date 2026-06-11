"""
API service for three-stage LLM analysis of rumor classification.

Stage 1: LLM independently judges the statement (no ML output seen).
Stage 2: LLM compares its verdict with the ML result.
Stage 3: LLM analyzes root cause — divergence or convergence.
"""

import json
import re

from openai import OpenAI

from .config import API_BASE_URL, API_MODEL, API_SECRET

client = OpenAI(api_key=API_SECRET, base_url=API_BASE_URL)

STAGE1_PROMPT = """You are a rumor detection analyst. Analyze the following statement independently and determine if it is likely a rumor.

Consider:
- Verifiability: can the claim be checked against authoritative sources?
- Source credibility: are identifiable experts or institutions cited?
- Logical coherence: does the argument hold together or rely on fallacies?
- Emotional language: does the text use fear, outrage, or sensationalism?
- Specificity: are concrete numbers, dates, or names provided?

Respond with ONLY a JSON object (no markdown, no extra text):
{{
  "is_rumor": 0 or 1,
  "label": "RUMOR" or "NOT RUMOR",
  "reasoning": "step-by-step explanation of why you reached this verdict, citing the textual evidence",
  "supporting_indicators": ["specific textual clue 1", "specific textual clue 2", ...]
}}"""

STAGE2_PROMPT = """A machine learning model independently classified the same statement:

ML model result:
- is_rumor: {ml_is_rumor} (0 = not rumor, 1 = rumor)
- confidence: {ml_confidence} (0.0 to 1.0)

Your independent analysis concluded:
- is_rumor: {llm_is_rumor}
- label: {llm_label}

Compare the two results. Do they agree or disagree on the final verdict?

Respond with ONLY a JSON object:
{{
  "agreement": true or false,
  "comparison_summary": "brief summary comparing the two verdicts and noting any notable differences in reasoning"
}}"""

STAGE3_DIVERGE_PROMPT = """The ML model and the LLM analysis reached DIFFERENT conclusions about this statement.

ML model: is_rumor = {ml_is_rumor}, confidence = {ml_confidence}
LLM analysis: is_rumor = {llm_is_rumor}, label = {llm_label}

LLM supporting indicators: {supporting_indicators}

Analyze WHY the divergence occurred:
- What features in the text might have misled the statistical model?
- What did the LLM catch that the model may have missed (or vice versa)?
- Which system is more likely correct in this case, and why?

Respond with ONLY a JSON object:
{{
  "root_cause_analysis": "detailed analysis of the divergence",
  "key_indicators": ["most important clue 1", "most important clue 2", ...]
}}"""

STAGE3_CONVERGE_PROMPT = """Both the ML model and the LLM analysis reached the SAME conclusion about this statement.

Result: is_rumor = {ml_is_rumor}, confidence = {ml_confidence}
LLM label: {llm_label}

LLM supporting indicators: {supporting_indicators}

Synthesize a unified explanation of WHY this statement is likely true or false, weaving together:
- The statistical confidence from the ML model
- The linguistic evidence from the LLM analysis

Respond with ONLY a JSON object:
{{
  "root_cause_analysis": "unified explanation combining both systems' evidence",
  "key_indicators": ["most important clue 1", "most important clue 2", ...]
}}"""

def format_rag_evidence(evidence: list[dict] | None) -> str:
    """Format RAG retrieved evidence for LLM prompt."""
    if not evidence:
        return "No RAG evidence was retrieved."

    lines = []
    for i, ev in enumerate(evidence, 1):
        source = ev.get("source", "unknown")
        label = ev.get("label", "unknown")
        distance = ev.get("distance", "unknown")
        text = ev.get("text", "")

        lines.append(
            f"Evidence {i}:\n"
            f"- Source: {source}\n"
            f"- Label: {label}\n"
            f"- Distance: {distance}\n"
            f"- Text: {text}"
        )

    return "\n\n".join(lines)


def judge_statement(
    statement: str,
    rag_evidence: list[dict] | None = None,
) -> dict[str, object]:
    """Stage 1: LLM independently judges the statement without seeing ML output.

    Returns:
        Dict with llm_is_rumor (int), llm_label (str),
        reasoning (str), supporting_indicators (list[str]).
    """

    rag_text = format_rag_evidence(rag_evidence)

    user_prompt = f"""
    Statement:
    {statement}

    Retrieved RAG Evidence:
    {rag_text}

    Please analyze the statement. The RAG evidence is only a reference.
    If the evidence is weak, irrelevant, or insufficient, explicitly mention that.
    Do not blindly trust retrieved evidence.
    """

    response = client.chat.completions.create(
        model=API_MODEL,
        messages=[
            {"role": "system", "content": STAGE1_PROMPT},
            {"role": "user", "content": user_prompt},
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


def compare_results(
    statement: str,
    ml_result: dict[str, int | float],
    llm_result: dict[str, object],
) -> dict[str, object]:
    """Stage 2: LLM compares its verdict with the ML result.

    Returns:
        Dict with agreement (bool), comparison_summary (str).
    """
    prompt = STAGE2_PROMPT.format(
        ml_is_rumor=ml_result["is_rumor"],
        ml_confidence=ml_result["confidence"],
        llm_is_rumor=llm_result.get("is_rumor", "unknown"),
        llm_label=llm_result.get("label", "unknown"),
    )

    response = client.chat.completions.create(
        model=API_MODEL,
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": statement},
        ],
        temperature=0.3,
    )

    raw = response.choices[0].message.content or ""
    return _parse_json(raw)


def analyze_root_cause(
    statement: str,
    ml_result: dict[str, int | float],
    llm_result: dict[str, object],
    agreement: bool,
) -> dict[str, object]:
    """Stage 3: Root cause analysis contextualized by agreement/disagreement.

    Returns:
        Dict with root_cause_analysis (str), key_indicators (list[str]).
    """
    indicators = llm_result.get("supporting_indicators", [])

    if agreement:
        prompt = STAGE3_CONVERGE_PROMPT.format(
            ml_is_rumor=ml_result["is_rumor"],
            ml_confidence=ml_result["confidence"],
            llm_label=llm_result.get("label", "unknown"),
            supporting_indicators=json.dumps(indicators, ensure_ascii=False),
        )
    else:
        prompt = STAGE3_DIVERGE_PROMPT.format(
            ml_is_rumor=ml_result["is_rumor"],
            ml_confidence=ml_result["confidence"],
            llm_is_rumor=llm_result.get("is_rumor", "unknown"),
            llm_label=llm_result.get("label", "unknown"),
            supporting_indicators=json.dumps(indicators, ensure_ascii=False),
        )

    response = client.chat.completions.create(
        model=API_MODEL,
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": statement},
        ],
        temperature=0.3,
    )

    raw = response.choices[0].message.content or ""
    return _parse_json(raw)


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
