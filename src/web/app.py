import os
from pathlib import Path

from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from src.config import API_MODEL
from src.model import classify_statement
from src.rag.service import retrieve_rag_evidence
from src.web.llm_service import analyze_root_cause, compare_results, judge_statement


def _env_enabled(name: str, default: bool = True) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


def _binary_verdict(value: object) -> int | None:
    return value if type(value) is int and value in (0, 1) else None


def _agreement(value: object) -> bool | None:
    return value if type(value) is bool else None


def create_runtime_app() -> FastAPI:
    """Build the app inside a Uvicorn reload worker."""
    from src.model import configure_inference

    configure_inference(
        config_path=os.getenv("RUMOR_CONFIG_PATH", "configs/bertweet.yaml"),
        checkpoint=os.getenv("RUMOR_CHECKPOINT_PATH") or None,
        device=os.getenv("RUMOR_DEVICE") or None,
        force_mock=_env_enabled("RUMOR_FORCE_MOCK", False),
    )
    return create_app(
        enable_rag=_env_enabled("RUMOR_ENABLE_RAG"),
        enable_llm=_env_enabled("RUMOR_ENABLE_LLM"),
    )


def create_app(enable_rag: bool = True, enable_llm: bool = True) -> FastAPI:
    app = FastAPI(title="Rumor Detection System")

    templates_dir = Path(__file__).parent / "templates"
    templates = Jinja2Templates(directory=str(templates_dir))

    @app.get("/", response_class=HTMLResponse)
    async def index(request: Request):
        return templates.TemplateResponse(request, "index.html", {
            "model_name": API_MODEL,
        })

    @app.post("/analyze", response_class=HTMLResponse)
    async def analyze(request: Request, statement: str = Form(...)):
        statement = statement.strip()
        if not statement:
            return templates.TemplateResponse(request, "index.html", {
                "model_name": API_MODEL,
                "error": "Please enter a statement to analyze.",
            })

        # Step 1: ML model classifies (returns raw {is_rumor, confidence})
        try:
            raw_result = classify_statement(statement)
        except Exception as exc:
            return templates.TemplateResponse(request, "index.html", {
                "model_name": API_MODEL,
                "statement": statement,
                "error": f"Classification failed: {exc}",
            })

        rag_evidence = []
        if enable_rag:
            try:
                rag_evidence = retrieve_rag_evidence(statement)
            except Exception:
                rag_evidence = []

        if not enable_llm:
            label = "RUMOR" if raw_result["is_rumor"] else "NOT RUMOR"
            return templates.TemplateResponse(request, "index.html", {
                "model_name": API_MODEL,
                "statement": statement,
                "ml_is_rumor": raw_result["is_rumor"],
                "ml_confidence": raw_result["confidence"],
                "llm_label": "DISABLED",
                "llm_is_rumor": None,
                "agreement": None,
                "llm_reasoning": f"LLM analysis is disabled. ML-only verdict: {label}.",
                "supporting_indicators": [],
                "comparison_summary": "LLM comparison is disabled.",
                "root_cause_analysis": "No LLM root-cause analysis was generated.",
                "key_indicators": [],
                "rag_evidence": rag_evidence,
            })

        # Step 2: Stage 1 — LLM independently judges the statement
        try:
            llm_result = judge_statement(statement, rag_evidence=rag_evidence)
        except Exception as exc:
            return templates.TemplateResponse(request, "index.html", {
                "model_name": API_MODEL,
                "statement": statement,
                "error": f"LLM analysis failed: {exc}",
            })

        llm_is_rumor = _binary_verdict(llm_result.get("is_rumor"))
        raw_llm_label = llm_result.get("label")
        if llm_is_rumor is None:
            llm_label = "UNAVAILABLE"
        elif isinstance(raw_llm_label, str) and raw_llm_label.strip():
            llm_label = raw_llm_label.strip()
        else:
            llm_label = "RUMOR" if llm_is_rumor else "NOT RUMOR"

        # Step 3: Stage 2 — LLM compares its verdict with ML result
        comparison = {}
        if llm_is_rumor is not None:
            try:
                comparison = compare_results(statement, raw_result, llm_result)
            except Exception:
                pass

        agreement = _agreement(comparison.get("agreement"))

        # Step 4: Stage 3 — Root cause analysis
        analysis = {}
        if agreement is not None:
            try:
                analysis = analyze_root_cause(
                    statement,
                    raw_result,
                    llm_result,
                    agreement,
                )
            except Exception:
                pass

        return templates.TemplateResponse(request, "index.html", {
            "model_name": API_MODEL,
            "statement": statement,
            "ml_is_rumor": raw_result["is_rumor"],
            "ml_confidence": raw_result["confidence"],
            "llm_label": llm_label,
            "llm_is_rumor": llm_is_rumor,
            "agreement": agreement,
            "llm_reasoning": llm_result.get("reasoning", ""),
            "supporting_indicators": llm_result.get("supporting_indicators", []),
            "comparison_summary": comparison.get(
                "comparison_summary",
                "LLM comparison is unavailable." if agreement is None else "",
            ),
            "root_cause_analysis": analysis.get("root_cause_analysis", ""),
            "key_indicators": analysis.get("key_indicators", []),
            "rag_evidence": rag_evidence,
        })

    return app
