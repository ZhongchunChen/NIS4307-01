from pathlib import Path

from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from src.config import API_MODEL
from src.model import classify_statement
from src.api_service import analyze_root_cause, compare_results, judge_statement
from src.rag_service import retrieve_rag_evidence

def create_app() -> FastAPI:
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
        
        # Optional Step: RAG retrieves related evidence
        try:
            rag_evidence = retrieve_rag_evidence(statement)
        except Exception:
            rag_evidence = []


        # Step 2: Stage 1 — LLM independently judges the statement
        try:
            llm_result = judge_statement(statement, rag_evidence=rag_evidence)
        except Exception as exc:
            return templates.TemplateResponse(request, "index.html", {
                "model_name": API_MODEL,
                "statement": statement,
                "error": f"LLM analysis failed: {exc}",
            })

        # Step 3: Stage 2 — LLM compares its verdict with ML result
        try:
            comparison = compare_results(statement, raw_result, llm_result)
        except Exception:
            comparison = {}

        agreement = comparison.get("agreement", True)

        # Step 4: Stage 3 — Root cause analysis
        try:
            analysis = analyze_root_cause(statement, raw_result, llm_result, bool(agreement))
        except Exception:
            analysis = {}

        return templates.TemplateResponse(request, "index.html", {
            "model_name": API_MODEL,
            "statement": statement,
            "ml_is_rumor": raw_result["is_rumor"],
            "ml_confidence": raw_result["confidence"],
            "llm_label": llm_result.get("label", "Unknown"),
            "llm_is_rumor": llm_result.get("is_rumor", -1),
            "agreement": agreement,
            "llm_reasoning": llm_result.get("reasoning", ""),
            "supporting_indicators": llm_result.get("supporting_indicators", []),
            "comparison_summary": comparison.get("comparison_summary", ""),
            "root_cause_analysis": analysis.get("root_cause_analysis", ""),
            "key_indicators": analysis.get("key_indicators", []),
            "rag_evidence": rag_evidence,
        })

    return app
