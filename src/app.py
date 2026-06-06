from pathlib import Path

from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from src.config import API_MODEL
from src.model import classify_statement
from src.api_service import enrich_classification, explain_result


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

        # Step 2: API enriches with text (label, reasoning, key indicators)
        try:
            enriched = enrich_classification(statement, raw_result)
        except Exception:
            enriched = {}

        classification: dict[str, object] = {**raw_result, **enriched}

        # Step 3: API generates user-friendly explanation
        try:
            explanation = explain_result(statement, classification)
        except Exception:
            explanation = None

        return templates.TemplateResponse(request, "index.html", {
            "model_name": API_MODEL,
            "statement": statement,
            "classification": classification,
            "explanation": explanation,
        })

    return app
