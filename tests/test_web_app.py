"""Regression tests for web result-state rendering."""

from __future__ import annotations

import asyncio
import inspect
from unittest.mock import Mock

from fastapi import FastAPI, Request
from starlette.responses import Response

from src.web import app as web_app


def _analyze(app: FastAPI, statement: str) -> Response:
    route = next(route for route in app.routes if route.path == "/analyze")
    request = Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/analyze",
            "root_path": "",
            "scheme": "http",
            "query_string": b"",
            "headers": [],
            "client": ("test", 50000),
            "server": ("test", 80),
        }
    )
    response = route.endpoint(request, statement)
    if inspect.isawaitable(response):
        return asyncio.run(response)
    return response


def test_invalid_llm_verdict_renders_unavailable(monkeypatch) -> None:
    compare = Mock()
    analyze = Mock()
    monkeypatch.setattr(
        web_app,
        "classify_statement",
        Mock(return_value={"is_rumor": 0, "confidence": 0.8}),
    )
    monkeypatch.setattr(web_app, "judge_statement", Mock(return_value={}))
    monkeypatch.setattr(web_app, "compare_results", compare)
    monkeypatch.setattr(web_app, "analyze_root_cause", analyze)
    app = web_app.create_app(enable_rag=False, enable_llm=True)
    analyze_route = next(route for route in app.routes if route.path == "/analyze")

    response = _analyze(app, "Example claim")
    response_text = response.body.decode()

    assert response.status_code == 200
    assert "LLM comparison is unavailable." in response_text
    assert "UNAVAILABLE" in response_text
    assert "agree on this verdict" not in response_text
    compare.assert_not_called()
    analyze.assert_not_called()
    assert not inspect.iscoroutinefunction(analyze_route.endpoint)


def test_missing_comparison_does_not_claim_agreement(monkeypatch) -> None:
    analyze = Mock()
    monkeypatch.setattr(
        web_app,
        "classify_statement",
        Mock(return_value={"is_rumor": 1, "confidence": 0.7}),
    )
    monkeypatch.setattr(
        web_app,
        "judge_statement",
        Mock(return_value={"is_rumor": 1, "label": "RUMOR"}),
    )
    monkeypatch.setattr(web_app, "compare_results", Mock(return_value={}))
    monkeypatch.setattr(web_app, "analyze_root_cause", analyze)
    app = web_app.create_app(enable_rag=False, enable_llm=True)

    response = _analyze(app, "Example claim")
    response_text = response.body.decode()

    assert response.status_code == 200
    assert "LLM comparison is unavailable." in response_text
    assert "RUMOR" in response_text
    assert "agree on this verdict" not in response_text
    analyze.assert_not_called()


def test_checkpoint_failure_renders_classification_error(monkeypatch) -> None:
    monkeypatch.setattr(
        web_app,
        "classify_statement",
        Mock(side_effect=OSError("checkpoint unavailable")),
    )
    app = web_app.create_app(enable_rag=False, enable_llm=False)

    response = _analyze(app, "Example claim")
    response_text = response.body.decode()

    assert response.status_code == 200
    assert "Classification failed: checkpoint unavailable" in response_text
    assert "ML Verdict" not in response_text
