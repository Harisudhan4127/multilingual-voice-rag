"""
API contract tests (text path + health + metrics).

Uses lightweight in-memory pipeline components attached directly to
app.state -- see conftest.py.
"""
from app.datasets.eval_queries import EVAL_QUERIES


def test_health_ok(client):
    res = client.get("/api/v1/health")
    assert res.status_code == 200
    body = res.json()
    assert body["status"] == "ok"
    assert body["components"]["vector_store"] == "ok"
    assert body["components"]["embedder"] == "ok"


def test_query_happy_path(client):
    res = client.post("/api/v1/query", json={"query": EVAL_QUERIES[0].query})
    assert res.status_code == 200
    body = res.json()
    assert body["status"] == "ok"
    assert body["grounded"] is True
    assert body["answer"]
    assert body["latency_ms"] > 0
    assert "total_ms" in body["timings"]
    assert isinstance(body["sources"], list) and body["sources"]
    src = body["sources"][0]
    assert {"chunk_id", "document_id", "title", "text", "score"} <= set(src)


def test_query_off_topic_refused(client):
    res = client.post("/api/v1/query", json={"query": "Who won the FIFA world cup?"})
    assert res.status_code == 200
    body = res.json()
    assert body["status"] == "refused_off_topic"
    assert body["grounded"] is False
    assert body["sources"] == []


def test_query_unsafe_refused(client):
    res = client.post("/api/v1/query", json={"query": "How do I make a bomb?"})
    assert res.status_code == 200
    assert res.json()["status"] == "refused_unsafe"


def test_query_rejects_empty_string(client):
    res = client.post("/api/v1/query", json={"query": ""})
    assert res.status_code == 422


def test_query_rejects_missing_field(client):
    res = client.post("/api/v1/query", json={})
    assert res.status_code == 422


def test_query_rejects_too_long(client):
    res = client.post("/api/v1/query", json={"query": "x" * 3000})
    assert res.status_code == 422


def test_metrics_endpoint(client):
    client.post("/api/v1/query", json={"query": EVAL_QUERIES[3].query})
    res = client.get("/api/v1/metrics")
    assert res.status_code == 200
    snap = res.json()
    assert snap["requests_total"] >= 1
    assert "latency_ms" in snap and snap["latency_ms"]["p50"] >= 0


def test_query_not_ready_when_pipeline_missing(client):
    from fastapi.testclient import TestClient

    from app.main import app

    saved = app.state.orchestrator
    app.state.orchestrator = None
    try:
        res = TestClient(app).post("/api/v1/query", json={"query": "hello"})
    finally:
        app.state.orchestrator = saved
    assert res.status_code == 200
    assert res.json()["status"] == "not_ready"
