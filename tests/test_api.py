from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"


def test_reset_state_step_flow():
    response = client.post("/reset", json={"task_id": "task_easy_screen_backend"})
    assert response.status_code == 200

    response = client.get("/state")
    assert response.status_code == 200

    step = client.post(
        "/step",
        json={
            "action_type": "shortlist_candidates",
            "payload": {"candidate_ids": ["C001", "C002"]},
        },
    )
    assert step.status_code == 200
    body = step.json()
    assert "observation" in body
    assert "reward" in body


def test_reset_allows_empty_body():
    response = client.post("/reset")
    assert response.status_code == 200
    body = response.json()
    assert "observation" in body
    assert "state" in body
