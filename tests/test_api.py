import pytest
from httpx import AsyncClient, ASGITransport
from apps.orchestrator.main import app

@pytest.mark.asyncio
async def test_start_endpoint():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        payload = {"repo_url":"https://github.com/test/repo","issue_text":"Sample crash"}
        r = await ac.post("/start", json=payload)
        assert r.status_code == 200
        body = r.json()
        assert "final_state" in body
        assert body["final_state"]["repo_url"] == payload["repo_url"]
