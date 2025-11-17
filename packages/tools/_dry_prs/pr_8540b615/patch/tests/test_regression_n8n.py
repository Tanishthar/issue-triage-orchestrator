import subprocess, time, os, signal, urllib.request
import pytest
from playwright.sync_api import sync_playwright

@pytest.fixture(scope="session")
def n8n_server():
    """Start the n8n development server before tests and shut it down afterwards."""
    repo_path = os.getenv("N8N_REPO_PATH", ".")
    # Use npm to start the dev server
    proc = subprocess.Popen(
        ["npm", "run", "dev"],
        cwd=repo_path,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        preexec_fn=os.setsid,
    )
    # Wait for the server to become reachable (max 30 seconds)
    for _ in range(30):
        try:
            urllib.request.urlopen("http://localhost:5678", timeout=2)
            break
        except Exception:
            time.sleep(1)
    else:
        # Server didn't start – terminate and fail the fixture
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        pytest.fail("n8n dev server failed to start within timeout")
    yield
    # Clean up the server process after tests
    os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    proc.wait()

def test_rootstore_rest_url(n8n_server):
    """Verify that rootStore.restUrl returns a full absolute URL in the UI."""
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        page.goto("http://localhost:5678", wait_until="networkidle")
        # Evaluate the rootStore.restUrl value in the browser context
        rest_url = page.evaluate("()=> window.rootStore?.restUrl")
        assert isinstance(rest_url, str), "restUrl should be a string"
        assert rest_url.startswith("http://localhost:5678/rest"), f"Expected absolute URL, got {rest_url}"
        browser.close()
