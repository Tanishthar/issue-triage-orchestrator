import pathlib
import re
import requests
import pytest

@pytest.fixture(scope="module")
def readme_path():
    # Assume the test runs from the repository root
    return pathlib.Path(__file__).parent.parent / "README.md"

def extract_link(text, link_text):
    # Regex to find markdown link with given visible text
    pattern = rf"\[\s*{re.escape(link_text)}\s*\]\(([^)]+)\)"
    match = re.search(pattern, text)
    return match.group(1) if match else None

def test_python_reddit_bot_link_is_valid(readme_path):
    # Load README content
    readme_content = readme_path.read_text(encoding="utf-8")
    # Extract the URL for the specific link text
    url = extract_link(readme_content, "Python: Build a Reddit bot")
    assert url is not None, "Link with text 'Python: Build a Reddit bot' not found in README"
    # If the URL is relative, convert it to an absolute GitHub raw URL for testing
    if url.startswith("./") or not url.startswith("http"):
        # Assume the repo is hosted at https://github.com/Tanishthar/AI-ML-Projects
        base = "https://github.com/Tanishthar/AI-ML-Projects/blob/main/"
        url = base + url.lstrip("./")
    # Perform a GET request (allow redirects) and ensure we get a 200 OK
    response = requests.get(url, timeout=10)
    assert response.status_code == 200, f"Link URL '{url}' returned status {response.status_code}, expected 200"
