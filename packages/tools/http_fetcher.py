import os
import time
import hashlib
import requests
from urllib.parse import urlparse
from urllib import robotparser

CACHE_DIR = "packages/tools/_http_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

def _cache_path(url: str) -> str:
    key = hashlib.md5(url.encode("utf-8")).hexdigest()
    return os.path.join(CACHE_DIR, f"{key}.cache")

def allowed_by_robots(url: str, user_agent: str = "*") -> bool:
    parsed = urlparse(url)
    if not parsed.scheme or not parsed.netloc:
        # treat as disallowed for safety
        return False
    robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
    rp = robotparser.RobotFileParser()
    try:
        rp.set_url(robots_url)
        rp.read()
        return rp.can_fetch(user_agent, url)
    except Exception:
        # If robots.txt can't be fetched or parsed, default to conservative allow
        return True

def fetch(url: str, timeout: int = 10, use_cache: bool = True) -> dict:
    """
    Returns: { 'url': url, 'status': int, 'text': str, 'from_cache': bool, 'fetched_at': ts }
    Raises Exception on disallowed or network failure.
    """
    if not allowed_by_robots(url):
        raise Exception(f"Fetch disallowed by robots.txt: {url}")

    cache_path = _cache_path(url)
    if use_cache and os.path.exists(cache_path):
        with open(cache_path, "r", encoding="utf-8") as f:
            body = f.read()
        return {"url": url, "status": 200, "text": body, "from_cache": True, "fetched_at": time.time()}

    resp = requests.get(url, timeout=timeout, headers={"User-Agent": "IssueTriageBot/0.1"})
    resp.raise_for_status()
    text = resp.text

    if use_cache and resp.status_code == 200:
        with open(cache_path, "w", encoding="utf-8") as f:
            f.write(text)

    return {"url": url, "status": resp.status_code, "text": text, "from_cache": False, "fetched_at": time.time()}
