import hashlib
import json
import re
import unicodedata
from pathlib import Path

import httpx
from bs4 import BeautifulSoup

import time
from ddgs import DDGS
from urllib.parse import urlparse
import threading

CACHE_DIR = Path("/home/morg/students/gottesman3/rlm/scraped_webpages")
URL_MAP_FILE = CACHE_DIR / "url_map.json"

last_domain_hit: dict[str, float] = {}
_lock = threading.Lock()
CRAWL_DELAY = 1.0

def _normalize_url(url: str) -> str:
    """Normalize URL to a consistent cache key."""
    url = url.strip().lower()
    url = url.rstrip("/")
    for prefix in ("https://", "http://", "www."):
        url = url.removeprefix(prefix)
    return url


def _url_to_filename(url: str) -> str:
    """Convert a URL to a safe filename using a hash."""
    normalized = _normalize_url(url)
    url_hash = hashlib.md5(normalized.encode()).hexdigest()[:12]
    # Make a readable prefix from the URL
    slug = re.sub(r"[^\w]", "_", normalized)[:60]
    return f"{slug}_{url_hash}.txt"


def _load_url_map() -> dict:
    if URL_MAP_FILE.exists():
        return json.loads(URL_MAP_FILE.read_text())
    return {}


def _save_url_map(url_map: dict):
    URL_MAP_FILE.write_text(json.dumps(url_map, indent=2))


def scrape_url(url: str) -> str:
    domain = urlparse(url).netloc

    with _lock:
        now = time.time()
        last_hit = last_domain_hit.get(domain)
        if last_hit:
            elapsed = now - last_hit
            if elapsed < CRAWL_DELAY:
                time.sleep(CRAWL_DELAY - elapsed)

    CACHE_DIR.mkdir(exist_ok=True)
    url_map = _load_url_map()
    normalized = _normalize_url(url)

    # Cache hit
    if normalized in url_map:
        cached_file = CACHE_DIR / url_map[normalized]
        if cached_file.exists():
            return cached_file.read_text()

    # Cache miss — scrape
    try:
        response = httpx.get(url, timeout=10, follow_redirects=True, headers={
            "User-Agent": "Mozilla/5.0 (compatible; research-bot/1.0)"
        })
        last_domain_hit[domain] = time.time()
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")

        for tag in soup(["script", "style", "nav", "footer", "header", "aside", "form"]):
            tag.decompose()

        # Prefer main content area
        main = (
            soup.find("main") or
            soup.find("article") or
            soup.find(id="content") or
            soup.find(class_="content") or
            soup.body
        )

        text = main.get_text(separator=" ")
        text = re.sub(r"\s+", " ", text).strip()
        text = unicodedata.normalize("NFKC", text)

    except Exception as e:
        text = f"[Could not scrape: {e}]"

    # Save to disk and update map
    filename = _url_to_filename(url)
    (CACHE_DIR / filename).write_text(text)
    url_map[normalized] = filename
    _save_url_map(url_map)

    return text

def web_search(query: str, max_results: int = 5) -> str:
    with DDGS() as ddgs:
        results = list(ddgs.text(query, max_results=max_results))
    if not results:
        return "No results found."

    output = []
    for i, r in enumerate(results, 1):
        url = r["href"]
        domain = urlparse(url).netloc

        # Rate limit per domain
        last_hit = last_domain_hit.get(domain)
        if last_hit:
            elapsed = time.time() - last_hit
            if elapsed < CRAWL_DELAY:
                time.sleep(CRAWL_DELAY - elapsed)

        scraped = scrape_url(url)
        last_domain_hit[domain] = time.time()
        output.append(f"{i}. {r['title']}\n{url}\n{scraped}")
    return "\n\n".join(output)