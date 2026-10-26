"""Pluggable literature/web search backends.

Connectivity was probed from this workspace on 2026-10-10:
- arXiv Atom API: reachable -> default backend;
- PubMed eutils / DuckDuckGo HTML: intermittent TLS resets / bot challenges
  from this network -> optional, best-effort with retries;
- Tavily-style keyed backends: not configured (no key in this workspace).

All backends return abstracts when the source provides them, so the research
loop never needs to scrape full pages.
"""

from __future__ import annotations

import re
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Any

import requests

ATOM = "{http://www.w3.org/2005/Atom}"


@dataclass
class SearchResult:
    title: str
    url: str
    abstract: str
    source: str
    published: str = ""


@dataclass
class SearchLog:
    queries: list[dict[str, Any]] = field(default_factory=list)

    def add(self, backend: str, query: str, n_results: int,
            error: str | None = None) -> None:
        self.queries.append({
            "backend": backend,
            "query": query,
            "n_results": n_results,
            "error": error,
        })


class ArxivBackend:
    name = "arxiv"
    endpoint = "http://export.arxiv.org/api/query"
    # arXiv intermittently answers 429 even within politeness limits
    # (observed 2026-10-19: every query rate-limited from this network).
    RETRY_DELAYS = (5.0, 15.0, 45.0)

    def __init__(self, max_results: int = 8, timeout: float = 30.0,
                 politeness_seconds: float = 3.0):
        self.max_results = max_results
        self.timeout = timeout
        self.politeness_seconds = politeness_seconds
        self._last_request = 0.0

    def _get(self, params: dict[str, Any]) -> str:
        last_error: Exception | None = None
        for attempt in range(len(self.RETRY_DELAYS) + 1):
            wait = self.politeness_seconds - (
                time.monotonic() - self._last_request
            )
            if wait > 0:
                time.sleep(wait)
            self._last_request = time.monotonic()
            try:
                response = requests.get(
                    self.endpoint, params=params, timeout=self.timeout,
                    headers={
                        "User-Agent": "era-idea-system/0.1 (research reproduction)"
                    },
                )
                response.raise_for_status()
                return response.text
            except requests.RequestException as error:
                status = (getattr(error.response, "status_code", None)
                          if isinstance(error, requests.HTTPError) else None)
                transient = status is None or status in (429, 500, 502, 503, 504)
                last_error = error
                if attempt < len(self.RETRY_DELAYS) and transient:
                    time.sleep(self.RETRY_DELAYS[attempt])
                    continue
                raise
        raise last_error  # unreachable

    def search(self, query: str) -> list[SearchResult]:
        params = {
            "search_query": f'all:{query}',
            "start": 0,
            "max_results": self.max_results,
            "sortBy": "relevance",
        }
        root = ET.fromstring(self._get(params))
        results: list[SearchResult] = []
        for entry in root.findall(f"{ATOM}entry"):
            title = (entry.findtext(f"{ATOM}title") or "").strip()
            abstract = (entry.findtext(f"{ATOM}summary") or "").strip()
            url = (entry.findtext(f"{ATOM}id") or "").strip()
            published = (entry.findtext(f"{ATOM}published") or "")[:10]
            if title and url:
                results.append(SearchResult(
                    title=re.sub(r"\s+", " ", title),
                    url=url,
                    abstract=re.sub(r"\s+", " ", abstract),
                    source="arxiv",
                    published=published,
                ))
        return results


class PubmedBackend:
    name = "pubmed"
    esearch = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    efetch = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"

    def __init__(self, max_results: int = 8, timeout: float = 30.0,
                 politeness_seconds: float = 1.0, retries: int = 3):
        self.max_results = max_results
        self.timeout = timeout
        self.politeness_seconds = politeness_seconds
        self.retries = retries
        self._last_request = 0.0

    def _get(self, url: str, params: dict[str, Any]) -> str:
        last_error: Exception | None = None
        for attempt in range(self.retries):
            wait = self.politeness_seconds - (
                time.monotonic() - self._last_request
            )
            if wait > 0:
                time.sleep(wait)
            self._last_request = time.monotonic()
            try:
                response = requests.get(
                    url, params=params, timeout=self.timeout,
                    headers={
                        "User-Agent": "era-idea-system/0.1 (research reproduction)"
                    },
                )
                response.raise_for_status()
                return response.text
            except requests.RequestException as error:
                last_error = error
                time.sleep(2.0 * (attempt + 1))
        raise RuntimeError(f"PubMed request failed: {type(last_error).__name__}")

    def search(self, query: str) -> list[SearchResult]:
        ids_text = self._get(self.esearch, {
            "db": "pubmed", "term": query, "retmax": self.max_results,
            "retmode": "json", "sort": "relevance",
        })
        import json
        ids = json.loads(ids_text)["esearchresult"].get("idlist", [])
        if not ids:
            return []
        xml_text = self._get(self.efetch, {
            "db": "pubmed", "id": ",".join(ids), "retmode": "xml",
        })
        root = ET.fromstring(xml_text)
        results: list[SearchResult] = []
        for article in root.iter("PubmedArticle"):
            medline = article.find("MedlineCitation")
            article_node = medline.find("Article") if medline is not None else None
            if article_node is None:
                continue
            title = (article_node.findtext("ArticleTitle") or "").strip()
            pmid = (medline.findtext("PMID") or "").strip() if medline else ""
            abstract_node = article_node.find("Abstract")
            parts = [
                (elem.text or "").strip()
                for elem in (abstract_node.findall("AbstractText")
                             if abstract_node is not None else [])
            ]
            results.append(SearchResult(
                title=title,
                url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else "",
                abstract=" ".join(parts),
                source="pubmed",
            ))
        return results


class DuckDuckGoLiteBackend:
    """Best-effort general web backend; may return nothing when the endpoint
    answers with an anti-bot challenge (observed as HTTP 202)."""
    name = "duckduckgo"
    endpoint = "https://lite.duckduckgo.com/lite/"

    def __init__(self, max_results: int = 8, timeout: float = 30.0):
        self.max_results = max_results
        self.timeout = timeout

    def search(self, query: str) -> list[SearchResult]:
        response = requests.post(
            self.endpoint, data={"q": query}, timeout=self.timeout,
            headers={"User-Agent": "Mozilla/5.0 (research reproduction)"},
        )
        if response.status_code != 200 or not response.text:
            return []
        results: list[SearchResult] = []
        for match in re.finditer(
            r'<a[^>]+class="result-link"[^>]+href="([^"]+)"[^>]*>(.*?)</a>',
            response.text, flags=re.DOTALL,
        ):
            url, title = match.group(1), re.sub(r"<[^>]+>", "", match.group(2))
            title = title.strip()
            if title and url.startswith("http"):
                results.append(SearchResult(
                    title=title, url=url, abstract="", source="duckduckgo",
                ))
            if len(results) >= self.max_results:
                break
        return results


class FixtureBackend:
    """Offline backend for tests and dry runs."""

    def __init__(self, results_by_keyword: dict[str, list[SearchResult]] | None = None):
        self.results_by_keyword = results_by_keyword or {}
        self.name = "fixture"

    def search(self, query: str) -> list[SearchResult]:
        lowered = query.lower()
        for keyword, results in self.results_by_keyword.items():
            if keyword in lowered:
                return results
        return []


def build_backends(names: list[str], search_config: dict[str, Any]):
    max_results = int(search_config.get("max_results_per_query", 8))
    timeout = float(search_config.get("request_timeout_seconds", 30))
    politeness = float(search_config.get("politeness_seconds", 3.0))
    backends = []
    for name in names:
        if name == "arxiv":
            backends.append(ArxivBackend(max_results, timeout, politeness))
        elif name == "pubmed":
            backends.append(PubmedBackend(max_results, timeout, 1.0))
        elif name == "duckduckgo":
            backends.append(DuckDuckGoLiteBackend(max_results, timeout))
        elif name == "fixture":
            backends.append(FixtureBackend())
        else:
            raise ValueError(f"Unknown search backend: {name}")
    return backends
