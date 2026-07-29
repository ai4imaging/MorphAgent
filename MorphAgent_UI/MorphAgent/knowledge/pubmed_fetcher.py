"""Automatic literature retrieval for the RAG knowledge base.

Given a set of keywords, this module searches the biomedical literature and
downloads open-access full-text PDFs into the ``RAG/`` folder, so the existing
RAG pipeline (lite PDF text -> LLM summary) can consume them.

Design notes
------------
* No API key is required. Search uses the public Europe PMC REST API (it mirrors
  PubMed / PubMed Central and reports which records are open access and have a
  PDF). PDF download tries several open-access sources in order:
    1. The PDF URLs Europe PMC reports for the record (``fullTextUrlList``).
    2. The NCBI PMC Open Access service (``oa.fcgi``): a direct PDF link when
       available, otherwise the article ``.tar.gz`` package from which the PDF
       is extracted.
    3. The PMC article PDF endpoint.
* Only ``requests`` is required (already a common dependency); FTP links are
  fetched with the standard library. There is no dependency on biopython.
* Downloads are validated (must start with the ``%PDF`` magic bytes) and named
  deterministically so re-runs are idempotent.

Network note
------------
This step needs outbound internet access to NCBI / EBI. On restricted servers
(no outbound HTTP/FTP, or region-blocked) downloads may fail even though search
succeeds; in that case run it on a machine with internet (optionally behind a
proxy via the standard HTTP(S)_PROXY env vars) or simply drop PDFs into the
RAG/ folder manually.
"""
from __future__ import annotations

import io
import json
import re
import tarfile
import time
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Dict, Optional

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:  # pragma: no cover
    REQUESTS_AVAILABLE = False
    requests = None  # type: ignore

EUROPEPMC_SEARCH = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
NCBI_OA_SERVICE = "https://www.ncbi.nlm.nih.gov/pmc/utils/oa/oa.fcgi"
_BROWSER_UA = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0 Safari/537.36 MorphAgent/1.0"
)


def _session(email: str = "") -> "requests.Session":
    s = requests.Session()
    s.headers.update({"User-Agent": _BROWSER_UA, "Accept": "application/pdf,*/*"})
    return s


def _safe_name(article: Dict, fallback: str) -> str:
    """Build a filesystem-safe PDF file name for an article."""
    ident = article.get("pmcid") or article.get("pmid") or article.get("id") or fallback
    ident = re.sub(r"[^A-Za-z0-9_.-]", "_", str(ident))
    return f"{ident}.pdf"


def search_europepmc(
    query: str,
    max_results: int = 8,
    min_year: int = 0,
    open_access_only: bool = True,
    email: str = "",
) -> List[Dict]:
    """Search Europe PMC and return a list of article metadata dicts.

    Each dict contains: id, source, pmid, pmcid, title, doi, journal, year,
    is_open_access, has_pdf, pdf_urls (list of candidate PDF URLs).
    """
    if not REQUESTS_AVAILABLE:
        raise ImportError("The 'requests' package is required for literature retrieval (pip install requests).")

    q = query.strip()
    if open_access_only:
        q = f"({q}) AND (OPEN_ACCESS:y) AND (HAS_PDF:y)"
    if min_year and min_year > 0:
        q = f"{q} AND (PUB_YEAR:[{min_year} TO 3000])"

    params = {
        "query": q,
        "format": "json",
        "pageSize": max(1, min(max_results, 100)),
        "resultType": "core",
        "sort": "CITED desc",
    }
    sess = _session(email)
    print(f"  [PubMed] Searching Europe PMC: {q}")
    resp = sess.get(EUROPEPMC_SEARCH, params=params, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    results = (data.get("resultList") or {}).get("result") or []

    articles: List[Dict] = []
    for r in results:
        pdf_urls = []
        for u in ((r.get("fullTextUrlList") or {}).get("fullTextUrl") or []):
            if (u.get("documentStyle") == "pdf") and u.get("url"):
                pdf_urls.append(u["url"])
        articles.append({
            "id": r.get("id"),
            "source": r.get("source"),
            "pmid": r.get("pmid"),
            "pmcid": r.get("pmcid"),
            "title": r.get("title"),
            "doi": r.get("doi"),
            "journal": r.get("journalTitle"),
            "year": r.get("pubYear"),
            "is_open_access": r.get("isOpenAccess"),
            "has_pdf": r.get("hasPDF"),
            "pdf_urls": pdf_urls,
        })
    print(f"  [PubMed] Found {len(articles)} candidate articles")
    return articles[:max_results]


def _fetch_bytes(url: str, sess: "requests.Session", timeout: int = 120) -> Optional[bytes]:
    """Fetch raw bytes from an http(s) or ftp URL (ftp via the stdlib)."""
    try:
        if url.startswith("ftp://"):
            with urllib.request.urlopen(url, timeout=timeout) as r:  # noqa: S310
                return r.read()
        resp = sess.get(url, timeout=timeout, allow_redirects=True)
        if resp.status_code == 200:
            return resp.content
    except Exception as e:
        print(f"    [PubMed] fetch failed ({url[:60]}...): {e}")
    return None


def _oa_service_links(pmcid: str, sess: "requests.Session") -> List[tuple]:
    """Query the NCBI PMC Open Access service; return [(format, href), ...]."""
    try:
        resp = sess.get(NCBI_OA_SERVICE, params={"id": pmcid}, timeout=60)
        if resp.status_code != 200:
            return []
        root = ET.fromstring(resp.text)
        return [(lnk.get("format"), lnk.get("href")) for lnk in root.iter("link")]
    except Exception:
        return []


def _pdf_from_tgz(data: bytes) -> Optional[bytes]:
    """Extract the first PDF found inside an OA .tar.gz package."""
    try:
        tf = tarfile.open(fileobj=io.BytesIO(data))
    except Exception:
        return None
    pdfs = [m for m in tf.getmembers() if m.name.lower().endswith(".pdf")]
    if not pdfs:
        return None
    try:
        return tf.extractfile(pdfs[0]).read()
    except Exception:
        return None


def _candidate_pdf_sources(article: Dict, sess: "requests.Session") -> List[tuple]:
    """Yield (kind, url) PDF sources for an article, in priority order.

    kind is one of: 'pdf' (direct PDF bytes) or 'tgz' (package to extract).
    """
    sources: List[tuple] = []
    # 1) URLs reported by Europe PMC for this record.
    for u in article.get("pdf_urls", []) or []:
        sources.append(("pdf", u))
    pmcid = article.get("pmcid")
    if pmcid:
        # 2) Constructed Europe PMC render URL.
        sources.append(("pdf", f"https://europepmc.org/articles/{pmcid}?pdf=render"))
        # 3) NCBI OA service (direct pdf link and/or tgz package).
        for fmt, href in _oa_service_links(pmcid, sess):
            https = href.replace("ftp://ftp.ncbi.nlm.nih.gov", "https://ftp.ncbi.nlm.nih.gov")
            if fmt == "pdf":
                sources.append(("pdf", https))
                sources.append(("pdf", href))  # ftp fallback
            elif fmt == "tgz":
                sources.append(("tgz", https))
                sources.append(("tgz", href))  # ftp fallback
        # 4) PMC article PDF endpoint.
        sources.append(("pdf", f"https://pmc.ncbi.nlm.nih.gov/articles/{pmcid}/pdf/"))
    return sources


def download_pdf(article: Dict, out_dir: Path, email: str = "", overwrite: bool = False) -> Optional[Path]:
    """Download the open-access PDF for a single article. Returns the saved path or None."""
    if not REQUESTS_AVAILABLE:
        raise ImportError("The 'requests' package is required for literature retrieval (pip install requests).")

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / _safe_name(article, fallback="article")
    if out_path.exists() and not overwrite and out_path.stat().st_size > 1024:
        print(f"  [PubMed] Already present, skipping: {out_path.name}")
        return out_path

    sess = _session(email)
    for kind, url in _candidate_pdf_sources(article, sess):
        data = _fetch_bytes(url, sess)
        if not data:
            continue
        if kind == "tgz":
            data = _pdf_from_tgz(data)
            if not data:
                continue
        if data[:4] == b"%PDF" and len(data) > 1024:
            out_path.write_bytes(data)
            host = re.sub(r"^\w+://", "", url).split("/")[0]
            print(f"  [PubMed] Downloaded {out_path.name} ({len(data)//1024} KB) via {host}")
            return out_path

    print(f"  [PubMed] No open-access PDF could be downloaded for "
          f"{article.get('pmcid') or article.get('pmid') or article.get('id')}")
    return None


def fetch_pubmed_literature(
    query: str,
    out_dir: Path,
    max_results: int = 8,
    min_year: int = 0,
    open_access_only: bool = True,
    email: str = "",
    api_key: str = "",  # reserved for NCBI polite access; unused by Europe PMC
    overwrite: bool = False,
) -> List[Path]:
    """Search + download open-access PDFs for ``query`` into ``out_dir``.

    Returns the list of successfully downloaded PDF paths and writes a
    ``retrieval_manifest.json`` describing what was fetched.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[Literature Retrieval] Query: {query!r} (max {max_results} papers)")
    try:
        articles = search_europepmc(
            query, max_results=max_results, min_year=min_year,
            open_access_only=open_access_only, email=email,
        )
    except Exception as e:
        print(f"  [PubMed] Search failed: {e}")
        return []

    saved: List[Path] = []
    manifest = []
    for art in articles:
        path = download_pdf(art, out_dir, email=email, overwrite=overwrite)
        entry = {k: v for k, v in art.items() if k != "pdf_urls"}
        entry["downloaded"] = bool(path)
        entry["file"] = path.name if path else None
        manifest.append(entry)
        if path:
            saved.append(path)
        time.sleep(0.34)  # be polite to the public services

    try:
        with open(out_dir / "retrieval_manifest.json", "w", encoding="utf-8") as f:
            json.dump({"query": query, "articles": manifest}, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

    print(f"[Literature Retrieval] Downloaded {len(saved)}/{len(articles)} open-access PDFs into {out_dir}")
    return saved


if __name__ == "__main__":  # simple manual smoke test
    import argparse
    ap = argparse.ArgumentParser(description="Download open-access PDFs for a query into a folder.")
    ap.add_argument("query")
    ap.add_argument("--out", default="./RAG")
    ap.add_argument("--max-results", type=int, default=5)
    ap.add_argument("--min-year", type=int, default=0)
    ap.add_argument("--all", action="store_true", help="include non-open-access candidates")
    args = ap.parse_args()
    fetch_pubmed_literature(
        args.query, Path(args.out), max_results=args.max_results,
        min_year=args.min_year, open_access_only=not args.all,
    )
