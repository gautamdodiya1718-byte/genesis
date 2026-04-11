"""
crawler/web_crawler.py
-----------------------
Multi-source web image crawler. Merged from AutoDiff web_crawler.py
with updated imports for the Genesis unified system.

Sources:
  - Openverse public API (CC-licensed, no auth needed)
  - Wikimedia Commons API (public domain)
  - Playwright (Unsplash, JS-rendered pages)
"""

from __future__ import annotations
import asyncio, hashlib, logging, os, re, time
from pathlib import Path
from typing import List, Optional, Dict, Set
from urllib.parse import quote

import httpx

from core.image_utils import validate_image_bytes, bytes_to_image, save_image

logger = logging.getLogger(__name__)


class DownloadResult:
    def __init__(self, url: str, local_path: str, source: str, query: str):
        self.url = url; self.local_path = local_path
        self.source = source; self.query = query
        self.filename = Path(local_path).name

    def to_dict(self) -> dict:
        return {"url": self.url, "local_path": self.local_path,
                "source": self.source, "query": self.query, "filename": self.filename}


class OpenverseScraper:
    API = "https://api.openverse.org/v1/images/"

    async def search(self, query: str, client: httpx.AsyncClient, max_results: int = 20) -> List[Dict]:
        try:
            r = await client.get(self.API, params={"q": query, "page_size": min(max_results, 20),
                "license_type": "all"}, timeout=15)
            r.raise_for_status()
            data = r.json()
            results = []
            for item in data.get("results", []):
                url = item.get("url") or item.get("thumbnail")
                if url:
                    results.append({"url": url, "title": item.get("title",""),
                                    "license": item.get("license",""), "source": "openverse"})
            logger.info(f"Openverse: {len(results)} results for '{query}'")
            return results
        except Exception as e:
            logger.warning(f"Openverse failed for '{query}': {e}")
            return []


class WikimediaScraper:
    API = "https://commons.wikimedia.org/w/api.php"

    async def search(self, query: str, client: httpx.AsyncClient, max_results: int = 20) -> List[Dict]:
        try:
            r = await client.get(self.API, params={
                "action":"query","format":"json","list":"search",
                "srsearch":f"{query} filetype:bitmap","srnamespace":6,
                "srlimit":min(max_results,50)}, timeout=15)
            r.raise_for_status()
            titles = [i["title"] for i in r.json().get("query",{}).get("search",[])]
            if not titles: return []
            return await self._resolve_urls(titles[:20], client)
        except Exception as e:
            logger.warning(f"Wikimedia failed for '{query}': {e}")
            return []

    async def _resolve_urls(self, titles: List[str], client: httpx.AsyncClient) -> List[Dict]:
        try:
            r = await client.get(self.API, params={
                "action":"query","format":"json","titles":"|".join(titles),
                "prop":"imageinfo","iiprop":"url|mime"}, timeout=15)
            r.raise_for_status()
            results = []
            for page in r.json().get("query",{}).get("pages",{}).values():
                for info in page.get("imageinfo",[]):
                    if info.get("mime","").startswith("image/"):
                        results.append({"url": info["url"], "title": page.get("title",""),
                                        "license": "various", "source": "wikimedia"})
            return results
        except Exception as e:
            logger.warning(f"Wikimedia URL resolution failed: {e}")
            return []


class ImageCrawler:
    def __init__(self, cfg):
        self.cfg = cfg
        self.download_dir = Path(cfg.crawler.download_dir)
        self.download_dir.mkdir(parents=True, exist_ok=True)
        self.openverse = OpenverseScraper()
        self.wikimedia = WikimediaScraper()
        self._urls: Set[str] = set()
        self._reg = self.download_dir / ".downloaded_urls.txt"
        self._load_registry()

    def _load_registry(self):
        if self._reg.exists():
            self._urls = set(l.strip() for l in self._reg.read_text().splitlines() if l.strip())

    def _save_url(self, url: str):
        self._urls.add(url)
        with open(self._reg, "a") as f: f.write(url + "\n")

    def crawl(self, query: str, sources: Optional[List[str]] = None,
               max_images: Optional[int] = None) -> List[DownloadResult]:
        sources = sources or ["openverse", "wikimedia"]
        max_images = max_images or self.cfg.crawler.max_images_per_query
        logger.info(f"Crawling '{query}' | sources={sources} | max={max_images}")
        return asyncio.run(self._crawl_async(query, sources, max_images))

    def crawl_multiple(self, queries: List[str], sources=None, max_per_query=None) -> List[DownloadResult]:
        all_results = []
        for i, q in enumerate(queries):
            logger.info(f"Query {i+1}/{len(queries)}: '{q}'")
            all_results.extend(self.crawl(q, sources, max_per_query))
            time.sleep(1)
        return all_results

    async def _crawl_async(self, query, sources, max_images) -> List[DownloadResult]:
        headers = {"User-Agent": self.cfg.crawler.user_agent}
        all_data = []
        async with httpx.AsyncClient(headers=headers, follow_redirects=True) as client:
            if "openverse" in sources:
                for item in await self.openverse.search(query, client, max_images):
                    if item["url"] not in self._urls: all_data.append(item)
            if "wikimedia" in sources:
                for item in await self.wikimedia.search(query, client, max_images):
                    if item["url"] not in self._urls: all_data.append(item)

            # Deduplicate URLs
            seen, unique = set(), []
            for item in all_data:
                if item["url"] not in seen:
                    seen.add(item["url"]); unique.append(item)

            sem = asyncio.Semaphore(self.cfg.crawler.concurrent_downloads)
            tasks = [self._download_one(item, client, query, sem) for item in unique[:max_images]]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            return [r for r in results if isinstance(r, DownloadResult)]

    async def _download_one(self, item, client, query, sem) -> Optional[DownloadResult]:
        url = item["url"]; source = item.get("source", "unknown")
        async with sem:
            try:
                r = await client.get(url, timeout=self.cfg.crawler.timeout_seconds)
                r.raise_for_status()
                data = r.content
                if not validate_image_bytes(data, self.cfg.crawler.min_image_size):
                    return None
                url_hash = hashlib.md5(url.encode()).hexdigest()[:12]
                ext = self._ext(url, r.headers.get("content-type",""))
                fname = f"{source}_{url_hash}{ext}"
                lpath = self.download_dir / fname
                img = bytes_to_image(data)
                if img is None: return None
                save_image(img, str(lpath))
                self._save_url(url)
                return DownloadResult(url=url, local_path=str(lpath), source=source, query=query)
            except Exception as e:
                logger.debug(f"Download failed {url}: {e}")
                return None

    @staticmethod
    def _ext(url: str, ct: str) -> str:
        for mime, ext in [("image/jpeg",".jpg"),("image/png",".png"),("image/webp",".webp")]:
            if mime in ct: return ext
        for ext in [".jpg",".jpeg",".png",".webp"]:
            if ext in url.lower(): return ext
        return ".jpg"
