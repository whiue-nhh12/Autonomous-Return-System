"""Crawl public Shopee and TikTok Shop return/refund policies with Firecrawl."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

_LOGGER = logging.getLogger(__name__)
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_OUTPUT_DIR = _PROJECT_ROOT / "rag" / "data" / "tool" / "return_refund"


@dataclass(frozen=True)
class PolicySource:
	platform: str
	query: str
	allowed_domains: tuple[str, ...]


@dataclass(frozen=True)
class CrawledPolicy:
	platform: str
	tool: str
	title: str
	url: str
	crawled_at: str
	markdown_path: str


POLICY_SOURCES = (
	PolicySource(
		platform="shopee",
		query="https://help.shopee.vn/portal/4/article/77251-CH%C3%8DNH-S%C3%81CH-TR%E1%BA%A2-H%C3%80NG-V%C3%80-HO%C3%80N-TI%E1%BB%80N",
		allowed_domains=("help.shopee.vn",),
	),
	PolicySource(
			platform="shopee",
			query="https://help.shopee.vn/portal/4/article/77262",
			allowed_domains=("help.shopee.vn",),
		),
	PolicySource(
				platform="shopee",
				query="https://help.shopee.vn/portal/4/article/77250",
				allowed_domains=("help.shopee.vn",),
			),
	PolicySource(
					platform="shopee",
					query="https://help.shopee.vn/portal/4/article/77247",
					allowed_domains=("help.shopee.vn",),
				),
	PolicySource(
					platform="shopee",
					query="https://banhang.shopee.vn/edu/article/8327/Huong-dan-dang-ban-thuc-pham-tuoi-sang-dong-lanh",
					allowed_domains=("banhang.shopee.vn",),
				),
	PolicySource(
					platform="shopee",
					query="https://banhang.shopee.vn/edu/article/239/Huong-dan-dong-goi-hang-chuan-Shopee",
					allowed_domains=("banhang.shopee.vn",),
				),
	PolicySource(
					platform="shopee",
					query="https://help.shopee.vn/portal/4/article/188931-%5BTr%E1%BA%A3-h%C3%A0ng/Ho%C3%A0n-ti%E1%BB%81n%5D-Nh%E1%BB%AFng-quy-%C4%91%E1%BB%8Bnh-chung-v%E1%BB%81-Tr%E1%BA%A3-h%C3%A0ng/Ho%C3%A0n-ti%E1%BB%81n-c%E1%BB%A7a-Shopee",
					allowed_domains=("help.shopee.vn",),
				),
	PolicySource(
						platform="shopee",
						query="https://help.shopee.vn/portal/4/article/189473",
						allowed_domains=("help.shopee.vn",),
				),
	PolicySource(
							platform="shopee",
							query="https://help.shopee.vn/portal/4/article/190242",
							allowed_domains=("help.shopee.vn",),
				),
	PolicySource(
					platform="shopee",
					query="https://help.shopee.vn/portal/4/article/189477",
					allowed_domains=("help.shopee.vn",),
				),
	PolicySource(
					platform="shopee",
					query="https://banhang.shopee.vn/edu/article/10626",
					allowed_domains=("banhang.shopee.vn",),
				),
	)


def _value(item: Any, key: str, default: Any = None) -> Any:
	if isinstance(item, dict):
		return item.get(key, default)
	return getattr(item, key, default)


def _web_results(response: Any) -> list[Any]:
	web = _value(response, "web", [])
	if isinstance(web, dict):
		web = web.get("results", [])
	if isinstance(web, list):
		return web
	results = _value(response, "results", [])
	return results if isinstance(results, list) else []


def _is_official_url(url: str, allowed_domains: tuple[str, ...]) -> bool:
	try:
		parsed = urlparse(url)
		hostname = (parsed.hostname or "").lower().rstrip(".")
	except ValueError:
		return False
	return parsed.scheme in {"http", "https"} and any(
		hostname == domain or hostname.endswith(f".{domain}")
		for domain in allowed_domains
	)


def _safe_filename(platform: str, url: str) -> str:
	slug = re.sub(r"[^a-zA-Z0-9]+", "-", urlparse(url).path).strip("-").lower()
	url_hash = hashlib.sha256(url.encode("utf-8")).hexdigest()[:10]
	return f"{platform}-{slug[:80] or 'policy'}-{url_hash}.md"


def _scraped_markdown(response: Any) -> tuple[str, str]:
	markdown = _value(response, "markdown", "")
	metadata = _value(response, "metadata", {}) or {}
	title = _value(metadata, "title", "") or _value(metadata, "og_title", "")
	return str(markdown or "").strip(), str(title or "").strip()


def crawl_return_refund_policies(
	*,
	api_key: str | None = None,
	output_dir: str | Path = _DEFAULT_OUTPUT_DIR,
	limit_per_platform: int = 5,
	client: Any | None = None,
) -> list[CrawledPolicy]:
	"""Search official policy pages, scrape their markdown, and save them locally."""
	if limit_per_platform <= 0:
		raise ValueError("limit_per_platform must be greater than zero")
	if client is None:
		try:
			from firecrawl import Firecrawl
		except ImportError as exc:
			raise ImportError(
				"Firecrawl SDK is required. Install it with: pip install firecrawl-py"
			) from exc
		client = Firecrawl(api_key=api_key or os.getenv("FIRECRAWL_API_KEY"))

	destination = Path(output_dir)
	destination.mkdir(parents=True, exist_ok=True)
	crawled_at = datetime.now(timezone.utc).isoformat()
	documents: list[CrawledPolicy] = []
	seen_urls: set[str] = set()

	for source in POLICY_SOURCES:
		try:
			print(f"Searching for {source.platform} policies...")
			response = client.search(source.query, limit=limit_per_platform)
			print(f"Found {len(_web_results(response))} results for {source.platform}")
		except Exception:
			_LOGGER.exception("Firecrawl search failed for %s", source.platform)
			continue

		accepted = 0
		for result in _web_results(response):
			url = str(_value(result, "url", "") or _value(result, "source_url", "")).strip()
			if not url or url in seen_urls or not _is_official_url(url, source.allowed_domains):
				continue
			seen_urls.add(url)
			try:
				print(f"Scraping {url}...")
				scraped = client.scrape(url, formats=["markdown"])
				print(f"Scraped {url} successfully")
			except Exception as e:
				print(f"Failed to scrape {url}")
				_LOGGER.exception("Firecrawl scrape failed for %s", url, exc_info=e)
				continue
			markdown, scraped_title = _scraped_markdown(scraped)
			if not markdown:
				_LOGGER.warning("Firecrawl returned no markdown for %s", url)
				continue

			title = scraped_title or str(_value(result, "title", "") or url)
			filename = _safe_filename(source.platform, url)
			(destination / filename).write_text(
				f"# {title}\n\nPlatform: {source.platform}\nSource URL: {url}\n"
				f"Crawled at (UTC): {crawled_at}\n\n---\n\n{markdown}\n",
				encoding="utf-8",
			)
			documents.append(
				CrawledPolicy(
					platform=source.platform,
					tool="firecrawl",
					title=title,
					url=url,
					crawled_at=crawled_at,
					markdown_path=filename,
				)
			)
			accepted += 1
			if accepted >= limit_per_platform:
				break

	manifest_path = destination / "manifest.json"
	manifest = (
		json.loads(manifest_path.read_text(encoding="utf-8"))
		if manifest_path.exists()
		else []
	)
	if not isinstance(manifest, list):
		raise ValueError(f"Expected a JSON array in manifest: {manifest_path}")
	manifest.extend(asdict(document) for document in documents)
	manifest_path.write_text(
		json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
		encoding="utf-8",
	)
	return documents


def main() -> int:
	parser = argparse.ArgumentParser(description="Crawl official Shopee/TikTok Shop return-refund policies")
	parser.add_argument("--output-dir", type=Path, default=_DEFAULT_OUTPUT_DIR)
	parser.add_argument("--limit", type=int, default=7, help="Maximum pages to save per platform")
	args = parser.parse_args()

	logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
	documents = crawl_return_refund_policies(
		output_dir=args.output_dir,
		limit_per_platform=args.limit,
		api_key=os.getenv("FIRECRAWL_API_KEY"),
	)
	print(f"Saved {len(documents)} policy pages to {args.output_dir}")
	return 0 if documents else 1


if __name__ == "__main__":
	raise SystemExit(main())
