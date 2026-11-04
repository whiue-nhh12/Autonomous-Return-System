"""Fetch and extract the official TikTok Shop return/refund policy."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup, Tag

_LOGGER = logging.getLogger(__name__)
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_OUTPUT_DIR = _PROJECT_ROOT / "rag" / "data" / "tool" / "return_refund"
TIKTOKSHOP_POLICY_URL = (
	"https://seller-vn.tiktok.com/university/essay?knowledge_id=6837773789234946&lang=vi-VN",
	"https://seller-vn.tiktok.com/university/essay?identity=1&role=1&knowledge_id=1766935302801169#th-i-gian-chi-ti-t",
	

)
_ALLOWED_HOST = "seller-vn.tiktok.com"
_MIN_CONTENT_LENGTH = 100


def _normalized_text(element: Tag) -> str:
	return re.sub(r"\s+", " ", element.get_text(" ", strip=True)).strip()


def _is_tiktokshop_policy_url(url: str) -> bool:
	try:
		parsed = urlparse(url)
	except ValueError:
		return False
	hostname = (parsed.hostname or "").lower().rstrip(".")
	return (
		parsed.scheme == "https"
		and hostname == _ALLOWED_HOST
		and parsed.path == "/university/essay"
	)


def _block_markdown(block: Tag) -> str:
	if block.name in {"h1", "h2", "h3", "h4", "h5", "h6"}:
		level = int(block.name[1])
		text = _normalized_text(block)
		return f"{'#' * level} {text}" if text else ""
	if block.name == "tr":
		cells = [
			_normalized_text(cell)
			for cell in block.find_all(["th", "td"], recursive=False)
		]
		return " | ".join(cell for cell in cells if cell)

	text = _normalized_text(block)
	if not text:
		return ""
	return f"- {text}" if block.name == "li" else text


def parse_tiktokshop_policy(html: str) -> tuple[str, str]:
	"""Extract the article title and policy text from TikTok Shop's HTML."""
	soup = BeautifulSoup(html, "html.parser")
	article = soup.select_one("article.article-main")
	if article is None:
		raise ValueError("TikTok Shop article container was not found in the HTML")

	title_element = article.select_one(".article-title")
	title = _normalized_text(title_element) if title_element else ""
	if not title:
		raise ValueError("TikTok Shop policy title was not found in the article")

	content = article.select_one(".converted-html")
	if content is None:
		raise ValueError("TikTok Shop policy content was not found in the article")

	blocks = [
		_block_markdown(block)
		for block in content.find_all(["h1", "h2", "h3", "h4", "h5", "h6", "p", "li", "tr"])
	]
	markdown = "\n\n".join(block for block in blocks if block)
	if len(markdown) < _MIN_CONTENT_LENGTH:
		raise ValueError("TikTok Shop policy content is missing or unexpectedly short")
	return title, markdown


def fetch_tiktokshop_policy(
	*,
	output_dir: str | Path = _DEFAULT_OUTPUT_DIR,
	url: str = TIKTOKSHOP_POLICY_URL,
) -> Path:
	"""Fetch the official policy page, extract its article, and save it as Markdown."""
	if not _is_tiktokshop_policy_url(url):
		raise ValueError(f"Refusing to fetch a non-official TikTok Shop policy URL: {url}")

	response = requests.get(
		url,
		headers={"User-Agent": "Mozilla/5.0 (compatible; PolicyDataFetcher/1.0)"},
		timeout=30,
	)
	response.raise_for_status()
	if not _is_tiktokshop_policy_url(response.url):
		raise ValueError(f"TikTok Shop redirected to a non-official URL: {response.url}")

	title, markdown = parse_tiktokshop_policy(response.text)
	crawl_time = datetime.now(timezone.utc)
	crawled_at = crawl_time.isoformat()
	url_hash = hashlib.sha256(url.encode("utf-8")).hexdigest()[:10]
	filename = (
		f"tiktok_shop-return-refund-{url_hash}-"
		f"{crawl_time.strftime('%Y%m%dT%H%M%S%fZ')}.md"
	)
	destination = Path(output_dir)
	destination.mkdir(parents=True, exist_ok=True)
	output_path = destination / filename
	output_path.write_text(
		f"# {title}\n\nPlatform: tiktok_shop\nSource URL: {url}\n"
		f"Crawled at (UTC): {crawled_at}\n\n---\n\n{markdown}\n",
		encoding="utf-8",
	)

	manifest_path = destination / "manifest.json"
	manifest = (
		json.loads(manifest_path.read_text(encoding="utf-8"))
		if manifest_path.exists()
		else []
	)
	if not isinstance(manifest, list):
		raise ValueError(f"Expected a JSON array in manifest: {manifest_path}")
	manifest.append(
		{
			"platform": "tiktok_shop",
			"tool": "beautifulsoup",
			"title": title,
			"url": url,
			"crawled_at": crawled_at,
			"markdown_path": filename,
		}
	)
	manifest_path.write_text(
		json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
		encoding="utf-8",
	)
	_LOGGER.info("Saved TikTok Shop policy to %s", output_path)
	return output_path


def main() -> int:
	parser = argparse.ArgumentParser(
		description="Fetch the official TikTok Shop return/refund policy with BeautifulSoup"
	)
	parser.add_argument("--output-dir", type=Path, default=_DEFAULT_OUTPUT_DIR)
	args = parser.parse_args()

	logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
	output_path = fetch_tiktokshop_policy(output_dir=args.output_dir)
	print(f"Saved TikTok Shop policy to {output_path}")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
