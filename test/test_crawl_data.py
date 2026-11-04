import json
import tempfile
import unittest
from pathlib import Path

from rag.data.tool.crawl_data import crawl_return_refund_policies


class FakeFirecrawl:
    def __init__(self):
        self.scraped_urls = []

    def search(self, query, *, limit):
        return {
            "web": [
                {
                    "url": "https://help.shopee.vn/portal/article/return-policy",
                    "title": "Shopee đổi trả và hoàn tiền",
                },
                {
                    "url": "https://seller-vn.tiktok.com/university/essay/return-policy",
                    "title": "TikTok Shop return and refund policy",
                },
                {
                    "url": "https://shopee.vn.attacker.example/return-policy",
                    "title": "Untrusted result",
                },
            ]
        }

    def scrape(self, url, *, formats):
        self.scraped_urls.append(url)
        return {
            "markdown": "Chính sách đổi trả, hoàn tiền.",
            "metadata": {"title": f"Policy for {url}"},
        }


class TestCrawlData(unittest.TestCase):
    def test_scrapes_only_official_domains_and_writes_markdown_manifest(self):
        client = FakeFirecrawl()
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            documents = crawl_return_refund_policies(
                client=client,
                output_dir=output_dir,
                limit_per_platform=2,
            )

            manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
            markdown_files = list(output_dir.glob("*.md"))
            markdown_contents = [path.read_text(encoding="utf-8") for path in markdown_files]

        self.assertEqual(len(documents), 2)
        self.assertEqual(len(client.scraped_urls), 2)
        self.assertTrue(all("attacker.example" not in url for url in client.scraped_urls))
        self.assertEqual(len(manifest), 2)
        self.assertTrue(all(record["tool"] == "firecrawl" for record in manifest))
        self.assertEqual(len(markdown_files), 2)
        self.assertTrue(all("Source URL:" in content for content in markdown_contents))

    def test_appends_new_documents_to_existing_manifest(self):
        client = FakeFirecrawl()
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            manifest_path = output_dir / "manifest.json"
            previous_record = {"platform": "tiktok_shop", "title": "Previous crawl"}
            manifest_path.write_text(json.dumps([previous_record]), encoding="utf-8")

            crawl_return_refund_policies(
                client=client,
                output_dir=output_dir,
                limit_per_platform=2,
            )

            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

        self.assertEqual(manifest[0], previous_record)
        self.assertEqual(len(manifest), 3)
        self.assertTrue(all(record["tool"] == "firecrawl" for record in manifest[1:]))


if __name__ == "__main__":
    unittest.main()