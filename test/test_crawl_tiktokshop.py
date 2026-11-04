import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from rag.data.tool.crawl_tiktokshop import (
	TIKTOKSHOP_POLICY_URL,
	fetch_tiktokshop_policy,
	parse_tiktokshop_policy,
)


POLICY_HTML = """
<html>
  <body>
    <nav>Unrelated sidebar policy</nav>
    <article class="article-main content-col">
      <h1 class="article-title">Chính sách hủy đơn hàng, trả hàng và hoàn tiền</h1>
      <div class="converted-html">
        <h2>Tổng quan</h2>
        <p>Đây là nội dung chính sách chính thức của TikTok Shop.</p>
        <ul><li>Người mua có thể yêu cầu trả hàng và hoàn tiền.</li></ul>
      </div>
    </article>
  </body>
</html>
"""


class TestCrawlTikTokShop(unittest.TestCase):
    def test_parse_extracts_article_without_sidebar(self):
        title, markdown = parse_tiktokshop_policy(POLICY_HTML)

        self.assertEqual(title, "Chính sách hủy đơn hàng, trả hàng và hoàn tiền")
        self.assertIn("## Tổng quan", markdown)
        self.assertIn("- Người mua có thể yêu cầu trả hàng và hoàn tiền.", markdown)
        self.assertNotIn("Unrelated sidebar policy", markdown)

    def test_fetch_saves_policy_markdown(self):
        response = Mock()
        response.url = TIKTOKSHOP_POLICY_URL
        response.text = POLICY_HTML
        response.raise_for_status.return_value = None

        with tempfile.TemporaryDirectory() as temp_dir, patch(
            "rag.data.tool.crawl_tiktokshop.requests.get",
            return_value=response,
        ) as get:
            output_path = fetch_tiktokshop_policy(output_dir=temp_dir)
            contents = output_path.read_text(encoding="utf-8")
            self.assertTrue(output_path.exists())
            manifest = json.loads((output_path.parent / "manifest.json").read_text(encoding="utf-8"))

        get.assert_called_once()
        self.assertIn("Platform: tiktok_shop", contents)
        self.assertIn("Source URL: " + TIKTOKSHOP_POLICY_URL, contents)
        self.assertIn("- Người mua có thể yêu cầu trả hàng và hoàn tiền.", contents)
        self.assertNotIn("Unrelated sidebar policy", contents)
        self.assertEqual(len(manifest), 1)
        self.assertEqual(manifest[0]["platform"], "tiktok_shop")
        self.assertEqual(manifest[0]["tool"], "beautifulsoup")
        self.assertEqual(manifest[0]["markdown_path"], output_path.name)

    def test_fetch_appends_to_existing_manifest(self):
        response = Mock()
        response.url = TIKTOKSHOP_POLICY_URL
        response.text = POLICY_HTML
        response.raise_for_status.return_value = None

        with tempfile.TemporaryDirectory() as temp_dir, patch(
            "rag.data.tool.crawl_tiktokshop.requests.get",
            return_value=response,
        ):
            manifest_path = Path(temp_dir) / "manifest.json"
            previous_record = {"platform": "shopee", "title": "Previous crawl"}
            manifest_path.write_text(
                json.dumps([previous_record]),
                encoding="utf-8",
            )

            output_path = fetch_tiktokshop_policy(output_dir=temp_dir)
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

        self.assertEqual(manifest[0], previous_record)
        self.assertEqual(len(manifest), 2)
        self.assertEqual(manifest[1]["tool"], "beautifulsoup")
        self.assertEqual(manifest[1]["markdown_path"], output_path.name)

    def test_rejects_non_official_url(self):
        with self.assertRaises(ValueError):
            fetch_tiktokshop_policy(url="https://seller-vn.tiktok.com.attacker.example/university/essay")

    def test_parse_rejects_html_without_article(self):
        with self.assertRaises(ValueError):
            parse_tiktokshop_policy("<html><body><p>Not the policy</p></body></html>")


if __name__ == "__main__":
    unittest.main()
