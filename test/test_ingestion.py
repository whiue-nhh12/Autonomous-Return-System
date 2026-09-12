from pathlib import Path
import unittest

from rag.ingestion.parser import DoclingParser
from rag.ingestion.pipeline import IngestionPipeline
from rag.ingestion.schemas import ParsedDocument
from rag.ingestion.validator import SourceValidator


SOURCE_FILE = Path(__file__).with_name("ingestion_sample.md")


class PassthroughVlm:
    """Skip OpenAI VLM calls for documents without image blocks."""

    def run(self, parsed: ParsedDocument) -> ParsedDocument:
        return parsed


class TestIngestionPipeline(unittest.TestCase):
    def test_ingestion_pipeline_from_local_file(self) -> None:
        pipeline = IngestionPipeline(
            validator=SourceValidator(),
            parser=DoclingParser(),
            vlm=PassthroughVlm(),
        )

        result = pipeline.run(str(SOURCE_FILE))

        print("\n=== INGESTION RESULT ===")
        print("Metadata:")
        for key, value in result.metadata.model_dump().items():
            print(f"  {key}: {value}")
        print(f"Blocks: {len(result.blocks)}")
        for index, block in enumerate(result.blocks, start=1):
            print(f"\nBlock {index} [{block.block_type.value}]")
            print(f"  page: {block.page_num}")
            print(f"  path: {block.hierarchical_path or 'root'}")
            print(f"  text: {block.raw_text}")
            print(f"")

        self.assertEqual(result.metadata.source, str(SOURCE_FILE.resolve()))
        self.assertEqual(result.metadata.file_type, "md")
        self.assertTrue(result.blocks)
        self.assertEqual(result.metadata.length_count, len(result.blocks))
        self.assertGreater(result.metadata.char_count, 0)

        block_types = {block.block_type.value for block in result.blocks}
        self.assertIn("heading", block_types)
        self.assertIn("paragraph", block_types)
        self.assertIn("list", block_types)
        self.assertIn("table", block_types)


if __name__ == "__main__":
    unittest.main()
