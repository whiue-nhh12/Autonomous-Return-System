from pathlib import Path

import unittest

from rag.ingestion.parser import DoclingParser
from rag.ingestion.pipeline import IngestionPipeline
from rag.core.schemas.ingestion_schemas import ParsedDocument
from rag.ingestion.validator import SourceValidator
from rag.ingestion.vlm.vlm import VlmEnricher


SOURCE_FILE = Path(__file__).with_name("T6S3-1-3.pdf").resolve()


class PassthroughVlm:
    """Skip OpenAI VLM calls for documents without image blocks."""

    def run(self, parsed: ParsedDocument) -> ParsedDocument:
        return parsed


class TestIngestionPipeline(unittest.TestCase):
    def test_ingestion_pipeline_from_local_file(self) -> None:
        pipeline = IngestionPipeline(
            validator=SourceValidator(),
            parser=DoclingParser(),
            vlm=VlmEnricher(),
        )

        result = pipeline.run(str(SOURCE_FILE))

        print("\n=== INGESTION RESULT ===")
        print("Metadata:")
        for key, value in result.metadata.model_dump().items():
            print(f"  {key}: {value}")
        print(f"Blocks: {len(result.blocks)}")
        for index, block in enumerate(result.blocks, start=1):
            print(f"\nBlock {index} [{block.block_type.value}]")
            print(f"  block_id : {block.block_id}")
            print(f"  content : {block.raw_text}")
            print(f"  page: {block.page_num}")
            print(f"  path: {block.hierarchical_path or 'root'}")
            print(f"  image_data: {block.image_data if block.image_data else 'None'}")
            """
            print(f"\nBlock {index} [{block.block_type.value}]")
            print(f"image_data : {block.image_data if block.image_data else 'None'}")
            print(f"  page: {block.page_num}")
            print(f"  path: {block.hierarchical_path or 'root'}")
            print(f"  text: {block.raw_text}")
            """
        self.assertEqual(result.metadata.source, str(SOURCE_FILE.resolve()))
        self.assertEqual(result.metadata.file_type, "pdf")
        self.assertTrue(result.blocks)
        self.assertEqual(result.metadata.length_count, len(result.blocks))
        self.assertGreater(result.metadata.char_count, 0)




if __name__ == "__main__":
    unittest.main()
