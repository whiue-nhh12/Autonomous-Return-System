import unittest
from pathlib import Path

from rag.chunking import DocumentChunker
from rag.cleaning.schemas import Cleaned_Document
from rag.core.schemas.ingestion_schemas import BlockType, ContentBlock, DocumentMetadata


class TestDocumentChunker(unittest.TestCase):
    def make_document(self, blocks: list[ContentBlock]) -> Cleaned_Document:
        return Cleaned_Document(
            metadata=DocumentMetadata(source=str(Path("sample.md")), file_type="md"),
            blocks=blocks,
        )

    def test_splits_large_heading_section_into_subsections(self) -> None:
        blocks = [
            ContentBlock(raw_text="Main", block_type=BlockType.HEADING, level=1, hierarchical_path="Main"),
            ContentBlock(raw_text="Section A", block_type=BlockType.HEADING, level=2, hierarchical_path="Main"),
            ContentBlock(raw_text="a " * 20, hierarchical_path="Main > Section A"),
            ContentBlock(raw_text="Section B", block_type=BlockType.HEADING, level=2, hierarchical_path="Main"),
            ContentBlock(raw_text="b " * 20, hierarchical_path="Main > Section B"),
        ]

        result = DocumentChunker(max_chunk_size=45, overlap=5, trace_id="test").run(self.make_document(blocks))

        self.assertEqual(result.block_number, 2)
        self.assertEqual([chunk.section for chunk in result.block], ["Section A", "Section B"])

    def test_packs_bullets_and_preserves_metadata(self) -> None:
        blocks = [
            ContentBlock(raw_text="Topic", block_type=BlockType.HEADING, level=1, hierarchical_path="Topic"),
            ContentBlock(
                raw_text="- first item\n- second item\n- third item",
                block_type=BlockType.LIST,
                hierarchical_path="Topic",
            ),
        ]

        result = DocumentChunker(max_chunk_size=20, overlap=3, trace_id="test").run(self.make_document(blocks))

        self.assertGreater(len(result.block), 1)
        self.assertTrue(all(chunk.top_heading == "Topic" for chunk in result.block))
        self.assertTrue(all(len(chunk.raw_text) <= 20 for chunk in result.block))

    def test_recursive_fallback_overlaps_paragraph_and_keeps_hard_block(self) -> None:
        paragraph = "word " * 30
        blocks = [
            ContentBlock(raw_text=paragraph, hierarchical_path="Topic"),
            ContentBlock(raw_text="x" * 50, block_type=BlockType.TABLE, hierarchical_path="Topic"),
        ]

        result = DocumentChunker(max_chunk_size=40, overlap=10, trace_id="test").run(self.make_document(blocks))

        self.assertTrue(any(chunk.raw_text == "x" * 50 for chunk in result.block))
        paragraph_chunks = [chunk for chunk in result.block if chunk.raw_text != "x" * 50]
        self.assertGreater(len(paragraph_chunks), 1)
        self.assertTrue(all(len(chunk.raw_text) <= 40 for chunk in paragraph_chunks))


if __name__ == "__main__":
    unittest.main()