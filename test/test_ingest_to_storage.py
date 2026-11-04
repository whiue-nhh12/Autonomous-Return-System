import tempfile
import unittest
from pathlib import Path

from rag.core.schemas.chunking_schemas import Chunking_Document
from rag.core.schemas.cleaning_schemas import Cleaned_Document
from rag.core.schemas.embedding_schemas import Embedding_Document
from rag.core.schemas.ingestion_schemas import DocumentMetadata, ParsedDocument
from scripts.ingest_to_storage import (
    _clean_document_content,
    collect_sources,
    process_document_content,
    stable_directory_doc_id,
)


class FakeIngestion:
    def run(self, source: str) -> ParsedDocument:
        self.source = Path(source)
        self.content = self.source.read_bytes()
        return ParsedDocument(
            metadata=DocumentMetadata(doc_id="", source=str(self.source), file_type="md")
        )


class FakeCleaner:
    def run(self, parsed: ParsedDocument) -> Cleaned_Document:
        self.parsed = parsed
        return Cleaned_Document(metadata=parsed.metadata, blocks=parsed.blocks)


class FakeChunker:
    def run(self, cleaned: Cleaned_Document) -> Chunking_Document:
        self.cleaned = cleaned
        return Chunking_Document(metadata=cleaned.metadata)


class FakeEmbedder:
    def run(self, chunked: Chunking_Document) -> Embedding_Document:
        return Embedding_Document(metadata=chunked.metadata)


class FakeClassifier:
    def __init__(self, events):
        self.documents = []
        self.events = events

    def run(self, cleaned: Cleaned_Document) -> Cleaned_Document:
        self.documents.append(cleaned)
        self.events.append("classified")
        return cleaned


class TestIngestToStorage(unittest.TestCase):
    def test_cleans_unchanged_content_for_graph_processing(self):
        ingestion = FakeIngestion()
        cleaner = FakeCleaner()

        cleaned = _clean_document_content(
            b"return and refund policy",
            "doc_42",
            source_name="policy.md",
            file_suffix=".md",
            ingestion=ingestion,
            cleaner=cleaner,
        )

        self.assertEqual(ingestion.content, b"return and refund policy")
        self.assertEqual(cleaned.metadata.doc_id, "doc_42")
        self.assertEqual(cleaned.metadata.source, "policy.md")
        self.assertFalse(ingestion.source.exists())

    def test_collects_supported_files_recursively_and_skips_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "return_refund"
            nested = root / "archive"
            nested.mkdir(parents=True)
            first = root / "policy.md"
            second = nested / "policy.pdf"
            first.write_text("# Policy", encoding="utf-8")
            second.write_bytes(b"%PDF")
            (root / "manifest.json").write_text("[]", encoding="utf-8")
            (root / "notes.txt").write_text("not an ingestion source", encoding="utf-8")

            sources = collect_sources(root, (".md", ".pdf"))

            expected = sorted(
                [first.resolve(), second.resolve()],
                key=lambda path: path.relative_to(root.resolve()).as_posix().casefold(),
            )
            self.assertEqual(sources, expected)

    def test_directory_doc_id_is_stable_and_distinguishes_relative_paths(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "return_refund"
            first = root / "one" / "policy.md"
            second = root / "two" / "policy.md"
            first.parent.mkdir(parents=True)
            second.parent.mkdir()
            first.touch()
            second.touch()

            first_id = stable_directory_doc_id(first, root)

            self.assertEqual(first_id, stable_directory_doc_id(first, root))
            self.assertNotEqual(first_id, stable_directory_doc_id(second, root))
            self.assertRegex(first_id, r"^batch_[0-9a-f]{32}$")

    def test_processes_content_and_preserves_database_identity(self) -> None:
        ingestion = FakeIngestion()
        cleaner = FakeCleaner()
        chunker = FakeChunker()
        embedder = FakeEmbedder()

        result = process_document_content(
            b"return and refund policy",
            "doc_42",
            source_name="policy.md",
            file_suffix=".md",
            ingestion=ingestion,
            cleaner=cleaner,
            chunker=chunker,
            embedder=embedder,
        )

        self.assertEqual(ingestion.content, b"return and refund policy")
        self.assertIsNotNone(result.metadata)
        if result.metadata is None:
            self.fail("Embedding document lost its metadata")
        self.assertEqual(result.metadata.doc_id, "doc_42")
        self.assertEqual(result.metadata.source, "policy.md")
        self.assertFalse(ingestion.source.exists())

    def test_classifies_cleaned_document_before_chunking_and_passes_it_to_graph_stage(self):
        ingestion = FakeIngestion()
        cleaner = FakeCleaner()
        events = []
        classifier = FakeClassifier(events)

        class RecordingChunker(FakeChunker):
            def run(self, cleaned: Cleaned_Document) -> Chunking_Document:
                events.append("chunked")
                return super().run(cleaned)

        chunker = RecordingChunker()
        embedder = FakeEmbedder()
        classified_documents = []

        process_document_content(
            b"return and refund policy",
            "doc_42",
            source_name="policy.md",
            file_suffix=".md",
            ingestion=ingestion,
            cleaner=cleaner,
            chunker=chunker,
            embedder=embedder,
            classifier=classifier,
            on_cleaned=classified_documents.append,
        )

        self.assertEqual(classifier.documents, [chunker.cleaned])
        self.assertEqual(classified_documents, [chunker.cleaned])
        self.assertEqual(events, ["classified", "chunked"])


if __name__ == "__main__":
    unittest.main()
