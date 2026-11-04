import unittest
import importlib.util
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

from rag.core.schemas.embedding_schemas import Embedding_Block, Embedding_Document
from rag.core.schemas.ingestion_schemas import DocumentMetadata
from rag.embedding.embedding import Sentence_transformer
from rag.vector_db.qdrant_db import QdrantDB


class FakeClient:
    def __init__(self) -> None:
        self.uploads = []
        self.query = None

    def upload_points(self, *, collection_name, points):
        self.uploads.append((collection_name, points))

    def close(self):
        pass

    def query_points(self, **kwargs):
        self.query = kwargs
        return SimpleNamespace(
            points=[
                SimpleNamespace(
                    id="point-1",
                    score=0.85,
                    payload={
                        "doc_id": "doc-1",
                        "version": 2,
                        "chunk_id": "chunk-1",
                        "chunk_index": 0,
                        "text": "vector result",
                    },
                )
            ]
        )


class TestQdrantDB(unittest.TestCase):
    def test_loads_points_in_batches_and_uses_stable_uuid(self) -> None:
        client = FakeClient()
        database = QdrantDB(client=client, trace_id="test")
        document = Embedding_Document(
            metadata=DocumentMetadata(doc_id="doc-1", source="sample.md", file_type="md"),
            Block_Embedding=[
                Embedding_Block(
                    chunk_id=f"chunk-{index}",
                    embedding=[0.1] * Sentence_transformer.EMBEDDING_DIMENSION,
                )
                for index in range(65)
            ],
        )

        self.assertEqual(database.load(document, version=1), 65)
        self.assertEqual([len(points) for _, points in client.uploads], [64, 1])
        first = client.uploads[0][1][0]
        self.assertEqual(first.payload["doc_id"], "doc-1")
        self.assertEqual(first.payload["version"], 1)
        self.assertEqual(first.id, database.point_id(doc_id="doc-1", chunk_index=0, version=1))

    def test_search_filters_to_published_doc_version(self) -> None:
        class Model:
            def __init__(self, **kwargs):
                self.__dict__.update(kwargs)

        models = SimpleNamespace(
            Filter=Model,
            FieldCondition=Model,
            MatchValue=Model,
        )
        client_package = ModuleType("qdrant_client")
        http_package = ModuleType("qdrant_client.http")
        http_package.models = models
        client_package.http = http_package

        client = FakeClient()
        database = QdrantDB(client=client, trace_id="test")
        with patch.dict(
            sys.modules,
            {"qdrant_client": client_package, "qdrant_client.http": http_package},
        ):
            hits = database.search(
                [0.1] * QdrantDB.VECTOR_SIZE,
                published_versions={"doc-1": 2},
                limit=5,
            )

        self.assertEqual(hits[0].doc_id, "doc-1")
        self.assertEqual(hits[0].version, 2)
        self.assertEqual(client.query["query_filter"].should[0].must[1].match.value, 2)

    @unittest.skipUnless(importlib.util.find_spec("qdrant_client"), "qdrant-client is not installed")
    def test_search_filter_uses_qdrant_sdk_models(self) -> None:
        from qdrant_client.http import models

        client = FakeClient()
        database = QdrantDB(client=client, trace_id="test")
        database.search(
            [0.1] * QdrantDB.VECTOR_SIZE,
            published_versions={"doc-1": 2},
            limit=5,
        )

        self.assertIsInstance(client.query["query_filter"], models.Filter)
        condition = client.query["query_filter"].should[0].must[1]
        self.assertIsInstance(condition, models.FieldCondition)
        self.assertEqual(condition.match.value, 2)

    @unittest.skipUnless(importlib.util.find_spec("qdrant_client"), "qdrant-client is not installed")
    def test_existing_collection_with_old_dimension_fails_before_ingestion(self) -> None:
        client = FakeClient()
        client.collection_exists = lambda _name: True
        client.get_collection = lambda _name: SimpleNamespace(
            config=SimpleNamespace(params=SimpleNamespace(vectors=SimpleNamespace(size=768)))
        )
        database = QdrantDB(client=client, trace_id="test")

        with self.assertRaisesRegex(ValueError, "uses 768-dimensional vectors"):
            database.create_collection()


if __name__ == "__main__":
    unittest.main()