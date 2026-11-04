import unittest
import importlib.util
from types import SimpleNamespace

from rag.core.schemas.embedding_schemas import Embedding_Block, Embedding_Document
from rag.core.schemas.ingestion_schemas import DocumentMetadata
from rag.search.base_keyword_search import BaseKeywordSearch
from rag.search.elasticsearch_index import ElasticsearchKeywordIndex
from rag.search.vietnamese_tokenizer import VietnameseTokenizer


class FakeIndices:
    def __init__(self):
        self.created = None

    def exists(self, *, index):
        return False

    def create(self, **kwargs):
        self.created = kwargs


class FakeElasticsearchClient:
    def __init__(self):
        self.indices = FakeIndices()
        self.bulk_calls = []
        self.search_calls = []
        self.search_response = {"hits": {"hits": []}}

    def bulk(self, **kwargs):
        self.bulk_calls.append(kwargs)
        return {"errors": False, "items": []}

    def search(self, **kwargs):
        self.search_calls.append(kwargs)
        return self.search_response

    def close(self):
        pass


class TestElasticsearchKeywordIndex(unittest.TestCase):
    def test_elasticsearch_implements_abstract_keyword_contract(self):
        self.assertTrue(issubclass(ElasticsearchKeywordIndex, BaseKeywordSearch))
        with self.assertRaises(TypeError):
            BaseKeywordSearch()

    def test_real_pyvi_segments_vietnamese_compounds(self):
        if importlib.util.find_spec("pyvi") is None:
            self.skipTest("PyVi is not installed")
        self.assertEqual(
            VietnameseTokenizer().tokenize("Thành phố Hồ Chí Minh, Hà Nội là thủ đô."),
            "thành_phố hồ_chí_minh , hà_nội là thủ_đô .",
        )

    def test_mapping_preserves_pyvi_compounds_and_bulk_indexes_chunks(self):
        client = FakeElasticsearchClient()
        tokenizer = VietnameseTokenizer(
            tokenize=lambda text: text.replace("Hà Nội", "Hà_Nội").replace("thủ đô", "thủ_đô")
        )
        index = ElasticsearchKeywordIndex(client=client, tokenizer=tokenizer, index_name="chunks")
        index.create_collection()
        document = Embedding_Document(
            metadata=DocumentMetadata(doc_id="doc_12", source="sample.md", file_type="md"),
            Block_Embedding=[
                Embedding_Block(chunk_id="chunk-1", payload={"text": "Hà Nội là thủ đô."}),
            ],
        )

        self.assertEqual(index.index_document(document, version=3), 1)
        mapping = client.indices.created["mappings"]["properties"]
        self.assertEqual(mapping["text_vi"]["analyzer"], "vietnamese_pyvi")
        body = client.bulk_calls[0]["operations"][1]
        self.assertEqual(body["text_vi"], "hà_nội là thủ_đô.")
        self.assertEqual(body["text_original"], "Hà Nội là thủ đô.")
        self.assertEqual(body["version"], 3)
        self.assertEqual(body["published_version_key"], "doc_12:3")

    def test_indexing_uses_batches_of_64(self):
        client = FakeElasticsearchClient()
        index = ElasticsearchKeywordIndex(
            client=client,
            tokenizer=VietnameseTokenizer(tokenize=lambda text: text),
        )
        document = Embedding_Document(
            metadata=DocumentMetadata(doc_id="doc_12", source="sample.md", file_type="md"),
            Block_Embedding=[
                Embedding_Block(chunk_id=f"chunk-{i}", payload={"text": f"text {i}"})
                for i in range(65)
            ],
        )

        self.assertEqual(index.index_document(document, version=1), 65)
        self.assertEqual([len(call["operations"]) // 2 for call in client.bulk_calls], [64, 1])

    def test_search_pre_filters_published_versions_and_maps_hits(self):
        client = FakeElasticsearchClient()
        client.search_response = {
            "hits": {
                "hits": [
                    {
                        "_score": 4.2,
                        "_source": {
                            "doc_id": "doc_12",
                            "version": 3,
                            "chunk_id": "chunk-1",
                            "chunk_index": 0,
                            "text_original": "Hà Nội là thủ đô.",
                            "payload": {"topic": ["city"]},
                        },
                    }
                ]
            }
        }
        index = ElasticsearchKeywordIndex(
            client=client,
            tokenizer=VietnameseTokenizer(tokenize=lambda text: text.replace("Hà Nội", "Hà_Nội")),
        )

        hits = index.search("Hà Nội", published_versions={"doc_12": 3})

        self.assertEqual(hits[0].version, 3)
        self.assertEqual(hits[0].text, "Hà Nội là thủ đô.")
        query = client.search_calls[0]["query"]
        terms = query["bool"]["filter"][0]["terms"]["published_version_key"]
        self.assertEqual(terms, ["doc_12:3"])


if __name__ == "__main__":
    unittest.main()