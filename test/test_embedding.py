import unittest

from rag.core.schemas.chunking_schemas import Chunking_Block, Chunking_Document
from rag.embedding import Sentence_transformer


class FakeEncoder:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def encode(self, sentences, *, batch_size, normalize_embeddings, convert_to_numpy):
        self.calls.append(
            {
                "sentences": list(sentences),
                "batch_size": batch_size,
                "normalize_embeddings": normalize_embeddings,
                "convert_to_numpy": convert_to_numpy,
            }
        )
        return [
            [0.1] * Sentence_transformer.EMBEDDING_DIMENSION
            for _ in sentences
        ]


class TestSentenceTransformer(unittest.TestCase):
    def test_embeds_contextualized_text_in_batches(self) -> None:
        encoder = FakeEncoder()
        document = Chunking_Document(
            block=[
                Chunking_Block(
                    block_id="chunk-1",
                    raw_text="raw text",
                    contextualized_text="context text",
                    hierachical_path="Topic",
                )
            ],
            block_number=1,
        )

        result = Sentence_transformer(model_instance=encoder, trace_id="test").run(document)

        self.assertEqual(result.number_block, 1)
        self.assertEqual(result.block_embedding[0].chunk_id, "chunk-1")
        self.assertEqual(
            len(result.block_embedding[0].embedding),
            Sentence_transformer.EMBEDDING_DIMENSION,
        )
        self.assertEqual(result.block_embedding[0].payload["text"], "context text")
        self.assertEqual(encoder.calls[0]["sentences"], ["context text"])
        self.assertEqual(encoder.calls[0]["batch_size"], 64)
        self.assertTrue(encoder.calls[0]["normalize_embeddings"])
        self.assertTrue(encoder.calls[0]["convert_to_numpy"])


if __name__ == "__main__":
    unittest.main()
