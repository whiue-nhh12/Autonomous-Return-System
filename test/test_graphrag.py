import json
import re
import unittest
from unittest.mock import patch

from sqlalchemy import create_engine, select

from rag.classifier.graph_classifier import GraphClassifier
from rag.core.config import GraphSettings, IngestionSettings, StorageSettings
from rag.core.schemas.cleaning_schemas import Cleaned_Document
from rag.core.schemas.graphrag_schemas import (
    Entity,
    ExtractionResult,
    GraphChunk,
    GraphSyncPayload,
    PolicyRule,
)
from rag.core.schemas.ingestion_schemas import ContentBlock, DocumentMetadata
from rag.document_store.graph_store.models import GraphBase, GraphChunkRow
from rag.document_store.graph_store.repository import PostgresGraphChunkRepository
from rag.graph.extractor import GraphExtractor
from rag.graph.normalizer import GraphNormalizer
from rag.graph.neo4j_repository import Neo4jGraphRepository
from rag.graph.pipeline import GraphRagPipeline, _graph_extraction_response_format
from rag.graph.rule_linker import RuleLinker


class FakeGenerator:
    def __init__(self, response=""):
        self.response = response
        self.calls = 0

    def generate(self, prompt, *, max_tokens, timeout_seconds):
        self.calls += 1
        return self.response


class FakeLogger:
    def __init__(self):
        self.records = []

    def emit(self, **kwargs):
        self.records.append(kwargs)
        return kwargs


class FakeExtractor:
    def extract(self, *, chunk_id, text):
        return ExtractionResult(
            chunk_id=chunk_id,
            entities=[
                Entity(
                    id="",
                    name="Premium Card",
                    canonical_name="Premium Card",
                    source_chunk_id=chunk_id,
                    source_chunks=[chunk_id],
                )
            ],
            rules=[
                PolicyRule(
                    id="",
                    product="Premium Card",
                    condition="Customer pays in full",
                    action="Eligible for refund",
                    source_chunk_id=chunk_id,
                    source_chunks=[chunk_id],
                )
            ],
        )


class FakeNeo4j:
    def __init__(self, *, fail_once=False, fail_list_once=False):
        self.rules = []
        self.payloads = []
        self.fail_once = fail_once
        self.fail_list_once = fail_list_once

    def list_policy_rules(self):
        if self.fail_list_once:
            self.fail_list_once = False
            raise RuntimeError("temporary Neo4j read outage")
        return list(self.rules)

    def upsert(self, payload):
        if self.fail_once:
            self.fail_once = False
            raise RuntimeError("temporary Neo4j outage")
        self.payloads.append(payload)
        self.rules.extend(payload.extraction.rules)

    def create_schema(self):
        pass

    def close(self):
        pass


class FakeNeo4jResult:
    def consume(self):
        return None


class FakeNeo4jSession:
    def __init__(self):
        self.queries = []

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def run(self, query, **parameters):
        self.queries.append((query, parameters))
        return FakeNeo4jResult()

    def execute_write(self, callback, data):
        return callback(self, data)


class FakeNeo4jDriver:
    def __init__(self):
        self.session_instance = FakeNeo4jSession()

    def session(self):
        return self.session_instance

    def close(self):
        pass


class TestGraphRag(unittest.TestCase):
    def test_pipeline_uses_groq_for_extraction_and_classification(self):
        settings = IngestionSettings(
            groq_api_key="groq-key",
            graph_settings=GraphSettings(),
            storage_settings=StorageSettings(
                postgres_url="postgresql://user:pass@localhost/test",
            ),
        )
        with (
            patch("rag.graph.pipeline.GroqTextGenerator") as groq_generator,
            patch("rag.graph.pipeline.PostgresGraphChunkRepository"),
            patch("rag.graph.pipeline.Neo4jGraphRepository"),
        ):
            GraphRagPipeline.from_settings(settings)

        self.assertEqual(groq_generator.call_count, 2)
        groq_generator.assert_any_call(
            model=settings.graph_settings.classifier_model,
            api_key="groq-key",
        )
        groq_generator.assert_any_call(
            model=settings.graph_settings.model,
            api_key="groq-key",
            timeout_seconds=settings.graph_settings.timeout_seconds,
            response_format={"type": "json_object"},
        )

    def test_graph_extraction_uses_json_object_response_format(self):
        self.assertEqual(
            _graph_extraction_response_format(),
            {"type": "json_object"},
        )

    def test_extractor_logs_success_metrics(self):
        logger = FakeLogger()
        extractor = GraphExtractor(
            generator=FakeGenerator(
                '{"chunk_id": "chunk-1", "entities": [], "rules": []}'
            ),
            logger=logger,
            trace_id="trace-1",
        )

        result = extractor.extract(chunk_id="chunk-1", text="policy text")

        self.assertEqual(result.chunk_id, "chunk-1")
        self.assertEqual(logger.records[-1]["trace_id"], "trace-1")
        self.assertEqual(logger.records[-1]["status"], "SUCCESS")
        self.assertEqual(logger.records[-1]["metrics"]["entity_count"], 0)
        self.assertEqual(logger.records[-1]["metrics"]["rule_count"], 0)

    def test_extractor_validates_text_and_range_window_fields(self):
        extractor = GraphExtractor(
            generator=FakeGenerator(
                json.dumps(
                    {
                        "chunk_id": "chunk-1",
                        "entities": [],
                        "rules": [
                            {
                                "id": "",
                                "window_text": "≤ 24 giờ",
                                "window_min": None,
                                "window_max": 24,
                            },
                            {
                                "id": "",
                                "window_text": "1-3 ngày làm việc",
                                "window_min": 1,
                                "window_max": 3,
                            },
                            {
                                "id": "",
                                "window_text": "Không cam kết",
                                "window_min": None,
                                "window_max": None,
                            },
                        ],
                    },
                    ensure_ascii=False,
                )
            ),
            logger=FakeLogger(),
        )

        result = extractor.extract(chunk_id="chunk-1", text="policy text")

        self.assertEqual(
            [(rule.window_text, rule.window_min, rule.window_max) for rule in result.rules],
            [
                ("≤ 24 giờ", None, 24),
                ("1-3 ngày làm việc", 1, 3),
                ("Không cam kết", None, None),
            ],
        )

    def test_policy_rule_migrates_legacy_window_days_values(self):
        numeric = PolicyRule.model_validate({"id": "numeric", "window_days": 5})
        descriptive = PolicyRule.model_validate(
            {"id": "descriptive", "window_days": "1-3 ngày làm việc"}
        )

        self.assertEqual((numeric.window_text, numeric.window_min, numeric.window_max), ("5", 5, 5))
        self.assertEqual(
            (descriptive.window_text, descriptive.window_min, descriptive.window_max),
            ("1-3 ngày làm việc", None, None),
        )

    def test_policy_rule_rejects_inverted_window_bounds(self):
        with self.assertRaisesRegex(ValueError, "window_min must be less than or equal to window_max"):
            PolicyRule.model_validate(
                {"id": "invalid", "window_text": "3-1 ngày", "window_min": 3, "window_max": 1}
            )

    def test_extractor_logs_stage_when_response_json_is_invalid(self):
        logger = FakeLogger()
        extractor = GraphExtractor(
            generator=FakeGenerator("not valid json"),
            logger=logger,
            trace_id="trace-1",
        )

        with self.assertRaises(json.JSONDecodeError):
            extractor.extract(chunk_id="chunk-1", text="policy text")

        self.assertEqual(logger.records[-1]["status"], "FAIL")
        self.assertIn("parsing_json", logger.records[-1]["message"])
        self.assertEqual(logger.records[-1]["metrics"]["chunk_id"], "chunk-1")

    def test_extractor_still_validates_json_payload_against_schema(self):
        extractor = GraphExtractor(
            generator=FakeGenerator(
                '{"chunk_id": "chunk-1", "entities": [], '
                '"rules": [{"rule_type": "not_a_rule"}]}'
            ),
            logger=FakeLogger(),
        )

        with self.assertRaises(ValueError):
            extractor.extract(chunk_id="chunk-1", text="policy text")

    def test_content_block_graph_status_defaults_to_pending(self):
        block = ContentBlock(raw_text="policy")

        self.assertEqual(block.graph_status, "PENDING")

    def test_classifier_uses_lexical_gate_before_llm(self):
        generator = FakeGenerator(
            '{"results": [{"index": 1, "graph_worthy": false}]}'
        )
        document = Cleaned_Document(
            metadata=DocumentMetadata(doc_id="doc_1", source="a.md", file_type="md"),
            blocks=[
                ContentBlock(raw_text="Trong trường hợp khách hàng đủ điều kiện, khoản hoàn tiền được áp dụng."),
                ContentBlock(raw_text="Giới thiệu chung về chương trình khách hàng."),
            ],
        )

        classified = GraphClassifier(generator=generator, logger=FakeLogger()).run(document)

        self.assertEqual(classified.blocks[0].graph_status, "GRAPH_WORTHY")
        self.assertEqual(classified.blocks[1].graph_status, "NORMAL")
        self.assertEqual(generator.calls, 1)

    def test_classifier_sends_llm_candidates_in_batches_of_five(self):
        class BatchGenerator:
            def __init__(self):
                self.prompts = []
                self.batch_sizes = []

            def generate(self, prompt, *, max_tokens, timeout_seconds):
                self.prompts.append(prompt)
                indices = [
                    int(index)
                    for index in re.findall(r"(?m)^ITEM (\d+):$", prompt)
                ]
                self.batch_sizes.append(len(indices))
                results = [
                    {"index": index, "graph_worthy": index % 2 == 0}
                    for index in indices
                ]
                return json.dumps({"results": results})

        generator = BatchGenerator()
        blocks = [
            ContentBlock(raw_text=f"Mô tả tổng quan sản phẩm mã {index}.")
            for index in range(11)
        ]
        document = Cleaned_Document(
            metadata=DocumentMetadata(doc_id="doc_1", source="a.md", file_type="md"),
            blocks=blocks,
        )

        GraphClassifier(generator=generator, logger=FakeLogger()).run(document)

        self.assertEqual(generator.batch_sizes, [5, 5, 1])
        self.assertEqual(
            [block.graph_status for block in blocks],
            [
                "NORMAL",
                "GRAPH_WORTHY",
                "NORMAL",
                "GRAPH_WORTHY",
                "NORMAL",
                "NORMAL",
                "GRAPH_WORTHY",
                "NORMAL",
                "GRAPH_WORTHY",
                "NORMAL",
                "NORMAL",
            ],
        )

    def test_classifier_marks_failed_when_llm_response_is_invalid(self):
        document = Cleaned_Document(
            metadata=DocumentMetadata(doc_id="doc_1", source="a.md", file_type="md"),
            blocks=[ContentBlock(raw_text="Chương trình cung cấp nhiều quyền lợi khách hàng.")],
        )

        with self.assertRaises(ValueError):
            GraphClassifier(generator=FakeGenerator("not json"), logger=FakeLogger()).run(document)

        self.assertEqual(document.blocks[0].graph_status, "FAILED")

    def test_normalizer_applies_alias_and_stable_ids(self):
        chunk_id = "chunk-1"
        extraction = ExtractionResult(
            chunk_id=chunk_id,
            entities=[
                Entity(
                    id="",
                    name="Premium Card",
                    canonical_name="Premium Card",
                    source_chunk_id=chunk_id,
                )
            ],
            rules=[
                PolicyRule(
                    id="",
                    product="Card Plus",
                    condition="pays in full",
                    window_text="1-3 ngày làm việc",
                    window_min=1,
                    window_max=3,
                    source_chunk_id=chunk_id,
                )
            ],
        )
        normalizer = GraphNormalizer(aliases={"Card Plus": "Premium Card"})

        first = normalizer.normalize(extraction)
        second = normalizer.normalize(extraction)

        self.assertEqual(first.entities[0].canonical_name, "Premium Card")
        self.assertEqual(first.entities[0].id, second.entities[0].id)
        self.assertEqual(first.rules[0].product, "Premium Card")
        self.assertEqual(
            (first.rules[0].window_text, first.rules[0].window_min, first.rules[0].window_max),
            ("1-3 ngày làm việc", 1, 3),
        )
        self.assertEqual(first.rules[0].source_chunks, [chunk_id])

    def test_rule_linker_does_not_merge_conflicting_actions(self):
        candidate = PolicyRule(
            id="rule-1",
            condition="Customer pays in full",
            action="Refund",
        )
        incoming = PolicyRule(
            id="rule-2",
            condition="Customer pays in full",
            action="Reject refund",
        )

        result = RuleLinker(threshold=0.8).link(incoming, [candidate])

        self.assertEqual(result.decision, "NEW")

    def test_rule_linker_reuses_deterministic_id_without_condition(self):
        rule = PolicyRule(id="stable-rule")

        result = RuleLinker().link(rule, [rule])

        self.assertEqual(result.decision, "MERGE")
        self.assertEqual(result.candidate_rule_id, "stable-rule")

    def test_neo4j_repository_writes_reference_relationships_without_chunk_text(self):
        chunk_id = "chunk-1"
        extraction = GraphNormalizer().normalize(
            ExtractionResult(
                chunk_id=chunk_id,
                entities=[
                    Entity(
                        id="",
                        name="Premium Card",
                        canonical_name="Premium Card",
                        source_chunk_id=chunk_id,
                    )
                ],
                rules=[
                    PolicyRule(
                        id="",
                        product="Premium Card",
                        condition="Customer pays in full",
                        window_text="≤ 24 giờ",
                        window_min=None,
                        window_max=24,
                        source_chunk_id=chunk_id,
                    )
                ],
            )
        )
        payload = GraphSyncPayload(
            chunk=GraphChunk(
                chunk_id=chunk_id,
                doc_id="doc-1",
                version_number=1,
                raw_text="full private source text",
                contextual_text="heading\nfull private source text",
                hierarchical_path="heading",
            ),
            extraction=extraction,
        )
        driver = FakeNeo4jDriver()
        repository = Neo4jGraphRepository(
            uri="bolt://localhost:7687",
            username="neo4j",
            password="test",
            driver=driver,
        )

        repository.create_schema()
        repository.upsert(payload)

        statements = [query for query, _ in driver.session_instance.queries]
        parameters = [params for _, params in driver.session_instance.queries]
        self.assertTrue(any("MENTIONED_IN" in query for query in statements))
        self.assertTrue(any("SUPPORTED_BY" in query for query in statements))
        self.assertNotIn(
            "full private source text",
            repr(parameters),
        )
        rule_parameters = next(
            params for query, params in driver.session_instance.queries if "MERGE (rule:PolicyRule" in query
        )
        self.assertEqual(
            (
                rule_parameters["properties"]["window_text"],
                rule_parameters["properties"]["window_min"],
                rule_parameters["properties"]["window_max"],
            ),
            ("≤ 24 giờ", None, 24),
        )

    def test_postgres_outbox_keeps_chunk_text_in_columns_and_restores_retry_payload(self):
        engine = create_engine("sqlite:///:memory:")
        GraphBase.metadata.create_all(engine)
        repository = PostgresGraphChunkRepository("sqlite:///:memory:", engine=engine)
        payload = GraphSyncPayload(
            chunk=GraphChunk(
                id="record-1",
                chunk_id="chunk-1",
                doc_id="doc-1",
                version_number=1,
                raw_text="full source text",
                contextual_text="heading\nfull source text",
                hierarchical_path="heading",
            ),
            extraction=ExtractionResult(chunk_id="chunk-1"),
        )

        repository.enqueue(payload)
        pending = repository.get_pending()
        with repository.session_factory() as session:
            row = session.scalar(select(GraphChunkRow).where(GraphChunkRow.chunk_id == "chunk-1"))

        self.assertEqual(row.raw_text, "full source text")
        self.assertEqual(row.sync_payload["chunk"]["raw_text"], "")
        self.assertEqual(pending[0].chunk.raw_text, "full source text")
        self.assertEqual(repository.get_chunk("chunk-1").id, "record-1")
        engine.dispose()

    def test_pipeline_retries_failed_neo4j_sync_without_losing_source_text(self):
        engine = create_engine("sqlite:///:memory:")
        postgres = PostgresGraphChunkRepository("sqlite:///:memory:", engine=engine)
        postgres.create_schema()
        neo4j = FakeNeo4j(fail_once=True)
        logger = FakeLogger()
        text = "Trong trường hợp khách hàng đủ điều kiện, khoản hoàn tiền được áp dụng."
        document = Cleaned_Document(
            metadata=DocumentMetadata(doc_id="doc-1", source="a.md", file_type="md"),
            blocks=[ContentBlock(raw_text=text, contextual_text=text)],
        )
        pipeline = GraphRagPipeline(
            classifier=GraphClassifier(generator=FakeGenerator(), logger=logger),
            extractor=FakeExtractor(),
            postgres=postgres,
            neo4j=neo4j,
            logger=logger,
            trace_id="trace-1",
        )

        with self.assertRaisesRegex(RuntimeError, "temporary Neo4j outage"):
            pipeline.run(document, version_number=1)

        self.assertEqual(document.blocks[0].graph_status, "FAILED")
        with postgres.session_factory() as session:
            row = session.scalar(select(GraphChunkRow).where(GraphChunkRow.chunk_id == document.blocks[0].block_id))
        self.assertEqual(row.sync_status, "FAILED")

        self.assertEqual(pipeline.flush_pending(), 1)
        self.assertEqual(postgres.get_chunk(document.blocks[0].block_id).raw_text, text)
        with postgres.session_factory() as session:
            row = session.scalar(select(GraphChunkRow).where(GraphChunkRow.chunk_id == document.blocks[0].block_id))
        self.assertEqual(row.sync_status, "SYNCED")
        engine.dispose()

    def test_pipeline_outboxes_chunk_before_rule_candidate_lookup(self):
        engine = create_engine("sqlite:///:memory:")
        postgres = PostgresGraphChunkRepository("sqlite:///:memory:", engine=engine)
        postgres.create_schema()
        neo4j = FakeNeo4j(fail_list_once=True)
        text = "Trong trường hợp khách hàng đủ điều kiện, khoản hoàn tiền được áp dụng."
        document = Cleaned_Document(
            metadata=DocumentMetadata(doc_id="doc-1", source="a.md", file_type="md"),
            blocks=[ContentBlock(raw_text=text, contextual_text=text)],
        )
        pipeline = GraphRagPipeline(
            classifier=GraphClassifier(generator=FakeGenerator(), logger=FakeLogger()),
            extractor=FakeExtractor(),
            postgres=postgres,
            neo4j=neo4j,
            logger=FakeLogger(),
        )

        with self.assertRaisesRegex(RuntimeError, "temporary Neo4j read outage"):
            pipeline.run(document, version_number=1)
        self.assertEqual(len(postgres.get_pending()), 1)

        self.assertEqual(pipeline.flush_pending(), 1)
        self.assertEqual(len(neo4j.payloads), 1)
        engine.dispose()


if __name__ == "__main__":
    unittest.main()
