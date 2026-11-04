"""Orchestrate classification, extraction, and durable dual-store graph writes."""

from __future__ import annotations

from time import perf_counter
from typing import Protocol
from uuid import uuid4

from rag.classifier.graph_classifier import GraphClassifier
from rag.core.logger.rag.graphrag_logger import GraphRagLogger
from rag.core.llm.text_generation import GeminiTextGenerator, GroqTextGenerator
from rag.core.schemas.cleaning_schemas import Cleaned_Document
from rag.core.schemas.graphrag_schemas import (
    ExtractionResult,
    GraphChunk,
    GraphEvidence,
    ItemCondition,
    GraphProcess,
    GraphProcessStatus,
    GraphRagRunResult,
    GraphSyncPayload,
)
from rag.document_store.graph_store.repository import PostgresGraphChunkRepository
from rag.graph.extractor import GraphExtractor
from rag.graph.normalizer import GraphNormalizer
from rag.graph.neo4j_repository import Neo4jGraphRepository
from rag.graph.rule_linker import RuleLinker


class GraphLogger(Protocol):
    def emit(
        self,
        *,
        trace_id: str,
        process: GraphProcess,
        status: GraphProcessStatus,
        message: str = "",
        metrics: dict | None = None,
    ) -> object:
        ...


def _graph_extraction_response_format() -> dict:
    """Request valid JSON; GraphExtractor validates its structure locally."""
    return {"type": "json_object"}


class GraphRagPipeline:
    def __init__(
        self,
        *,
        classifier: GraphClassifier,
        extractor: GraphExtractor,
        postgres: PostgresGraphChunkRepository,
        neo4j: Neo4jGraphRepository,
        normalizer: GraphNormalizer | None = None,
        linker: RuleLinker | None = None,
        logger: GraphLogger | None = None,
        trace_id: str | None = None,
    ) -> None:
        self.classifier = classifier
        self.extractor = extractor
        self.postgres = postgres
        self.neo4j = neo4j
        self.normalizer = normalizer or GraphNormalizer()
        self.linker = linker or RuleLinker()
        self.logger = logger or GraphRagLogger()
        self.trace_id = trace_id or str(uuid4())

    @classmethod
    def from_settings(
        cls,
        settings,
        *,
        trace_id: str | None = None,
        aliases: dict | None = None,
    ) -> GraphRagPipeline:
        trace_id = trace_id or str(uuid4())
        graph_settings = settings.graph_settings
        gemini_settings = settings.vlm_settings.genai
        classifier_generator = GroqTextGenerator(
            model=graph_settings.classifier_model,
            api_key=settings.groq_api_key.get_secret_value(),
        )
        extractor_generator = GroqTextGenerator(
            model=graph_settings.model,
            api_key=settings.groq_api_key.get_secret_value(),
            timeout_seconds=graph_settings.timeout_seconds,
            response_format=_graph_extraction_response_format(),
        )
        logger = GraphRagLogger()
        return cls(
            classifier=GraphClassifier(
                generator=classifier_generator,
                logger=logger,
                trace_id=trace_id,
                keyword_density_threshold=graph_settings.keyword_density_threshold,
                minimum_keyword_hits=graph_settings.minimum_keyword_hits,
                max_tokens=min(graph_settings.max_output_tokens, 256),
                timeout_seconds=graph_settings.timeout_seconds,
            ),
            extractor=GraphExtractor(
                generator=extractor_generator,
                max_tokens=graph_settings.max_output_tokens,
                timeout_seconds=graph_settings.timeout_seconds,
                aliases=aliases,
                logger=logger,
                trace_id=trace_id,
            ),
            postgres=PostgresGraphChunkRepository(settings.storage_settings.postgres_url_value),
            neo4j=Neo4jGraphRepository(
                uri=graph_settings.neo4j_uri,
                username=graph_settings.neo4j_username,
                password=graph_settings.neo4j_password.get_secret_value(),
            ),
            normalizer=GraphNormalizer(aliases=aliases if aliases is not None else graph_settings.aliases),
            linker=RuleLinker(threshold=graph_settings.rule_link_threshold),
            logger=logger,
            trace_id=trace_id,
        )

    def create_schema(self) -> None:
        self.postgres.create_schema()
        self.neo4j.create_schema()

    def run(self, cleaned: Cleaned_Document, *, version_number: int) -> GraphRagRunResult:
        if version_number <= 0:
            raise ValueError("version_number must be greater than zero")
        if cleaned.metadata is None or not cleaned.metadata.doc_id:
            raise ValueError("Cleaned document must include a doc_id")

        classified = self.classifier.run(cleaned)
        eligible = [block for block in classified.blocks if block.graph_status == "GRAPH_WORTHY"]
        persisted = 0
        entity_count = 0
        rule_count = 0
        for block in eligible:
            current_process: GraphProcess = "EXTRACTING"
            process_started = perf_counter()
            enqueued = False
            try:
                extraction = self.extractor.extract(
                    chunk_id=block.block_id,
                    text=block.contextual_text or block.raw_text,
                )
                self._emit(
                    "EXTRACTING",
                    "Đã trích xuất entity và policy rule.",
                    "SUCCESS",
                    process_started,
                    entity_count=len(extraction.entities),
                    rule_count=len(extraction.rules),
                )

                current_process = "NORMALIZING"
                process_started = perf_counter()
                extraction = self.normalizer.normalize(extraction)
                self._emit(
                    "NORMALIZING",
                    "Đã chuẩn hóa entity và policy rule.",
                    "SUCCESS",
                    process_started,
                    entity_count=len(extraction.entities),
                    rule_count=len(extraction.rules),
                )

                payload = GraphSyncPayload(
                    chunk=GraphChunk(
                        id=block.block_id,
                        chunk_id=block.block_id,
                        doc_id=cleaned.metadata.doc_id,
                        version_number=version_number,
                        raw_text=block.raw_text,
                        contextual_text=block.contextual_text,
                        hierarchical_path=block.hierarchical_path,
                    ),
                    extraction=extraction,
                )

                current_process = "PERSISTING_POSTGRES"
                process_started = perf_counter()
                self.postgres.enqueue(payload)
                enqueued = True
                self._emit(
                    "PERSISTING_POSTGRES",
                    "Đã lưu graph chunk và outbox đồng bộ.",
                    "SUCCESS",
                    process_started,
                    chunk_count=1,
                )

                current_process = "LINKING"
                process_started = perf_counter()
                payload = self._link_payload(payload)
                self._emit(
                    "LINKING",
                    "Đã liên kết policy rule với graph hiện có.",
                    "SUCCESS",
                    process_started,
                    rule_count=len(payload.extraction.rules),
                )

                current_process = "PERSISTING_POSTGRES"
                process_started = perf_counter()
                self.postgres.enqueue(payload)
                self._emit(
                    "PERSISTING_POSTGRES",
                    "Đã cập nhật outbox bằng rule đã liên kết.",
                    "SUCCESS",
                    process_started,
                    chunk_count=1,
                )

                current_process = "PERSISTING_NEO4J"
                process_started = perf_counter()
                self.neo4j.upsert(payload)
                self.postgres.mark_synced(block.block_id)
                self._emit(
                    "PERSISTING_NEO4J",
                    "Đã đồng bộ graph vào Neo4j.",
                    "SUCCESS",
                    process_started,
                    entity_count=len(extraction.entities),
                    rule_count=len(extraction.rules),
                )
                persisted += 1
                entity_count += len(extraction.entities)
                rule_count += len(extraction.rules)
            except Exception as exc:
                block.graph_status = "FAILED"
                if enqueued:
                    self._mark_sync_failed(block.block_id, exc)
                self._emit_failure(
                    current_process,
                    "Graph chunk processing failed.",
                    process_started,
                    exc,
                )
                raise

        return GraphRagRunResult(
            doc_id=cleaned.metadata.doc_id,
            version_number=version_number,
            classified_block_count=len(classified.blocks),
            graph_worthy_block_count=len(eligible),
            persisted_chunk_count=persisted,
            entity_count=entity_count,
            rule_count=rule_count,
        )

    def flush_pending(self, *, limit: int = 100) -> int:
        pending = self.postgres.get_pending(limit=limit)
        flushed = 0
        for payload in pending:
            started = perf_counter()
            current_process: GraphProcess = "LINKING"
            try:
                payload = self._link_payload(payload)
                current_process = "PERSISTING_POSTGRES"
                self.postgres.enqueue(payload)
                current_process = "PERSISTING_NEO4J"
                self.neo4j.upsert(payload)
                self.postgres.mark_synced(payload.chunk.chunk_id)
            except Exception as exc:
                self._mark_sync_failed(payload.chunk.chunk_id, exc)
                self._emit_failure(
                    current_process,
                    "Retry failed; outbox entry remains available for another attempt.",
                    started,
                    exc,
                )
                raise
            self._emit(
                "PERSISTING_NEO4J",
                "Đã retry đồng bộ graph vào Neo4j.",
                "SUCCESS",
                started,
                chunk_count=1,
            )
            flushed += 1
        return flushed

    def close(self) -> None:
        try:
            self.neo4j.close()
        finally:
            self.postgres.engine.dispose()

    def _link_rules(self, extraction: ExtractionResult) -> ExtractionResult:
        if not extraction.rules:
            return extraction
        rules = list(extraction.rules)
        candidates = self.neo4j.list_policy_rules()
        candidate_by_id = {candidate.id: candidate for candidate in candidates}
        linked = []
        for rule in rules:
            result = self.linker.link(rule, list(candidate_by_id.values()))
            if result.decision == "MERGE" and result.candidate_rule_id is not None:
                existing = candidate_by_id[result.candidate_rule_id]
                rule = self.linker.merge(existing, rule)
                rule = rule.model_copy(update={"id": existing.id})
            candidate_by_id[rule.id] = rule
            linked.append(rule)
        return extraction.model_copy(update={"rules": linked})

    def _link_payload(self, payload: GraphSyncPayload) -> GraphSyncPayload:
        original_rules = payload.extraction.rules
        linked_extraction = self._link_rules(payload.extraction)
        previous_evidence = {item.rule_id: item for item in payload.evidence}
        evidence: list[GraphEvidence] = []
        for original_rule, linked_rule in zip(original_rules, linked_extraction.rules):
            prior = previous_evidence.get(original_rule.id)
            evidence.append(
                GraphEvidence(
                    rule_id=linked_rule.id,
                    source_chunks=list(
                        dict.fromkeys([*(prior.source_chunks if prior else []), payload.chunk.chunk_id])
                    ),
                    evidence_text=prior.evidence_text if prior else [],
                )
            )
        return payload.model_copy(update={"extraction": linked_extraction, "evidence": evidence})

    def _mark_sync_failed(self, chunk_id: str, original_error: Exception) -> None:
        try:
            self.postgres.mark_failed(chunk_id, str(original_error))
        except Exception as status_error:
            raise RuntimeError(
                f"Graph sync failed for chunk {chunk_id!r}, and PostgreSQL could not record the failure"
            ) from status_error

    def _emit(
        self,
        process: GraphProcess,
        message: str,
        status: GraphProcessStatus,
        started: float,
        **metrics: int,
    ) -> None:
        self.logger.emit(
            trace_id=self.trace_id,
            process=process,
            status=status,
            message=message,
            metrics={"elapsed_ms": round((perf_counter() - started) * 1000, 2), **metrics},
        )

    def _emit_failure(
        self,
        process: GraphProcess,
        message: str,
        started: float,
        exc: Exception,
    ) -> None:
        self.logger.emit(
            trace_id=self.trace_id,
            process=process,
            status="FAIL",
            message=f"{message} {type(exc).__name__}: {exc}",
            metrics={"elapsed_ms": round((perf_counter() - started) * 1000, 2)},
        )
