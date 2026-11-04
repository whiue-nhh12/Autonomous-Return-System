"""Neo4j persistence for policy/entity nodes and graph relationships."""

from __future__ import annotations

from typing import Protocol

from rag.core.schemas.graphrag_schemas import (
    GraphEvidence,
    GraphSyncPayload,
    PolicyRule,
    QueryResult,
    QueryUnderstanding,
)


class Neo4jDriver(Protocol):
    def session(self):
        ...

    def close(self) -> None:
        ...


class Neo4jGraphRepository:
    def __init__(
        self,
        *,
        uri: str,
        username: str,
        password: str,
        driver=None,
    ) -> None:
        if driver is None:
            try:
                from neo4j import GraphDatabase
            except ImportError as exc:
                raise ImportError(
                    "Neo4jGraphRepository requires the neo4j Python driver. "
                    "Install it with: pip install neo4j"
                ) from exc
            driver = GraphDatabase.driver(uri, auth=(username, password))
        self.driver: Neo4jDriver = driver

    def create_schema(self) -> None:
        constraints = (
            "CREATE CONSTRAINT graph_entity_id IF NOT EXISTS "
            "FOR (node:Entity) REQUIRE node.id IS UNIQUE",
            "CREATE CONSTRAINT graph_policy_rule_id IF NOT EXISTS "
            "FOR (node:PolicyRule) REQUIRE node.id IS UNIQUE",
            "CREATE CONSTRAINT graph_chunk_ref_id IF NOT EXISTS "
            "FOR (node:GraphChunkRef) REQUIRE node.chunk_id IS UNIQUE",
        )
        with self.driver.session() as session:
            for statement in constraints:
                session.run(statement).consume()

    def list_policy_rules(self) -> list[PolicyRule]:
        with self.driver.session() as session:
            result = session.run("MATCH (rule:PolicyRule) RETURN properties(rule) AS properties")
            return [PolicyRule.model_validate(record["properties"]) for record in result]

    def query_policy_rules(self, query: QueryUnderstanding, *, limit: int = 20) -> list[QueryResult]:
        """Find policy rules matching the fields explicitly extracted from a query."""
        if limit <= 0:
            raise ValueError("limit must be greater than zero")
        filters = query.model_dump(exclude_none=True)
        if not filters:
            return []
        parameters = {
            "product": query.product,
            "product_category": query.product_category.value if query.product_category else None,
            "item_conditions": [item.value for item in query.item_conditions] if query.item_conditions else None,
            "condition": query.condition,
            "customer_type": query.customer_type,
            "purchase_age_days": query.purchase_age_days,
            "action": query.action,
            "limit": limit,
        }
        with self.driver.session() as session:
            records = session.run(
                """
                MATCH (rule:PolicyRule)
                WITH rule,
                    CASE WHEN $product IS NOT NULL AND toLower(coalesce(rule.product, '')) CONTAINS toLower($product) THEN 1 ELSE 0 END +
                    CASE WHEN $product_category IS NOT NULL AND rule.product_category = $product_category THEN 1 ELSE 0 END +
                    CASE WHEN $item_conditions IS NOT NULL AND any(item_condition IN $item_conditions
                                   WHERE item_condition IN coalesce(rule.item_conditions, [])) THEN 1 ELSE 0 END +
                    CASE WHEN $condition IS NOT NULL AND toLower(coalesce(rule.condition, '')) CONTAINS toLower($condition) THEN 1 ELSE 0 END +
                    CASE WHEN $customer_type IS NOT NULL AND toLower(coalesce(rule.customer_type, '')) CONTAINS toLower($customer_type) THEN 1 ELSE 0 END +
                    CASE WHEN $action IS NOT NULL AND toLower(coalesce(rule.action, '')) CONTAINS toLower($action) THEN 1 ELSE 0 END +
                    CASE WHEN $purchase_age_days IS NOT NULL AND rule.window_max IS NOT NULL
                                   AND (
                                       toLower(coalesce(rule.window_text, '')) CONTAINS 'ngày'
                                       OR toLower(coalesce(rule.window_text, '')) CONTAINS 'calendar day'
                                       OR toLower(coalesce(rule.window_text, '')) CONTAINS 'calendar-day'
                                       OR toLower(coalesce(rule.window_text, '')) CONTAINS ' day'
                                       OR toLower(coalesce(rule.window_text, '')) CONTAINS 'days'
                                   )
                                   AND NOT toLower(coalesce(rule.window_text, '')) CONTAINS 'làm việc'
                                   AND NOT toLower(coalesce(rule.window_text, '')) CONTAINS 'business'
                                   AND NOT toLower(coalesce(rule.window_text, '')) CONTAINS 'working'
                                   AND rule.window_max >= $purchase_age_days THEN 1 ELSE 0 END AS matched
                WITH rule, matched, $requested AS requested
                WHERE matched > 0
                OPTIONAL MATCH (rule)-[support:SUPPORTED_BY]->(chunk:GraphChunkRef)
                RETURN properties(rule) AS rule,
                       collect(DISTINCT chunk.chunk_id) AS chunk_ids,
                       collect(DISTINCT support.evidence_text) AS evidence_text,
                       toFloat(matched) / requested AS score
                ORDER BY score DESC, rule.id
                LIMIT $limit
                """,
                requested=len(filters),
                **parameters,
            )
            results = []
            for record in records:
                rule = PolicyRule.model_validate(record["rule"])
                chunk_ids = [item for item in (record.get("chunk_ids") or []) if item]
                evidence_text = [
                    text
                    for group in (record.get("evidence_text") or [])
                    if isinstance(group, list)
                    for text in group
                    if text
                ]
                results.append(
                    QueryResult(
                        rule_id=rule.id,
                        rule=rule,
                        evidence=GraphEvidence(
                            rule_id=rule.id,
                            source_chunks=chunk_ids,
                            evidence_text=evidence_text,
                        ),
                        score=max(0.0, min(1.0, float(record["score"]))),
                    )
                )
            return results

    def upsert(self, payload: GraphSyncPayload) -> None:
        data = payload.model_dump(mode="json")
        with self.driver.session() as session:
            session.execute_write(self._upsert_transaction, data)

    @staticmethod
    def _upsert_transaction(transaction, data: dict) -> None:
        chunk = data["chunk"]
        transaction.run(
            """
            MERGE (chunk:GraphChunkRef {chunk_id: $chunk_id})
            SET chunk.doc_id = $doc_id,
                chunk.version_number = $version_number,
                chunk.hierarchical_path = $hierarchical_path
            """,
            chunk_id=chunk["chunk_id"],
            doc_id=chunk["doc_id"],
            version_number=chunk["version_number"],
            hierarchical_path=chunk["hierarchical_path"],
        ).consume()

        for entity in data["extraction"]["entities"]:
            transaction.run(
                """
                MATCH (chunk:GraphChunkRef {chunk_id: $chunk_id})
                MERGE (entity:Entity {id: $id})
                SET entity.name = $name,
                    entity.canonical_name = $canonical_name,
                    entity.source_chunks = reduce(
                        accumulated = coalesce(entity.source_chunks, []),
                        source_chunk_id IN $source_chunks |
                        CASE WHEN source_chunk_id IN accumulated
                            THEN accumulated
                            ELSE accumulated + source_chunk_id
                        END
                        )
                MERGE (entity)-[:MENTIONED_IN]->(chunk)
                """,
                **entity,
                chunk_id=chunk["chunk_id"],
            ).consume()

        evidence_by_rule = {item["rule_id"]: item for item in data["evidence"]}
        for rule in data["extraction"]["rules"]:
            properties = {key: value for key, value in rule.items() if key not in ("id", "source_chunks")}
            transaction.run(
                """
                MATCH (chunk:GraphChunkRef {chunk_id: $chunk_id})
                MERGE (rule:PolicyRule {id: $id})
                SET rule += $properties
                REMOVE rule.window_days
                SET rule.source_chunks = reduce(
                    accumulated = coalesce(rule.source_chunks, []),
                    source_chunk_id IN $source_chunks |
                    CASE WHEN source_chunk_id IN accumulated
                        THEN accumulated
                        ELSE accumulated + source_chunk_id
                    END
                )
                MERGE (rule)-[:SUPPORTED_BY]->(chunk)
                """,
                id=rule["id"],
                properties=properties,
                source_chunks=rule["source_chunks"],
                chunk_id=chunk["chunk_id"],
            ).consume()
            evidence = evidence_by_rule.get(rule["id"])
            if evidence is not None and evidence["evidence_text"]:
                transaction.run(
                    """
                    MATCH (rule:PolicyRule {id: $rule_id})
                    MATCH (chunk:GraphChunkRef {chunk_id: $chunk_id})
                    MERGE (rule)-[support:SUPPORTED_BY]->(chunk)
                    SET support.evidence_text = $evidence_text
                    """,
                    rule_id=rule["id"],
                    chunk_id=chunk["chunk_id"],
                    evidence_text=evidence["evidence_text"],
                ).consume()

    def close(self) -> None:
        self.driver.close()
