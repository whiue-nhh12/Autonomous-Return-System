"""Truy vấn Neo4j và in kết quả ra màn hình."""

from __future__ import annotations

import argparse
import json
from typing import Any

from neo4j.graph import Node, Path, Relationship
from neo4j.time import Date, DateTime, Duration, Time

from rag.core.config import get_settings
from rag.graph.neo4j_repository import Neo4jGraphRepository

PRESET_QUERIES: dict[str, str] = {
    "summary": """
        MATCH (n)
        UNWIND labels(n) AS name
        RETURN 'node' AS kind, name, count(*) AS count
        UNION ALL
        MATCH ()-[r]->()
        RETURN 'relationship' AS kind, type(r) AS name, count(*) AS count
        ORDER BY kind, name
    """,
    "entities": """
        MATCH (entity:Entity)
        OPTIONAL MATCH (entity)-[:MENTIONED_IN]->(chunk:GraphChunkRef)
        RETURN entity.id AS id,
               entity.name AS name,
               entity.canonical_name AS canonical_name,
               entity.source_chunks AS source_chunks,
               collect(DISTINCT chunk.chunk_id) AS mentioned_in
        ORDER BY coalesce(entity.canonical_name, entity.name, entity.id)
    """,
    "rules": """
        MATCH (rule:PolicyRule)
        WHERE $rule_id IS NULL OR rule.id = $rule_id
        OPTIONAL MATCH (rule)-[support:SUPPORTED_BY]->(chunk:GraphChunkRef)
        RETURN rule.id AS id,
               rule.rule_type AS rule_type,
               rule.return_reason AS return_reason,
               rule.product_category AS product_category,
               rule.product AS product,
               rule.item_conditions AS item_conditions,
               rule.condition AS condition,
               rule.customer_type AS customer_type,
               rule.action AS action,
               rule.who_pay_ship AS who_pay_ship,
               rule.window_text AS window_text,
               rule.window_min AS window_min,
               rule.window_max AS window_max,
               rule.requirements AS requirements,
               rule.exclusions AS exclusions,
               rule.source_chunk_id AS source_chunk_id,
               rule.source_chunks AS source_chunks,
               collect(DISTINCT {
                   chunk_id: chunk.chunk_id,
                   doc_id: chunk.doc_id,
                   version_number: chunk.version_number,
                   evidence_text: support.evidence_text
               }) AS supported_by
        ORDER BY rule.id
    """,
    "chunks": """
        MATCH (chunk:GraphChunkRef)
        RETURN chunk.chunk_id AS chunk_id,
               chunk.doc_id AS doc_id,
               chunk.version_number AS version_number,
               chunk.hierarchical_path AS hierarchical_path
        ORDER BY chunk.doc_id, chunk.version_number, chunk.chunk_id
    """,
    "mentions": """
        MATCH (entity:Entity)-[:MENTIONED_IN]->(chunk:GraphChunkRef)
        RETURN entity.id AS entity_id,
               entity.name AS entity_name,
               entity.canonical_name AS canonical_name,
               chunk.chunk_id AS chunk_id,
               chunk.doc_id AS doc_id
        ORDER BY entity.canonical_name, chunk.chunk_id
    """,
    "evidence": """
        MATCH (rule:PolicyRule)-[support:SUPPORTED_BY]->(chunk:GraphChunkRef)
        RETURN rule.id AS rule_id,
               rule.product AS product,
               rule.action AS action,
               chunk.chunk_id AS chunk_id,
               chunk.doc_id AS doc_id,
               support.evidence_text AS evidence_text
        ORDER BY rule.id, chunk.chunk_id
    """,
}


def _to_jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (DateTime, Date, Time, Duration)):
        return str(value)
    if isinstance(value, Node):
        return {
            "element_id": value.element_id,
            "labels": list(value.labels),
            "properties": _to_jsonable(dict(value)),
        }
    if isinstance(value, Relationship):
        return {
            "element_id": value.element_id,
            "type": value.type,
            "properties": _to_jsonable(dict(value)),
            "start_node": value.start_node.element_id,
            "end_node": value.end_node.element_id,
        }
    if isinstance(value, Path):
        return {
            "nodes": [_to_jsonable(node) for node in value.nodes],
            "relationships": [_to_jsonable(rel) for rel in value.relationships],
        }
    if isinstance(value, dict):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_to_jsonable(item) for item in value]
    return str(value)


def _print_records(rows: list[dict[str, Any]]) -> None:
    print(f"Số bản ghi: {len(rows)}")
    if not rows:
        print("(trống)")
        return
    print(json.dumps(rows, ensure_ascii=False, indent=2, default=str))


def _print_policy_rules(rows: list[dict[str, Any]]) -> None:
    print(f"Policy rules: {len(rows)}")
    if not rows:
        print("(no policy rules found)")
        return

    fields = (
        ("rule_type", "Rule type"),
        ("return_reason", "Return reason"),
        ("product_category", "Product category"),
        ("product", "Product"),
        ("item_conditions", "Item conditions"),
        ("condition", "Condition"),
        ("customer_type", "Customer type"),
        ("action", "Action"),
        ("who_pay_ship", "Who pays shipping"),
        ("window_text", "Window text"),
        ("window_min", "Window minimum"),
        ("window_max", "Window maximum"),
        ("requirements", "Requirements"),
        ("exclusions", "Exclusions"),
        ("source_chunk_id", "Source chunk"),
        ("source_chunks", "Source chunks"),
    )
    for index, rule in enumerate(rows, start=1):
        print(f"\n[{index}] PolicyRule {rule.get('id') or '(missing id)'}")
        for key, label in fields:
            value = rule.get(key)
            if isinstance(value, (list, dict)):
                value = json.dumps(value, ensure_ascii=False)
            print(f"  {label}: {value if value not in (None, '') else '-'}")

        sources = [source for source in (rule.get("supported_by") or []) if source.get("chunk_id")]
        if not sources:
            print("  Supporting chunks/evidence: -")
            continue
        print("  Supporting chunks/evidence:")
        for source in sources:
            print(
                "    - "
                f"chunk={source.get('chunk_id')}, doc={source.get('doc_id')}, "
                f"version={source.get('version_number')}"
            )
            if source.get("evidence_text"):
                print(f"      Evidence: {source['evidence_text']}")


def query_neo4j(cypher: str, parameters: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    settings = get_settings().graph_settings
    repository = Neo4jGraphRepository(
        uri=settings.neo4j_uri,
        username=settings.neo4j_username,
        password=settings.neo4j_password.get_secret_value(),
    )
    try:
        with repository.driver.session() as session:
            result = session.run(cypher, parameters or {})
            return [_to_jsonable(record.data()) for record in result]
    finally:
        repository.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Truy vấn Neo4j và in kết quả.")
    parser.add_argument(
        "--preset",
        choices=[*PRESET_QUERIES, "all"],
        default="summary",
        help="Truy vấn có sẵn. Mặc định: summary.",
    )
    parser.add_argument(
        "--cypher",
        help="Câu Cypher tùy chỉnh. Nếu truyền flag này thì --preset bị bỏ qua.",
    )
    parser.add_argument(
        "--rule-id",
        help="Filter the rules preset to one PolicyRule ID.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    settings = get_settings().graph_settings
    print(f"Neo4j: {settings.neo4j_uri} (user={settings.neo4j_username})")

    if args.cypher:
        print("\n=== CUSTOM CYPHER ===")
        _print_records(query_neo4j(args.cypher))
        return 0

    if args.rule_id and args.preset not in ("rules", "all"):
        build_parser().error("--rule-id requires --preset rules (or --preset all)")

    presets = list(PRESET_QUERIES) if args.preset == "all" else [args.preset]
    for name in presets:
        print(f"\n=== {name.upper()} ===")
        parameters = {"rule_id": args.rule_id} if name == "rules" else None
        rows = query_neo4j(PRESET_QUERIES[name], parameters)
        if name == "rules":
            _print_policy_rules(rows)
        else:
            _print_records(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
