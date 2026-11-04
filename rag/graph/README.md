# GraphRAG ingestion

GraphRAG is an opt-in stage that consumes a `Cleaned_Document` and a document version:

```python
from rag.graph import GraphRagPipeline
from rag.core.config import get_settings

graph = GraphRagPipeline.from_settings(get_settings(), trace_id="ingestion-trace")
graph.create_schema()
try:
    result = graph.run(cleaned_document, version_number=version_number)
finally:
    graph.close()
```

Install the optional Neo4j driver with `pip install neo4j`. Configure the existing PostgreSQL URL,
`GROQ_API_KEY`, and `GRAPH_SETTINGS__NEO4J_URI`, `GRAPH_SETTINGS__NEO4J_USERNAME`, and
`GRAPH_SETTINGS__NEO4J_PASSWORD` in the environment or `.env`. Groq classifies blocks for
graph-worthiness and extracts graph entities and rules. Override the classifier and extraction
models with `GRAPH_SETTINGS__CLASSIFIER_MODEL` and `GRAPH_SETTINGS__MODEL` respectively, if needed.
Extraction requests JSON Object Mode; the response is validated locally against the extraction
schema. Optional aliases can be supplied as a JSON object in `GRAPH_SETTINGS__ALIASES`.
Policy-rule time limits are stored as `window_text` (the source wording), `window_min`, and
`window_max`. Numeric bounds use the unit preserved in `window_text`; for example, a 1–3 business-day
range is stored as bounds 1 and 3 with its original wording. Missing or non-numeric limits keep the
wording and leave numeric bounds null. Age-based graph matching only compares limits explicitly
stated in calendar days, avoiding comparisons between calendar days, business days, and hours.

`ContentBlock.graph_status` defaults to `PENDING`; classification sets it to `NORMAL`,
`GRAPH_WORTHY`, or `FAILED`. Blocks that need LLM classification are sent in batches of at most
five, with one indexed decision required for each block. Graph-worthy cleaned blocks are saved as source chunks in PostgreSQL.
Neo4j stores entity/policy nodes and relationships to lightweight chunk references, not chunk text.
PostgreSQL keeps the full raw/contextual text and an outbox payload for idempotent Neo4j retries;
call `flush_pending()` to retry failed synchronization. Use a `chunk_id` from Neo4j to load full text
with `PostgresGraphChunkRepository.get_chunk()`.

The local `scripts/ingest_to_storage.py` runner classifies cleaned blocks before publishing and sends
graph-worthy content through this pipeline after the document version is published. Classification
failures stop the document update; graph extraction/sync failures are surfaced by the runner. The
general `IngestionPipeline` still returns parsed documents and does not own cleaning,
document-version publication, or graph-store lifecycle.
