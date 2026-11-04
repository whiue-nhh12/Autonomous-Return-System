"""Schemas shared by graph classification, extraction, and persistence."""

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Entity(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    name: str
    canonical_name: str
    source_chunk_id: str
    source_chunks: list[str] = Field(default_factory=list)


class PolicyRuleType(str, Enum):
    REFUND = "refund"
    SHIPPING = "shipping"
    RETURN = "return"
    EXCHANGE = "exchange"
    CANCELLATION = "cancellation"
    OTHER = "other"


class ShippingFeePayer(str, Enum):
    SHOP = "shop"
    CUSTOMER = "customer"


class ReturnReason(str, Enum):
    CHANGE_OF_MIND = "CHANGE_OF_MIND"
    DEFECTIVE_PRODUCT = "DEFECTIVE_PRODUCT"
    WRONG_ITEM = "WRONG_ITEM"
    DAMAGED_ITEM = "DAMAGED_ITEM"
    MISSING_ITEM = "MISSING_ITEM"
    NOT_RECEIVED = "NOT_RECEIVED"


class ProductCategory(str, Enum):
    GENERAL_MERCHANDISE = "general_merchandise"
    ELECTRONICS = "electronics"
    CLOTHING = "clothing"
    COSMETICS = "cosmetics"
    FOOD_AND_BEVERAGE = "food_and_beverage"
    OTHER = "other"


class ItemCondition(str, Enum):
    UNUSED = "UNUSED"
    INTACT = "INTACT"
    ORIGINAL_PACKAGING = "ORIGINAL_PACKAGING"
    TAGS_ATTACHED = "TAGS_ATTACHED"
    SEALED = "SEALED"


class PolicyRule(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    rule_type: PolicyRuleType = PolicyRuleType.OTHER
    return_reason: ReturnReason | None = None
    product_category: ProductCategory | None = None
    product: str | None = None
    item_conditions: list[ItemCondition] = Field(default_factory=list)
    condition: str | None = None
    customer_type: str | None = None
    action: str | None = None
    who_pay_ship: ShippingFeePayer | None = None
    window_text: str | None = None
    window_min: int | None = Field(default=None, ge=0)
    window_max: int | None = Field(default=None, ge=0)
    requirements: list[str] = Field(default_factory=list)
    exclusions: list[str] = Field(default_factory=list)
    source_chunks: list[str] = Field(default_factory=list)
    source_chunk_id: str | None = None

    @model_validator(mode="before")
    @classmethod
    def migrate_legacy_window_days(cls, value: Any) -> Any:
        if not isinstance(value, dict) or "window_days" not in value:
            return value

        migrated = dict(value)
        legacy_value = migrated.pop("window_days")
        if migrated.get("window_text") is None and legacy_value is not None:
            migrated["window_text"] = str(legacy_value)
        if "window_min" not in migrated and "window_max" not in migrated:
            if isinstance(legacy_value, int) and not isinstance(legacy_value, bool) and legacy_value >= 0:
                migrated["window_min"] = legacy_value
                migrated["window_max"] = legacy_value
            elif isinstance(legacy_value, str) and legacy_value.isdecimal():
                migrated["window_min"] = int(legacy_value)
                migrated["window_max"] = int(legacy_value)
        return migrated

    @model_validator(mode="after")
    def validate_window_bounds(self) -> "PolicyRule":
        if (
            self.window_min is not None
            and self.window_max is not None
            and self.window_min > self.window_max
        ):
            raise ValueError("window_min must be less than or equal to window_max")
        return self


class ExtractionResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    chunk_id: str
    entities: list[Entity] = Field(default_factory=list)
    rules: list[PolicyRule] = Field(default_factory=list)


class RuleLinkCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rule_id: str
    score: float = Field(ge=0.0, le=1.0)
    reasons: list[str] = Field(default_factory=list)


class RuleLinkResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    decision: Literal["MERGE", "NEW"]
    candidate_rule_id: str | None = None
    score: float = Field(default=0.0, ge=0.0, le=1.0)
    reasons: list[str] = Field(default_factory=list)


class GraphEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rule_id: str
    source_chunks: list[str] = Field(default_factory=list)
    evidence_text: list[str] = Field(default_factory=list)


class QueryUnderstanding(BaseModel):
    model_config = ConfigDict(extra="forbid")

    product: str | None = None
    product_category: ProductCategory | None = None
    condition: str | None = None
    item_conditions: list[ItemCondition] | None = None
    customer_type: str | None = None
    purchase_age_days: int | None = Field(default=None, ge=0)
    action: str | None = None


class QueryResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rule_id: str | None = None
    rule: PolicyRule | None = None
    evidence: GraphEvidence | None = None
    score: float = Field(default=0.0, ge=0.0, le=1.0)


class GraphChunk(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(default_factory=lambda: str(uuid4()))
    chunk_id: str
    doc_id: str
    version_number: int = Field(ge=1)
    raw_text: str
    contextual_text: str = ""
    hierarchical_path: str = ""


class GraphSyncPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    chunk: GraphChunk
    extraction: ExtractionResult
    evidence: list[GraphEvidence] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_references(self) -> "GraphSyncPayload":
        if self.extraction.chunk_id != self.chunk.chunk_id:
            raise ValueError("extraction.chunk_id must match chunk.chunk_id")
        rule_ids = {rule.id for rule in self.extraction.rules}
        if any(item.rule_id not in rule_ids for item in self.evidence):
            raise ValueError("Graph evidence must reference a rule in the extraction")
        return self


class GraphRagRunResult(BaseModel):
    doc_id: str
    version_number: int
    classified_block_count: int
    graph_worthy_block_count: int
    persisted_chunk_count: int
    entity_count: int
    rule_count: int


GraphProcess = Literal[
    "CLASSIFYING",
    "EXTRACTING",
    "NORMALIZING",
    "LINKING",
    "PERSISTING_POSTGRES",
    "PERSISTING_NEO4J",
]
GraphProcessStatus = Literal["SUCCESS", "FAIL", "WARNING"]


class GraphLogRecord(BaseModel):
    trace_id: str
    process: GraphProcess
    status: GraphProcessStatus
    message: str = ""
    metrics: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
