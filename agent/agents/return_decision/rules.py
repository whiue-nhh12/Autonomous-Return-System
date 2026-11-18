"""Conservative, deterministic return eligibility rules."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Protocol

from pydantic import Field

from agent.agents.return_decision.state.result import (
    CheckStatus,
    DecisionCheck,
    ReturnClaimFacts,
    ReturnDecisionRequest,
    ReturnDecisionResult,
)
from agent.core.state.shared import OrderItemLookup, OrderLookupData, PolicyEvidence, StateModel


class ReturnPolicyRuleSet(StateModel):
    """Reviewed, structured rules derived from one published policy version.

    ``auto_decision_enabled`` is an explicit operational gate. The repository
    currently has no structured policy-rule source, so the default agent does
    not auto-approve or auto-reject from retrieved prose alone.
    """

    policy_id: str
    version: str
    auto_decision_enabled: bool = False
    effective_from: datetime | None = None
    allowed_return_days: int | None = Field(default=None, gt=0)
    eligible_order_statuses: set[str] = Field(default_factory=set)
    eligible_reasons: set[str] = Field(default_factory=set)
    eligible_item_conditions: set[str] = Field(default_factory=set)


class ReturnPolicyRuleProvider(Protocol):
    def get_rule_set(
        self, evidence: list[PolicyEvidence], *, evaluated_at: datetime
    ) -> ReturnPolicyRuleSet | None:
        """Return a reviewed rule set applicable to the supplied evidence."""


class NoStructuredPolicyRules:
    """Default provider until policy ingestion exposes structured rules."""

    def get_rule_set(
        self, evidence: list[PolicyEvidence], *, evaluated_at: datetime
    ) -> ReturnPolicyRuleSet | None:
        return None


def evaluate_return_request(
    request: ReturnDecisionRequest,
    rule_set: ReturnPolicyRuleSet | None,
) -> ReturnDecisionResult:
    """Evaluate a request without side effects or external calls."""
    checks: list[DecisionCheck] = []
    refs = [_evidence_ref(item) for item in request.policy_evidence]

    if request.claim.model_derived_fields:
        english = request.language.casefold().startswith("en")
        fields = ", ".join(sorted(request.claim.model_derived_fields))
        checks.append(
            _check(
                "model_claim_review",
                "unknown",
                "MODEL_EXTRACTED_CLAIM_REQUIRES_REVIEW",
                f"Thông tin được trích xuất bằng mô hình cần nhân viên rà soát: {fields}.",
                f"These model-extracted details require manual review: {fields}.",
                refs,
                english=english,
            )
        )
        return _result(request, "manual_review", checks, refs, rule_set)

    order = _select_order(request)
    if order is None:
        if len(request.orders) > 1 and request.claim.order_number is None:
            return _clarification(request, ["order_number"], checks, refs)
        checks.append(_check("order_found", "unknown", "ORDER_NOT_FOUND", "Không tìm thấy đơn hàng phù hợp.", "No matching order was found.", english=request.language.casefold().startswith("en")))
        return _result(request, "manual_review", checks, refs, rule_set)
    checks.append(_check("order_found", "pass", "ORDER_FOUND", "Đã tìm thấy đơn hàng.", "The order was found.", english=request.language.casefold().startswith("en")))

    item, item_issue = _select_item(order, request.claim)
    if item_issue == "ambiguous":
        return _clarification(request, ["item"], checks, refs)
    if item_issue == "missing":
        return _clarification(request, ["item"], checks, refs)
    if item is None:
        checks.append(_check("item_on_order", "fail", "ITEM_NOT_ON_ORDER", "Không tìm thấy sản phẩm này trong đơn hàng.", "The item was not found on this order.", english=request.language.casefold().startswith("en")))
        return _result(request, "reject", checks, refs, rule_set)
    checks.append(_check("item_on_order", "pass", "ITEM_ON_ORDER", "Sản phẩm thuộc đơn hàng.", "The item belongs to the order.", english=request.language.casefold().startswith("en")))

    if request.claim.requested_quantity is None:
        return _clarification(request, ["requested_quantity"], checks, refs)
    remaining = max(0, item.purchased_quantity - _reserved_return_quantity(order, item))
    quantity_status: CheckStatus = "pass" if request.claim.requested_quantity <= remaining else "fail"
    checks.append(_check(
        "quantity_available", quantity_status,
        "QUANTITY_AVAILABLE" if quantity_status == "pass" else "QUANTITY_EXCEEDS_REMAINING",
        f"Còn có thể yêu cầu trả tối đa {remaining} sản phẩm." if quantity_status == "pass" else f"Số lượng có thể trả còn lại là {remaining}.",
        f"Up to {remaining} item(s) remain available for return." if quantity_status == "pass" else f"Only {remaining} item(s) remain available for return.",
        english=request.language.casefold().startswith("en"),
    ))
    if quantity_status == "fail":
        return _result(request, "reject", checks, refs, rule_set)

    if request.policy_lookup_status == "failed":
        checks.append(_check("policy_available", "unknown", "POLICY_LOOKUP_FAILED", "Hiện chưa tra cứu được chính sách áp dụng.", "The applicable policy could not be retrieved.", english=request.language.casefold().startswith("en")))
        return _result(request, "manual_review", checks, refs, rule_set)
    if not request.policy_evidence:
        checks.append(_check("policy_evidence", "unknown", "POLICY_EVIDENCE_MISSING", "Chưa có bằng chứng chính sách phù hợp.", "No relevant policy evidence is available.", english=request.language.casefold().startswith("en")))
        return _result(request, "manual_review", checks, refs, rule_set)
    checks.append(_check("policy_evidence", "pass", "POLICY_EVIDENCE_FOUND", "Đã tìm thấy tài liệu chính sách liên quan.", "Relevant policy evidence was found.", refs, english=request.language.casefold().startswith("en")))

    if rule_set is None or not rule_set.auto_decision_enabled:
        checks.append(_check("structured_policy_rules", "unknown", "STRUCTURED_RULES_UNAVAILABLE", "Chính sách chưa được cấu hình thành bộ quy tắc đã duyệt để tự động quyết định.", "The policy has no reviewed structured rules for automated decisions.", refs, english=request.language.casefold().startswith("en")))
        return _result(request, "manual_review", checks, refs, rule_set)

    if not any((rule_set.allowed_return_days, rule_set.eligible_order_statuses, rule_set.eligible_reasons, rule_set.eligible_item_conditions)):
        checks.append(_check("structured_policy_rules", "unknown", "POLICY_RULES_EMPTY", "Bộ quy tắc chưa có điều kiện đủ để đánh giá.", "The rule set has no conditions to evaluate.", refs, english=request.language.casefold().startswith("en")))
        return _result(request, "manual_review", checks, refs, rule_set)

    if rule_set.effective_from and request.evaluated_at < rule_set.effective_from:
        checks.append(_check("policy_effective", "unknown", "POLICY_NOT_EFFECTIVE", "Chưa xác định được phiên bản chính sách áp dụng tại thời điểm này.", "The applicable policy version could not be established." , refs, english=request.language.casefold().startswith("en")))
        return _result(request, "manual_review", checks, refs, rule_set)

    hard_fail = False
    unknown = False
    if rule_set.eligible_order_statuses:
        passed = order.status.casefold() in {item.casefold() for item in rule_set.eligible_order_statuses}
        checks.append(_check("order_status", "pass" if passed else "fail", "ORDER_STATUS_ELIGIBLE" if passed else "ORDER_STATUS_INELIGIBLE", "Trạng thái đơn đáp ứng chính sách." if passed else "Trạng thái đơn không đáp ứng chính sách.", "The order status meets policy." if passed else "The order status does not meet policy.", refs, english=request.language.casefold().startswith("en")))
        hard_fail |= not passed
    if rule_set.allowed_return_days is not None:
        if order.delivered_at is None:
            checks.append(_check("return_window", "unknown", "DELIVERY_DATE_MISSING", "Đơn chưa có ngày giao để xác định thời hạn trả.", "The delivery date is unavailable for return-window evaluation.", refs, english=request.language.casefold().startswith("en")))
            unknown = True
        else:
            delivered_at = _as_aware(order.delivered_at)
            elapsed_days = (request.evaluated_at - delivered_at).days
            passed = 0 <= elapsed_days <= rule_set.allowed_return_days
            checks.append(_check("return_window", "pass" if passed else "fail", "WITHIN_RETURN_WINDOW" if passed else "RETURN_WINDOW_EXPIRED", "Đơn còn trong thời hạn trả hàng." if passed else "Đơn đã quá thời hạn trả hàng.", "The order is within the return window." if passed else "The return window has expired.", refs, english=request.language.casefold().startswith("en")))
            hard_fail |= not passed
    if rule_set.eligible_reasons:
        if not request.claim.reason_code:
            checks.append(_check("return_reason", "unknown", "RETURN_REASON_MISSING", "Cần biết lý do trả hàng để áp dụng chính sách.", "A return reason is required to apply the policy.", refs, english=request.language.casefold().startswith("en")))
            unknown = True
        else:
            passed = request.claim.reason_code.casefold() in {item.casefold() for item in rule_set.eligible_reasons}
            checks.append(_check("return_reason", "pass" if passed else "fail", "RETURN_REASON_ELIGIBLE" if passed else "RETURN_REASON_INELIGIBLE", "Lý do trả hàng được chính sách chấp nhận." if passed else "Lý do trả hàng không thuộc diện được chấp nhận.", "The return reason is allowed by policy." if passed else "The return reason is not allowed by policy.", refs, english=request.language.casefold().startswith("en")))
            hard_fail |= not passed
    if rule_set.eligible_item_conditions:
        if not request.claim.item_condition:
            checks.append(_check("item_condition", "unknown", "ITEM_CONDITION_MISSING", "Cần biết tình trạng sản phẩm để áp dụng chính sách.", "The item's condition is required to apply the policy.", refs, english=request.language.casefold().startswith("en")))
            unknown = True
        else:
            passed = request.claim.item_condition.casefold() in {item.casefold() for item in rule_set.eligible_item_conditions}
            checks.append(_check("item_condition", "pass" if passed else "fail", "ITEM_CONDITION_ELIGIBLE" if passed else "ITEM_CONDITION_INELIGIBLE", "Tình trạng sản phẩm đáp ứng chính sách." if passed else "Tình trạng sản phẩm không đáp ứng chính sách.", "The item condition meets policy." if passed else "The item condition does not meet policy.", refs, english=request.language.casefold().startswith("en")))
            hard_fail |= not passed

    outcome = "reject" if hard_fail else "manual_review" if unknown else "approve"
    return _result(request, outcome, checks, refs, rule_set)


def _select_order(request: ReturnDecisionRequest) -> OrderLookupData | None:
    if request.claim.order_number:
        matches = [order for order in request.orders if order.order_number.casefold() == request.claim.order_number.casefold()]
    else:
        matches = request.orders
    return matches[0] if len(matches) == 1 else None


def _select_item(order: OrderLookupData, claim: ReturnClaimFacts) -> tuple[OrderItemLookup | None, str | None]:
    if claim.order_item_id:
        matches = [item for item in order.items if item.order_item_id == claim.order_item_id]
    elif claim.item_query:
        query = claim.item_query.casefold().strip()
        matches = [item for item in order.items if query in item.product_name.casefold() or query == item.sku.casefold()]
    elif len(order.items) == 1:
        matches = order.items
    else:
        return None, "ambiguous"
    if len(matches) != 1:
        return None, "missing" if not matches else "ambiguous"
    return matches[0], None


def _reserved_return_quantity(order: OrderLookupData, item: OrderItemLookup) -> int:
    active = {"submitted", "reviewing", "approved", "item_received"}
    reserved = 0
    for request in order.returns:
        if request.status.casefold() not in active:
            continue
        reserved += sum(row.requested_quantity for row in request.items if row.order_item_id == item.order_item_id)
    return item.quantity_already_returned + reserved


def _as_aware(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value


def _check(code: str, status: CheckStatus, reason_code: str, vi: str, en: str, refs: list[str] | None = None, *, english: bool = False) -> DecisionCheck:
    return DecisionCheck(code=code, status=status, reason_code=reason_code, message=en if english else vi, evidence_refs=refs or [])


def _result(request: ReturnDecisionRequest, outcome: str, checks: list[DecisionCheck], refs: list[str], rules: ReturnPolicyRuleSet | None) -> ReturnDecisionResult:
    english = request.language.casefold().startswith("en")
    key = {"approve": ("Yêu cầu có vẻ đáp ứng các điều kiện đã cấu hình.", "The request appears to meet the configured conditions."), "reject": ("Yêu cầu không đáp ứng ít nhất một điều kiện trả hàng đã xác định.", "The request fails at least one established return condition."), "manual_review": ("Yêu cầu cần được nhân viên xem xét vì chưa đủ căn cứ để tự động quyết định.", "A staff member must review this request because the available evidence is insufficient for an automated decision.")}[outcome]
    failed_codes = [check.reason_code for check in checks if check.status != "pass"]
    return ReturnDecisionResult(status="completed", outcome=outcome, message=key[1] if english else key[0], checks=checks, reason_codes=failed_codes, evidence_refs=refs, rule_set_version=rules.version if rules else None, evaluated_at=request.evaluated_at)


def _clarification(request: ReturnDecisionRequest, missing: list[str], checks: list[DecisionCheck], refs: list[str]) -> ReturnDecisionResult:
    english = request.language.casefold().startswith("en")
    labels = {"order_number": ("mã đơn hàng", "the order number"), "item": ("sản phẩm cần trả", "the item to return"), "requested_quantity": ("số lượng muốn trả", "the quantity to return")}
    missing_label = ", ".join(labels.get(item, (item, item))[1 if english else 0] for item in missing)
    message = f"Please provide {missing_label}." if english else f"Vui lòng cho biết {missing_label}."
    return ReturnDecisionResult(status="needs_clarification", message=message, checks=checks, reason_codes=["REQUIRED_FACTS_MISSING"], evidence_refs=refs, evaluated_at=request.evaluated_at, missing_fields=missing)


def _evidence_ref(evidence: PolicyEvidence) -> str:
    return f"{evidence.doc_id}:v{evidence.version}:{evidence.chunk_id}"
