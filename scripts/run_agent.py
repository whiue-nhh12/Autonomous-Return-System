"""Run one request through the ManagerAgentGraph from a local CLI.

Policy search is backed by an in-memory demo adapter. Normal order requests use
the configured read-only database service. Pass ``--demo-return-decision`` to
run the return eligibility path offline with a sample order and explicitly
illustrative rules.

Usage from the repository root:
    python -m scripts.run_agent --message "What is the return policy?"
    python -m scripts.run_agent --demo-return-decision --message "Tôi có đủ điều kiện trả 1 sản phẩm trong đơn ORD-DEMO-001 vì bị hỏng và chưa mở không?" --language vi
    python -m scripts.run_agent --message "Xin chao" --language vi --json
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal
import json
import sys
from typing import Any
from uuid import UUID, uuid4

from agent.agents.order.graph import OrderAgentGraph
from agent.agents.policy.graph import PolicyAgentGraph
from agent.agents.return_decision.agent import ReturnDecisionAgent
from agent.agents.return_decision.claims import extract_return_claim
from agent.agents.return_decision.graph import ReturnDecisionAgentGraph
from agent.agents.return_decision.rules import ReturnPolicyRuleSet
from agent.core.state.shared import ReturnIntent
from agent.orchestration.graph import ManagerAgentGraph
from backend.models.enums import OrderStatus
from backend.models.order import Order, OrderItem
from backend.repositories.order_read import OrderReadBundle, OrderReadPage


@dataclass
class DemoHit:
    doc_id: str
    version: int
    chunk_id: str
    text: str
    metadata: dict[str, str]


@dataclass
class DemoSearchResult:
    hit: DemoHit
    score: float


class DemoPolicySearch:
    """Deterministic stand-in for RetrievalPipeline.search."""

    def __init__(self, *, return_evidence: bool = True) -> None:
        self.return_evidence = return_evidence
        self.queries: list[str] = []

    def search(
        self,
        query: str,
        *,
        limit: int,
        history: list[str] | None = None,
    ) -> list[DemoSearchResult]:
        self.queries.append(query)
        if not self.return_evidence:
            return []

        examples = [
            DemoSearchResult(
                hit=DemoHit(
                    doc_id="return-policy",
                    version=2,
                    chunk_id="return-policy-001",
                    text=(
                        "Items in original condition and with their tags attached may be "
                        "eligible for return within the stated return period."
                    ),
                    metadata={
                        "source_title": "Return policy",
                        "source_url": "https://example.com/return-policy",
                    },
                ),
                score=0.94,
            ),
            DemoSearchResult(
                hit=DemoHit(
                    doc_id="refund-policy",
                    version=1,
                    chunk_id="refund-policy-003",
                    text=(
                        "Refunds are processed after the returned item is received and inspected."
                    ),
                    metadata={"source_title": "Refund policy"},
                ),
                score=0.81,
            ),
        ]
        return examples[:limit]


class DemoOrderSearch:
    """In-memory order lookup used only by the offline decision demo."""

    def __init__(self, user_id: UUID) -> None:
        now = datetime.now(timezone.utc)
        order_id = uuid4()
        order = Order(
            id=order_id,
            order_number="ORD-DEMO-001",
            user_id=user_id,
            status=OrderStatus.DELIVERED,
            currency="VND",
            subtotal=Decimal("500000"),
            discount_amount=Decimal("0"),
            shipping_fee=Decimal("0"),
            total_amount=Decimal("500000"),
            placed_at=now - timedelta(days=8),
            delivered_at=now - timedelta(days=5),
        )
        item = OrderItem(
            id=uuid4(),
            order_id=order_id,
            product_id=uuid4(),
            product_name_snapshot="Demo wireless headphones",
            sku_snapshot="SKU-DEMO-HEADPHONES",
            quantity=1,
            unit_price=Decimal("500000"),
            line_total=Decimal("500000"),
        )
        self.user_id = user_id
        self.bundle = OrderReadBundle(
            order=order,
            items=(item,),
            payments=(),
            shipments=(),
            returns=(),
            return_items={},
            return_status_history={},
            refunds={},
            return_shipments={},
        )

    def search_history(
        self,
        *,
        user_id: UUID,
        order_number: str | None = None,
        limit: int = 10,
    ) -> OrderReadPage:
        if user_id != self.user_id:
            return OrderReadPage(orders=(), has_more=False)
        if order_number and order_number.casefold() != self.bundle.order.order_number.casefold():
            return OrderReadPage(orders=(), has_more=False)
        return OrderReadPage(orders=(self.bundle,), has_more=False)


class DemoPolicyRuleProvider:
    """Illustrative rules for demonstrating the decision path, not production policy."""

    def get_rule_set(self, evidence, *, evaluated_at: datetime) -> ReturnPolicyRuleSet | None:
        if not evidence:
            return None
        return ReturnPolicyRuleSet(
            policy_id="demo-return-policy",
            version="demo-v1",
            auto_decision_enabled=True,
            effective_from=evaluated_at - timedelta(days=365),
            allowed_return_days=30,
            eligible_order_statuses={"delivered"},
            eligible_reasons={"damaged", "wrong_item", "not_as_described", "changed_mind", "other"},
            eligible_item_conditions={"unopened", "opened", "used", "damaged", "unknown"},
        )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Send one input through the manager-agent graph and print its result."
    )
    parser.add_argument(
        "--message",
        help="User message. If omitted, the script prompts for one interactively.",
    )
    parser.add_argument("--language", default="en", help="Response language (default: en).")
    parser.add_argument(
        "--conversation-id",
        default=None,
        help="Optional conversation identifier to attach to this request.",
    )
    parser.add_argument(
        "--user-id",
        default="00000000-0000-4000-8000-000000000001",
        help="Trusted test user UUID (default is a local test identity).",
    )
    parser.add_argument(
        "--no-policy-evidence",
        action="store_true",
        help="Make the demo policy search return no results.",
    )
    parser.add_argument(
        "--demo-return-decision",
        action="store_true",
        help="Use an in-memory sample order and illustrative rules to display the full eligibility flow offline.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the complete final AgentSharedState as JSON.",
    )
    return parser.parse_args()


def _print_summary(
    state: Any,
    search: DemoPolicySearch,
    *,
    demo_return_decision: bool = False,
) -> None:
    intent = state.classification.intent.value if state.classification.intent else None
    print(f"Request ID: {state.context.request_id}")
    print(f"Intent: {intent}")
    print(f"Orchestration status: {state.orchestration.status}")
    print(f"Response status: {state.response.status}")
    print("\nResponse:")
    print(state.response.message or "(no response message)")

    if state.policy_lookup.evidence:
        print("\nPolicy evidence:")
        for evidence in state.policy_lookup.evidence:
            title = evidence.source_title or evidence.doc_id
            print(f"- {title} v{evidence.version}: {evidence.excerpt}")
            if evidence.source_url:
                print(f"  {evidence.source_url}")

    if search.queries:
        print("\nRetrieval queries:")
        for query in search.queries:
            print(f"- {query}")

    if state.orchestration.child_results:
        print("\nChild results:")
        for name, result in state.orchestration.child_results.items():
            print(f"- {name}: {result.status} ({result.error_code or 'no error'})")

    if state.classification.intent is ReturnIntent.RETURN_ELIGIBILITY:
        print("\nReturn decision flow:")
        print(f"1. Policy lookup: {state.policy_lookup.status}; {len(state.policy_lookup.evidence)} evidence item(s)")
        print(f"2. Order lookup: {state.order_lookup.status}; {len(state.order_lookup.orders)} order(s)")
        if state.order_lookup.data:
            print(f"   Selected order: {state.order_lookup.data.order_number}")

        claim = extract_return_claim(state.input.message, state.order_lookup.orders)
        item_description = claim.item_query or "(not specified)"
        if (
            not claim.item_query
            and state.order_lookup.data
            and len(state.order_lookup.data.items) == 1
        ):
            item_description = f"{state.order_lookup.data.items[0].product_name} (only order item)"
        print(
            "3. Extracted claim: "
            f"order={claim.order_number or '(not specified)'}, "
            f"item={item_description}, "
            f"quantity={claim.requested_quantity or '(not specified)'}, "
            f"reason={claim.reason_code or '(not identified)'}, "
            f"condition={claim.item_condition or '(not identified)'}"
        )

        print("4. Evaluator checks:")
        if state.business_validation.checks:
            for code, status in state.business_validation.checks.items():
                print(f"   - {code}: {status}")
        else:
            print("   - No checks completed (the request may need clarification or a worker may have failed).")

        proposal = state.decision_proposal
        print(f"5. Proposal: status={proposal.status}; outcome={proposal.outcome or '(none)'}")
        if proposal.reason_codes:
            print(f"   Reason codes: {', '.join(proposal.reason_codes)}")
        if proposal.rule_set_version:
            print(f"   Rule set: {proposal.rule_set_version}")
        if proposal.evidence_refs:
            print(f"   Evidence: {', '.join(proposal.evidence_refs)}")
        if demo_return_decision:
            print("   Demo mode: sample order and illustrative rules; not production policy.")


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    args = _parse_args()
    message = args.message if args.message is not None else input("Enter a message for the agent: ")
    if not message.strip():
        raise SystemExit("Message cannot be empty.")

    try:
        user_id = UUID(args.user_id)
    except ValueError as exc:
        raise SystemExit(f"Invalid --user-id UUID: {args.user_id}") from exc

    policy_search = DemoPolicySearch(return_evidence=not args.no_policy_evidence)
    policy_agent = PolicyAgentGraph(policy_search)
    if args.demo_return_decision:
        order_agent = OrderAgentGraph(DemoOrderSearch(user_id))
        decision_agent = ReturnDecisionAgentGraph(
            ReturnDecisionAgent(DemoPolicyRuleProvider())
        )
        manager = ManagerAgentGraph(
            policy_agent=policy_agent,
            order_agent=order_agent,
            return_decision_agent=decision_agent,
        )
    else:
        manager = ManagerAgentGraph(policy_agent=policy_agent)
    state = manager.invoke(
        message,
        user_id=user_id,
        language=args.language,
        conversation_id=args.conversation_id,
    )

    if args.json:
        print(json.dumps(state.model_dump(mode="json"), ensure_ascii=False, indent=2))
    else:
        _print_summary(state, policy_search, demo_return_decision=args.demo_return_decision)


if __name__ == "__main__":
    main()
