"""Run one request through the ManagerAgentGraph from a local CLI.

This script uses an in-memory policy search adapter, so it does not call the
database, vector store, or external LLM. It exercises manager routing and the
policy-agent boundary with deterministic sample evidence.

Usage from the repository root:
    python -m scripts.run_agent --message "What is the return policy?"
    python -m scripts.run_agent --message "Xin chao" --language vi --json
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import sys
from typing import Any
from uuid import UUID

from agent.agents.policy.graph import PolicyAgentGraph
from agent.orchestration.graph import ManagerAgentGraph


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
        "--json",
        action="store_true",
        help="Print the complete final AgentSharedState as JSON.",
    )
    return parser.parse_args()


def _print_summary(state: Any, search: DemoPolicySearch) -> None:
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
        _print_summary(state, policy_search)


if __name__ == "__main__":
    main()
