"""Run the policy-agent graph with a small in-memory retrieval pipeline.

From the repository root, run ``python -m scripts.policy_agent_demo``.
No database, vector store, or external API is required.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import sys

from agent.agents.policy.graph import PolicyAgentGraph


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
    """Small stand-in for the project's RAG retrieval pipeline."""

    def __init__(self, *, return_evidence: bool) -> None:
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
                    text="Sáº£n pháº©m cĂ²n nguyĂªn tem vĂ  chÆ°a qua sá»­ dá»¥ng cĂ³ thá»ƒ Ä‘Æ°á»£c yĂªu cáº§u Ä‘á»•i tráº£ trong thá»i háº¡n quy Ä‘á»‹nh.",
                    metadata={
                        "source_title": "ChĂ­nh sĂ¡ch Ä‘á»•i tráº£",
                        "source_url": "https://example.com/chinh-sach-doi-tra",
                    },
                ),
                score=0.94,
            ),
            DemoSearchResult(
                hit=DemoHit(
                    doc_id="refund-policy",
                    version=1,
                    chunk_id="refund-policy-003",
                    text="Tiá»n hoĂ n Ä‘Æ°á»£c xá»­ lĂ½ sau khi sáº£n pháº©m tráº£ vá» Ä‘Æ°á»£c tiáº¿p nháº­n vĂ  kiá»ƒm tra.",
                    metadata={"source_title": "ChĂ­nh sĂ¡ch hoĂ n tiá»n"},
                ),
                score=0.81,
            ),
        ]
        return examples[:limit]


def _run_scenario(name: str, message: str, *, return_evidence: bool) -> None:
    retrieval = DemoPolicySearch(return_evidence=return_evidence)
    result = PolicyAgentGraph(retrieval).invoke(message, language="vi")

    print(f"\n{'=' * 72}")
    print(f"Question: {message}")
    print(f"Lookup status: {result.lookup_status}")
    print(f"Evidence count: {len(result.evidence)}")
    print(f"\nResponse:\n{result.message}")


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(
        description="Cháº¡y thá»­ PolicyAgentGraph báº±ng dá»¯ liá»‡u retrieval giáº£ láº­p."
    )
    parser.add_argument(
        "--scenario",
        choices=("all", "evidence", "not-found"),
        default="all",
        help="Ká»‹ch báº£n muá»‘n cháº¡y (máº·c Ä‘á»‹nh: cháº¡y cáº£ ba).",
    )
    scenario = parser.parse_args().scenario

    scenarios: dict[str, tuple[str, str, bool]] = {
        "evidence": (
            "TĂ¬m tháº¥y tĂ i liá»‡u chĂ­nh sĂ¡ch",
            "ChĂ­nh sĂ¡ch Ä‘á»•i tráº£ sáº£n pháº©m nhÆ° tháº¿ nĂ o?",
            True,
        ),
        "not-found": (
            "KhĂ´ng tĂ¬m tháº¥y tĂ i liá»‡u phĂ¹ há»£p",
            "ChĂ­nh sĂ¡ch Ä‘á»•i tráº£ sáº£n pháº©m nhÆ° tháº¿ nĂ o?",
            False,
        ),
    }
    selected = scenarios.items() if scenario == "all" else [(scenario, scenarios[scenario])]
    for _, (name, message, return_evidence) in selected:
        _run_scenario(name, message, return_evidence=return_evidence)


if __name__ == "__main__":
    main()
