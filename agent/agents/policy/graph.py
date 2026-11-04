"""LangGraph workflow for retrieving and explaining published policy evidence."""

from __future__ import annotations

from typing import Literal, TypedDict

from langgraph.graph import END, START, StateGraph

from agent.agents.policy.agent import PolicyAgent, PolicySearchPipeline
from agent.agents.policy.state.retrieval import DocumentRetrievalLocalState
from agent.agents.policy.state.result import PolicyAgentRequest, PolicyAgentResult
from rag.core.logger.agents.policy_agent.policy_agent_logger import PolicyAgentLogger
from rag.core.schemas.policy_agent_schemas import PolicyAgentLogStatus, PolicyAgentProcess


class PolicyAgentGraphState(TypedDict, total=False):
    """Policy-local graph state. It never contains manager shared state."""

    request: PolicyAgentRequest
    retrieval: DocumentRetrievalLocalState
    result: PolicyAgentResult


class PolicyAgentGraph:
    """Retrieve policy evidence and return a typed result to the orchestrator.

    Routing, clarification, response assembly, and shared manager-state updates
    belong to orchestration. This child graph only performs policy lookup.
    """

    def __init__(
        self,
        retrieval: PolicySearchPipeline,
        logger: PolicyAgentLogger | None = None,
    ) -> None:
        self.policy_retriever = PolicyAgent(retrieval)
        self.logger = logger or PolicyAgentLogger()
        builder = StateGraph(PolicyAgentGraphState)
        builder.add_node("retrieve_policy", self._retrieve_policy)
        builder.add_node("answer_with_evidence", self._answer_with_evidence)
        builder.add_node("no_evidence", self._no_evidence)
        builder.add_edge(START, "retrieve_policy")
        builder.add_conditional_edges(
            "retrieve_policy",
            self._route_after_retrieval,
            {"answer_with_evidence": "answer_with_evidence", "no_evidence": "no_evidence"},
        )
        builder.add_edge("answer_with_evidence", END)
        builder.add_edge("no_evidence", END)
        self.graph = builder.compile()

    def invoke(
        self,
        request: str | PolicyAgentRequest,
        *,
        language: str = "vi",
        request_id: str | None = None,
    ) -> PolicyAgentResult:
        """Run one policy lookup from START and return only its local result."""
        policy_request = (
            request
            if isinstance(request, PolicyAgentRequest)
            else PolicyAgentRequest(message=request, language=language, request_id=request_id)
        )
        result = self.graph.invoke({"request": policy_request})
        return result["result"]

    def _retrieve_policy(self, state: PolicyAgentGraphState) -> dict[str, DocumentRetrievalLocalState]:
        request = state["request"]
        local = DocumentRetrievalLocalState(original_query=request.message)
        self.policy_retriever.retrieve(local)
        self._emit(
            request,
            "RETRIEVE_POLICY",
            {"completed": "SUCCESS", "not_found": "WARNING", "failed": "FAIL"}.get(
                local.status, "WARNING"
            ),
            local.error or local.status,
            {
                "candidate_count": len(local.candidates),
                "evidence_count": len(local.selected_evidence),
                "query_count": 1 + len(local.query_variants),
            },
        )
        return {"retrieval": local}

    @staticmethod
    def _route_after_retrieval(
        state: PolicyAgentGraphState,
    ) -> Literal["answer_with_evidence", "no_evidence"]:
        if state["retrieval"].status == "completed":
            return "answer_with_evidence"
        return "no_evidence"

    def _answer_with_evidence(self, state: PolicyAgentGraphState) -> dict[str, PolicyAgentResult]:
        request = state["request"]
        evidence = state["retrieval"].selected_evidence
        english = request.language.casefold().startswith("en")
        paragraphs = [
            "Based on the published policy documents I found:" if english else "Theo tài liệu chính sách đã tìm thấy:"
        ]
        for index, item in enumerate(evidence[:3], start=1):
            title = item.source_title or item.doc_id
            citation = f"[{index}] {title} (version {item.version})"
            if item.source_url:
                citation += f": {item.source_url}"
            source_label = "Source" if english else "Nguồn"
            paragraphs.append(f"- {item.excerpt}\n  {source_label}: {citation}")
        result = PolicyAgentResult(
            lookup_status="succeeded",
            message="\n\n".join(paragraphs),
            evidence=evidence,
        )
        self._emit(
            request,
            "ANSWER_WITH_EVIDENCE",
            "SUCCESS",
            "Policy answer assembled from retrieved evidence",
            {"evidence_count": min(len(evidence), 3)},
        )
        return {"result": result}

    def _no_evidence(self, state: PolicyAgentGraphState) -> dict[str, PolicyAgentResult]:
        request = state["request"]
        retrieval = state["retrieval"]
        english = request.language.casefold().startswith("en")
        if retrieval.status == "failed":
            message = (
                "Policy lookup is temporarily unavailable. Please try again later."
                if english
                else "Hiện chưa thể tra cứu chính sách. Vui lòng thử lại sau."
            )
            lookup_status: Literal["not_found", "failed"] = "failed"
            detail = retrieval.error or "Policy retrieval failed"
        else:
            message = (
                "I couldn't find a matching policy document. Which store, product, or policy type are you asking about?"
                if english
                else "Mình chưa tìm thấy tài liệu chính sách phù hợp. Bạn đang hỏi về cửa hàng, sản phẩm hay loại chính sách nào?"
            )
            lookup_status = "not_found"
            detail = "No matching published policy evidence"
        result = PolicyAgentResult(
            lookup_status=lookup_status,
            message=message,
            error=retrieval.error,
        )
        self._emit(
            request,
            "NO_EVIDENCE",
            "FAIL" if lookup_status == "failed" else "WARNING",
            detail,
            {"lookup_status": lookup_status},
        )
        return {"result": result}

    def _emit(
        self,
        request: PolicyAgentRequest,
        process: PolicyAgentProcess,
        status: PolicyAgentLogStatus,
        message: str,
        metrics: dict[str, object],
    ) -> None:
        self.logger.emit(
            trace_id=request.request_id or "policy-agent-request",
            process=process,
            status=status,
            metrics=metrics,
            message=message,
        )
