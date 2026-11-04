import unittest
from uuid import UUID

from pydantic import ValidationError

from agent.policy_agent.state import (
    AgentSharedState,
    ClassificationState,
    DocumentCandidate,
    DocumentRetrievalLocalState,
    PolicyEvidence,
    OrderItemLookup,
    OrderLookupData,
    RequestContext,
    RequestInput,
    ReturnIntent,
    StateTransition,
)


class AgentStateTest(unittest.TestCase):
    def test_shared_state_defaults_and_serializes(self) -> None:
        state = AgentSharedState(
            context=RequestContext(user_id=UUID("12345678-1234-5678-1234-567812345678")),
            input=RequestInput(message="Tôi có thể trả đơn ORD-123 không?"),
            classification=ClassificationState(
                status="completed",
                intent=ReturnIntent.RETURN_ELIGIBILITY,
                confidence=0.95,
                entities={"order_number": "ORD-123"},
            ),
        )

        serialized = state.model_dump(mode="json")
        self.assertEqual(serialized["classification"]["intent"], "return_eligibility")
        self.assertEqual(serialized["order_lookup"]["status"], "not_started")
        self.assertEqual(serialized["policy_lookup"]["evidence"], [])
        self.assertIsNotNone(serialized["context"]["request_id"])

    def test_shared_state_rejects_invalid_confidence_and_unknown_fields(self) -> None:
        with self.assertRaises(ValidationError):
            ClassificationState(confidence=1.1)

        with self.assertRaises(ValidationError):
            RequestInput(message="Hello", unexpected="value")

        with self.assertRaises(ValidationError):
            RequestInput(message="   ")

    def test_order_lookup_uses_minimal_typed_data(self) -> None:
        order_data = OrderLookupData(
            order_id=UUID("12345678-1234-5678-1234-567812345678"),
            order_number="ORD-123",
            status="delivered",
            items=[
                OrderItemLookup(
                    order_item_id=UUID("22345678-1234-5678-1234-567812345678"),
                    product_id=UUID("32345678-1234-5678-1234-567812345678"),
                    product_name="Sản phẩm",
                    sku="SKU-1",
                    purchased_quantity=2,
                    quantity_already_returned=1,
                )
            ],
        )

        state = AgentSharedState(
            context=RequestContext(),
            input=RequestInput(message="Kiểm tra điều kiện trả"),
        )
        state.order_lookup.status = "succeeded"
        state.order_lookup.data = order_data

        self.assertEqual(state.order_lookup.data.items[0].purchased_quantity, 2)
        with self.assertRaises(ValidationError):
            OrderItemLookup(
                order_item_id=UUID("22345678-1234-5678-1234-567812345678"),
                product_id=UUID("32345678-1234-5678-1234-567812345678"),
                product_name="Sản phẩm",
                sku="SKU-1",
                purchased_quantity=0,
            )

    def test_document_agent_local_state_copies_selected_citations(self) -> None:
        state = DocumentRetrievalLocalState(
            original_query="Điều kiện trả hàng điện tử",
            normalized_query="điều kiện đổi trả hàng điện tử",
            query_variants=["điều kiện hoàn trả sản phẩm điện tử"],
            candidates=[
                DocumentCandidate(
                    doc_id="policy-1",
                    version=2,
                    chunk_id="chunk-10",
                    text="Sản phẩm được trả trong vòng 7 ngày.",
                    score=0.91,
                    source_title="Chính sách đổi trả",
                    source_url="https://example.test/policy",
                )
            ],
        )

        state.set_selected_evidence(state.candidates)

        self.assertEqual(state.selected_evidence[0].doc_id, "policy-1")
        self.assertEqual(state.selected_evidence[0].version, 2)
        self.assertEqual(state.selected_evidence[0].excerpt, state.candidates[0].text)
        self.assertIsInstance(state.selected_evidence[0], PolicyEvidence)
        self.assertNotIn("candidates", AgentSharedState.model_fields)

    def test_each_state_has_independent_mutable_defaults(self) -> None:
        first = AgentSharedState(
            context=RequestContext(),
            input=RequestInput(message="Câu hỏi 1"),
        )
        second = AgentSharedState(
            context=RequestContext(),
            input=RequestInput(message="Câu hỏi 2"),
        )

        first.transitions.append(
            StateTransition(component="classifier", to_status="completed")
        )

        self.assertEqual(second.transitions, [])


if __name__ == "__main__":
    unittest.main()
