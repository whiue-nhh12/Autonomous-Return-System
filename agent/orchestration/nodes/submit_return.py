"""Create a customer return request from a genuine submit intent."""

from __future__ import annotations

import logging

from agent.agents.return_decision.claims import extract_return_claim
from agent.agents.return_decision.state.result import ReturnClaimFacts
from agent.core.state.shared import ChildAgentResult, StateTransition
from agent.orchestration.state import ManagerGraphState
from backend.db.session import get_session_factory
from service.return_submission import (
    ReturnSubmissionConflict,
    ReturnSubmissionInvalid,
    ReturnSubmissionNeedsFields,
    ReturnSubmissionNotFound,
    ReturnSubmissionService,
    ReturnSubmissionStale,
)


_LOGGER = logging.getLogger(__name__)


def create_submit_return_node(service: ReturnSubmissionService | None = None):
    """Build the persistence node with an optional service for host injection."""
    submission_service = service

    def submit_return(state: ManagerGraphState) -> dict:
        nonlocal submission_service
        shared = state["shared"]
        language = shared.input.language.casefold()
        english = language.startswith("en")
        order_result = state.get("eligibility_order_worker_state")
        if order_result is None or order_result.status != "succeeded":
            message = (
                order_result.message
                if order_result is not None and order_result.message
                else (
                    "I couldn't verify the order, so the return request was not created."
                    if english
                    else "Chưa xác minh được đơn hàng nên mình chưa tạo yêu cầu trả."
                )
            )
            result = ChildAgentResult(
                status=("needs_clarification" if order_result and order_result.status == "needs_clarification" else "failed"),
                message=message,
                error_code="RETURN_ORDER_LOOKUP_INCOMPLETE",
            )
            shared.response.status = "failed" if result.status == "failed" else "ready"
            shared.response.message = message
            if result.status == "needs_clarification":
                shared.clarification.needed = True
                shared.clarification.awaiting_user = True
                shared.clarification.question = message
            return {"shared": shared, "child_results": {"submit_return": result}}

        raw_claim = shared.classification.entities.get("return_claim", {})
        try:
            seed = ReturnClaimFacts.model_validate(raw_claim)
        except (TypeError, ValueError):
            seed = None
        transcript = "\n".join([*shared.context.prior_messages, shared.input.message])
        claim = extract_return_claim(
            shared.input.message,
            order_result.orders,
            seed=seed,
            item_context=transcript,
        )
        selected = _select_order(order_result.orders, claim.order_number)
        if selected is None:
            message = (
                "Which order number should I use for the return?"
                if english
                else "Bạn cho mình xin mã đơn hàng cần trả nhé?"
            )
            shared.clarification.needed = True
            shared.clarification.awaiting_user = True
            shared.clarification.question = message
            shared.clarification.missing_fields = ["order_number"]
            shared.response.status = "ready"
            shared.response.message = message
            result = ChildAgentResult(
                status="needs_clarification",
                message=message,
                error_code="RETURN_ORDER_NUMBER_REQUIRED",
                data={"missing_fields": ["order_number"]},
            )
            return {"shared": shared, "child_results": {"submit_return": result}}

        if shared.context.user_id is None:
            return _failed(
                shared,
                "An authenticated customer account is required to submit a return request.",
                "RETURN_SUBMISSION_UNAUTHENTICATED",
            )
        if submission_service is None:
            submission_service = ReturnSubmissionService(get_session_factory())
        try:
            receipt = submission_service.submit(
                request_id=shared.context.request_id,
                user_id=shared.context.user_id,
                order_snapshot=selected,
                claim=claim,
                reason_note=shared.input.message,
            )
        except ReturnSubmissionNeedsFields as exc:
            message = _clarification_message(exc.fields, english=english)
            shared.clarification.needed = True
            shared.clarification.awaiting_user = True
            shared.clarification.question = message
            shared.clarification.missing_fields = exc.fields
            shared.response.status = "ready"
            shared.response.message = message
            result = ChildAgentResult(
                status="needs_clarification",
                message=message,
                error_code="RETURN_DETAILS_REQUIRED",
                data={"missing_fields": exc.fields},
            )
            return {"shared": shared, "child_results": {"submit_return": result}}
        except ReturnSubmissionNotFound as exc:
            return _failed(shared, str(exc), "RETURN_ORDER_NOT_FOUND")
        except ReturnSubmissionStale as exc:
            return _failed(shared, str(exc), "RETURN_ORDER_DATA_STALE")
        except (ReturnSubmissionConflict, ReturnSubmissionInvalid) as exc:
            return _failed(shared, str(exc), "RETURN_SUBMISSION_REJECTED")
        except Exception:
            _LOGGER.exception(
                "Return request persistence failed",
                extra={"request_id": str(shared.context.request_id)},
            )
            return _failed(
                shared,
                "The return request could not be saved. Please try again later.",
                "RETURN_SUBMISSION_FAILED",
            )

        shared.transitions.append(
            StateTransition(
                component="submit_return",
                to_status="committed",
                detail=str(receipt.return_request_id),
            )
        )
        result = ChildAgentResult(
            status="succeeded",
            message=f"Return request {receipt.return_number} was submitted.",
            data={
                "return_request_id": str(receipt.return_request_id),
                "return_number": receipt.return_number,
                "replayed": receipt.replayed,
            },
        )
        return {
            "shared": shared,
            "return_request_id": receipt.return_request_id,
            "return_request_number": receipt.return_number,
            "return_submission_fingerprint": receipt.evaluated_data_fingerprint,
            "return_submission_claim": claim,
            "child_results": {"submit_return": result},
        }

    return submit_return


def submit_return(state: ManagerGraphState) -> dict:
    """Default entry point retained for direct graph integrations."""
    return create_submit_return_node()(state)


def _select_order(orders, order_number: str | None):
    matches = (
        [order for order in orders if order.order_number.casefold() == order_number.casefold()]
        if order_number
        else orders
    )
    return matches[0] if len(matches) == 1 else None


def _clarification_message(fields: list[str], *, english: bool) -> str:
    labels = {
        "order_number": "the order number" if english else "mã đơn hàng",
        "item": "the item to return" if english else "sản phẩm cần trả",
        "requested_quantity": "the quantity" if english else "số lượng",
        "reason_code": "the return reason" if english else "lý do trả hàng",
        "return_method": "a return method (pickup, drop-off, or self-ship)" if english else "cách trả (nhận tại nhà, gửi tại điểm, hoặc tự gửi)",
    }
    requested = ", ".join(labels.get(field, field) for field in fields)
    return (
        f"Please provide {requested}."
        if english
        else f"Vui lòng cho mình biết {requested}."
    )


def _failed(shared, message: str, code: str) -> dict:
    shared.response.status = "failed"
    shared.response.message = message
    return {
        "shared": shared,
        "child_results": {
            "submit_return": ChildAgentResult(
                status="failed", message=message, error_code=code
            )
        },
    }
