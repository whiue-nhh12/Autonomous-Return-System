"""HTTP contracts for initiating a refund after return receipt."""

from decimal import Decimal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator


class RefundInitiationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    amount: Decimal = Field(gt=0, max_digits=12, decimal_places=2)
    payment_id: UUID | None = None
    note: str = Field(min_length=1, max_length=1000)

    @field_validator("note")
    @classmethod
    def note_must_contain_text(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("note must contain non-whitespace text")
        return cleaned


class RefundInitiationResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    return_request_id: UUID
    refund_id: UUID
    return_status: str
    refund_status: str
    amount: Decimal
    currency: str
    replayed: bool = False
