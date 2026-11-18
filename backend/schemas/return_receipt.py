"""HTTP contract for recording receipt of approved return items."""

from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ReturnReceiptRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    note: str = Field(min_length=1, max_length=1000)

    @field_validator("note")
    @classmethod
    def note_must_contain_text(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("note must contain non-whitespace text")
        return cleaned


class ReturnReceiptResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    return_request_id: UUID
    return_status: str
    replayed: bool = False
