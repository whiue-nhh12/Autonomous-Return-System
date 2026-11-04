"""Shared CRUD operations for SQLAlchemy repositories."""

from typing import Any, Generic, TypeVar

from sqlalchemy import inspect, select
from sqlalchemy.orm import Session

from backend.models.base import Base

ModelT = TypeVar("ModelT", bound=Base)


class BaseRepository(Generic[ModelT]):
    """CRUD operations that flush changes but leave transactions to the caller."""

    model: type[ModelT]

    def __init__(self, session: Session) -> None:
        self.session = session

    def create(self, **values: Any) -> ModelT:
        instance = self.model(**values)
        self.session.add(instance)
        self.session.flush()
        return instance

    def get(self, entity_id: Any) -> ModelT | None:
        return self.session.get(self.model, entity_id)

    def list(self, *, limit: int = 100, offset: int = 0) -> list[ModelT]:
        if limit <= 0:
            raise ValueError("limit must be greater than zero")
        if offset < 0:
            raise ValueError("offset cannot be negative")

        primary_key = inspect(self.model).primary_key[0]
        statement = (
            select(self.model)
            .order_by(primary_key)
            .limit(limit)
            .offset(offset)
        )
        return list(self.session.scalars(statement))

    def find_one_by(self, **criteria: Any) -> ModelT | None:
        if not criteria:
            raise ValueError("At least one lookup criterion is required")

        mapper = inspect(self.model)
        invalid_fields = set(criteria).difference(mapper.column_attrs.keys())
        if invalid_fields:
            fields = ", ".join(sorted(invalid_fields))
            raise ValueError(f"Unknown {self.model.__name__} column(s): {fields}")

        conditions = [
            getattr(self.model, field_name).is_(None)
            if value is None
            else getattr(self.model, field_name) == value
            for field_name, value in criteria.items()
        ]
        return self.session.scalar(select(self.model).where(*conditions).limit(1))

    def update(self, entity_id: Any, **values: Any) -> ModelT | None:
        instance = self.get(entity_id)
        if instance is None:
            return None

        mapper = inspect(self.model)
        invalid_fields = set(values).difference(mapper.column_attrs.keys())
        if invalid_fields:
            fields = ", ".join(sorted(invalid_fields))
            raise ValueError(f"Unknown {self.model.__name__} column(s): {fields}")

        primary_key_names = {column.key for column in mapper.primary_key}
        if primary_key_names.intersection(values):
            raise ValueError("Primary key fields cannot be updated")

        for field_name, value in values.items():
            setattr(instance, field_name, value)
        self.session.flush()
        return instance

    def delete(self, entity_id: Any) -> bool:
        instance = self.get(entity_id)
        if instance is None:
            return False
        self.session.delete(instance)
        self.session.flush()
        return True
