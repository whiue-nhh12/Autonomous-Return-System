"""Enumerations used by backend persistence models."""

from enum import Enum

from sqlalchemy import Enum as SQLAlchemyEnum


class UserRole(str, Enum):
    CUSTOMER = "customer"
    ADMIN = "admin"
    AGENT = "agent"


class UserStatus(str, Enum):
    ACTIVE = "active"
    SUSPENDED = "suspended"
    PENDING_VERIFICATION = "pending_verification"


class ProductStatus(str, Enum):
    ACTIVE = "active"
    INACTIVE = "inactive"


class OrderStatus(str, Enum):
    PENDING = "pending"
    CONFIRMED = "confirmed"
    SHIPPED = "shipped"
    DELIVERED = "delivered"
    CANCELLED = "cancelled"


class PaymentStatus(str, Enum):
    PENDING = "pending"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


class PaymentMethod(str, Enum):
    CARD = "card"
    BANK_TRANSFER = "bank_transfer"
    COD = "cod"
    E_WALLET = "e_wallet"
    OTHER = "other"


class ShipmentDirection(str, Enum):
    OUTBOUND = "outbound"
    RETURN = "return"


class ShipmentStatus(str, Enum):
    PENDING = "pending"
    IN_TRANSIT = "in_transit"
    DELIVERED = "delivered"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ReturnStatus(str, Enum):
    SUBMITTED = "submitted"
    REVIEWING = "reviewing"
    APPROVED = "approved"
    REJECTED = "rejected"
    ITEM_RECEIVED = "item_received"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


class ReturnReason(str, Enum):
    DAMAGED = "damaged"
    WRONG_ITEM = "wrong_item"
    NOT_AS_DESCRIBED = "not_as_described"
    CHANGED_MIND = "changed_mind"
    OTHER = "other"


class ReturnMethod(str, Enum):
    PICKUP = "pickup"
    DROP_OFF = "drop_off"
    SELF_SHIP = "self_ship"


class DecisionSource(str, Enum):
    MANUAL = "manual"
    RULE = "rule"
    AI = "ai"


class ReturnItemCondition(str, Enum):
    UNOPENED = "unopened"
    OPENED = "opened"
    USED = "used"
    DAMAGED = "damaged"
    UNKNOWN = "unknown"


class RefundStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


class StatusChangeSource(str, Enum):
    USER = "user"
    STAFF = "staff"
    SYSTEM = "system"
    AI = "ai"


def enum_column(enum_class: type[Enum], name: str) -> SQLAlchemyEnum:
    return SQLAlchemyEnum(
        enum_class,
        name=name,
        native_enum=False,
        create_constraint=True,
        values_callable=lambda values: [item.value for item in values],
        validate_strings=True,
    )
