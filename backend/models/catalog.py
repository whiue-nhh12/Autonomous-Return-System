"""Product catalog model."""

from typing import TYPE_CHECKING

from sqlalchemy import String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.models.base import Base, TimestampMixin, UUIDPrimaryKeyMixin
from backend.models.enums import ProductStatus, enum_column


class Product(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "products"

    sku: Mapped[str] = mapped_column(String(100), nullable=False, unique=True, index=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    image_url: Mapped[str | None] = mapped_column(String(2048), nullable=True)
    status: Mapped[ProductStatus] = mapped_column(
        enum_column(ProductStatus, "product_status"),
        nullable=False,
        default=ProductStatus.ACTIVE,
        server_default=ProductStatus.ACTIVE.value,
    )

    order_items: Mapped[list["OrderItem"]] = relationship(back_populates="product")


if TYPE_CHECKING:
    from backend.models.order import OrderItem
