from decimal import Decimal
import unittest

from sqlalchemy import create_engine, event
from sqlalchemy.orm import Session, configure_mappers

from backend.models import (
    Base,
    Order,
    OrderItem,
    Payment,
    Product,
    Refund,
    ReturnItem,
    ReturnRequest,
    Shipment,
    User,
    UserProfile,
)
from backend.models.enums import PaymentMethod, ReturnMethod, ReturnReason, ShipmentDirection


class BackendModelsTest(unittest.TestCase):
    def test_models_create_schema_and_configure_relationships(self) -> None:
        configure_mappers()
        engine = create_engine("sqlite:///:memory:")

        @event.listens_for(engine, "connect")
        def enable_foreign_keys(connection, _record) -> None:
            connection.execute("PRAGMA foreign_keys=ON")

        Base.metadata.create_all(engine)

        self.assertTrue(
            {
                "users",
                "user_profiles",
                "addresses",
                "products",
                "orders",
                "order_items",
                "payments",
                "shipments",
                "return_requests",
                "return_items",
                "return_status_history",
                "refunds",
            }.issubset(Base.metadata.tables)
        )

        with Session(engine) as session:
            user = User(
                email="customer@example.com",
                password_hash="hashed-password",
                profile=UserProfile(full_name="Customer"),
            )
            product = Product(sku="SKU-1", name="Product")
            order = Order(
                order_number="ORD-1",
                user=user,
                subtotal=Decimal("100.00"),
                total_amount=Decimal("100.00"),
            )
            item = OrderItem(
                order=order,
                product=product,
                product_name_snapshot="Product",
                sku_snapshot="SKU-1",
                quantity=1,
                unit_price=Decimal("100.00"),
                line_total=Decimal("100.00"),
            )
            payment = Payment(
                order=order,
                provider="manual",
                method=PaymentMethod.BANK_TRANSFER,
                amount=Decimal("100.00"),
            )
            return_request = ReturnRequest(
                return_number="RET-1",
                order=order,
                user=user,
                reason_code=ReturnReason.DAMAGED,
                return_method=ReturnMethod.SELF_SHIP,
            )
            return_request.items.append(ReturnItem(order_item=item, requested_quantity=1))
            return_request.shipments.append(
                Shipment(direction=ShipmentDirection.RETURN, provider="courier")
            )
            refund = Refund(
                return_request=return_request,
                payment=payment,
                amount=Decimal("100.00"),
            )
            session.add_all([user, product, order, payment, return_request, refund])
            session.commit()

            self.assertEqual(order.items[0].product.name, "Product")
            self.assertIs(return_request.refunds[0].payment, payment)


if __name__ == "__main__":
    unittest.main()
