import unittest
from unittest.mock import patch

from sqlalchemy import create_engine

from backend import db as db_module
from backend.db import create_session_factory, init_db
from backend.models import Base, Product
from backend.repositories import (
    AddressRepository,
    OrderItemRepository,
    OrderRepository,
    PaymentRepository,
    ProductRepository,
    RefundRepository,
    ReturnItemRepository,
    ReturnRequestRepository,
    ReturnStatusHistoryRepository,
    ShipmentRepository,
    UserProfileRepository,
    UserRepository,
)


class BackendDatabaseTest(unittest.TestCase):
    def setUp(self) -> None:
        self.engine = create_engine("sqlite:///:memory:")
        init_db(self.engine)
        self.session_factory = create_session_factory(self.engine)

    def tearDown(self) -> None:
        self.engine.dispose()

    def test_repository_create_find_update_list_and_delete(self) -> None:
        with self.session_factory.begin() as session:
            products = ProductRepository(session)
            product = products.create(sku="SKU-1", name="Original")

            self.assertIs(products.get_by_sku("SKU-1"), product)
            self.assertEqual(products.list(), [product])

            updated = products.update(product.id, name="Updated")
            self.assertEqual(updated.name, "Updated")
            self.assertTrue(products.delete(product.id))
            self.assertIsNone(products.get(product.id))

    def test_repositories_use_caller_managed_session(self) -> None:
        with self.session_factory() as session:
            repository = UserRepository(session)
            user = repository.create(email="buyer@example.com", password_hash="hashed")
            self.assertIs(repository.get_by_email("buyer@example.com"), user)
            self.assertTrue(session.in_transaction())
            session.rollback()
            self.assertIsNone(repository.get_by_email("buyer@example.com"))

    def test_repository_rejects_unknown_columns_and_invalid_pagination(self) -> None:
        with self.session_factory() as session:
            products = ProductRepository(session)
            with self.assertRaisesRegex(ValueError, "Unknown Product column"):
                products.find_one_by(not_a_column="value")
            with self.assertRaisesRegex(ValueError, "limit"):
                products.list(limit=0)
            with self.assertRaisesRegex(ValueError, "offset"):
                products.list(offset=-1)

    def test_all_backend_models_are_registered(self) -> None:
        expected_tables = {
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
        }
        self.assertTrue(expected_tables.issubset(Base.metadata.tables))

    def test_repository_classes_cover_all_backend_models(self) -> None:
        with self.session_factory() as session:
            repositories = (
                (UserRepository, "users"),
                (UserProfileRepository, "user_profiles"),
                (AddressRepository, "addresses"),
                (ProductRepository, "products"),
                (OrderRepository, "orders"),
                (OrderItemRepository, "order_items"),
                (PaymentRepository, "payments"),
                (ShipmentRepository, "shipments"),
                (ReturnRequestRepository, "return_requests"),
                (ReturnItemRepository, "return_items"),
                (ReturnStatusHistoryRepository, "return_status_history"),
                (RefundRepository, "refunds"),
            )
            for repository_class, table_name in repositories:
                with self.subTest(table=table_name):
                    self.assertEqual(repository_class(session).model.__tablename__, table_name)

    def test_environment_configured_engine_and_request_session(self) -> None:
        previous_engine = db_module.session._engine
        previous_factory = db_module.session._session_factory
        db_module.session._engine = None
        db_module.session._session_factory = None

        try:
            with patch.dict("os.environ", {"DATABASE_URL": "sqlite:///:memory:"}):
                engine = db_module.get_engine()
                self.assertEqual(engine.dialect.name, "sqlite")
                self.assertIs(db_module.get_engine(), engine)

                factory = db_module.get_session_factory()
                self.assertIs(factory.kw["bind"], engine)

                request_session = db_module.get_db()
                session = next(request_session)
                self.assertIsNotNone(session)
                request_session.close()
        finally:
            db_module.session._engine = previous_engine
            db_module.session._session_factory = previous_factory


if __name__ == "__main__":
    unittest.main()
