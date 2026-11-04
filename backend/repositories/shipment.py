"""Repository for outbound and return shipments."""

from backend.models import Shipment
from backend.repositories.base import BaseRepository


class ShipmentRepository(BaseRepository[Shipment]):
    model = Shipment
