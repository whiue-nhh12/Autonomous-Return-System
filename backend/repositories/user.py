"""Repositories for user accounts, profiles, and addresses."""

from backend.models import Address, User, UserProfile
from backend.repositories.base import BaseRepository


class UserRepository(BaseRepository[User]):
    model = User

    def get_by_email(self, email: str) -> User | None:
        return self.find_one_by(email=email)


class UserProfileRepository(BaseRepository[UserProfile]):
    model = UserProfile


class AddressRepository(BaseRepository[Address]):
    model = Address
