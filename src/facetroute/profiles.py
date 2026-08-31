"""Local persistence for user routing preferences."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from .errors import ConfigurationError, PersistenceError
from .persistence import AtomicJsonStore
from .types import UserPreferences


class PreferenceStore:
    """Versioned user-profile registry backed by one local JSON file."""

    schema_version = 1

    def __init__(self, path: str | Path) -> None:
        self.store = AtomicJsonStore(path)

    def load_all(self) -> dict[str, UserPreferences]:
        payload = self.store.load({"schema_version": self.schema_version, "profiles": []})
        if not isinstance(payload, dict) or payload.get("schema_version") != self.schema_version:
            raise PersistenceError("Unsupported preference-store schema")
        records = payload.get("profiles", [])
        if not isinstance(records, list):
            raise PersistenceError("Preference store profiles must be a list")
        try:
            profiles = [UserPreferences.from_dict(item) for item in records]
        except ConfigurationError as exc:
            raise PersistenceError(f"Invalid preference-store profile: {exc}") from exc
        result = {profile.user_id: profile for profile in profiles}
        if len(result) != len(profiles):
            raise PersistenceError("Preference store contains duplicate user_id values")
        return result

    def save_all(self, profiles: Iterable[UserPreferences]) -> None:
        profile_list = sorted(profiles, key=lambda item: item.user_id)
        if len({item.user_id for item in profile_list}) != len(profile_list):
            raise PersistenceError("Cannot save duplicate user_id values")
        self.store.save(
            {
                "schema_version": self.schema_version,
                "profiles": [profile.to_dict() for profile in profile_list],
            }
        )

    def get(self, user_id: str) -> UserPreferences | None:
        return self.load_all().get(user_id)

    def upsert(self, profile: UserPreferences) -> None:
        profiles = self.load_all()
        profiles[profile.user_id] = profile
        self.save_all(profiles.values())
