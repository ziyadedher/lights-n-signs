"""Generic data structures for keeping track of trainer settings."""

from typing import TypeVar

from dataclasses import dataclass


@dataclass(frozen=True)
class Settings():
    """Encapsulates all settings for a specific trainer."""

    # Just used to make sure the dataclass is never empty
    __dummy_field: bool = True


SettingsType = TypeVar("SettingsType", bound=Settings)
