"""Generic data structures for keeping track of trainer settings."""

from typing import TypeVar

from dataclasses import dataclass


@dataclass(frozen=True)
class Settings():
    """Encapsulates all settings for a specific trainer."""


SettingsType = TypeVar("SettingsType", bound=Settings)
