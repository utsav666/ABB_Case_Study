from __future__ import annotations


class SettingsDatabaseConfigurationError(ValueError):
    """Raised when the settings database environment is incomplete."""


class SettingsDatabaseQueryError(RuntimeError):
    """Raised when a settings database query cannot be completed."""
