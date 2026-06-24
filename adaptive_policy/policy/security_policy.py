from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Allowed:
    """Policy allows the action."""
    pass


@dataclass(frozen=True)
class Denied:
    """Policy denies the action."""
    reason: str


PolicyResult = Allowed | Denied
