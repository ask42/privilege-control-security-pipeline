from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


class ProvenanceSource(str, Enum):
    USER = "user"
    TOOL = "tool"
    TRANSFORM = "transform"


@dataclass(frozen=True)
class Provenance:
    """Where a value came from and its origin context."""
    source: ProvenanceSource
    tool_name: str | None = None  # if source == TOOL
    inner_source: str | None = None  # email sender, file editor
    timestamp: float | None = None


@dataclass
class Value:
    """
    A value with provenance tracking, inspired by CaMeL's CaMeLValue.
    Tracks origin without full taint interpreter.
    """
    raw: Any
    provenance: Provenance
    readers: set[str] | None = None  # None = public, set = restricted to these users

    def is_user_sourced(self) -> bool:
        return self.provenance.source == ProvenanceSource.USER

    def is_tool_sourced(self) -> bool:
        return self.provenance.source == ProvenanceSource.TOOL

    def is_trusted(self) -> bool:
        return self.is_user_sourced()


def user_literal(val: Any) -> Value:
    """Value provided directly by the user."""
    return Value(val, Provenance(ProvenanceSource.USER))


def tool_result(val: Any, tool_name: str, inner_source: str | None = None) -> Value:
    """Value returned from a tool execution (untrusted)."""
    return Value(val, Provenance(
        ProvenanceSource.TOOL,
        tool_name=tool_name,
        inner_source=inner_source
    ))


def transformed(val: Any, from_value: Value) -> Value:
    """Value derived from transformation of another value."""
    return Value(
        val, 
        Provenance(
            ProvenanceSource.TRANSFORM, 
            tool_name=from_value.provenance.tool_name,
            inner_source=from_value.provenance.inner_source,
        )
    )
