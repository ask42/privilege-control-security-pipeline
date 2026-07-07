from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from collections.abc import Mapping, Sequence
from typing import Any

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

def _serialize_action_descriptor(action: Any) -> dict[str, Any]:
    """Normalize action inputs into a JSON-safe descriptor for the scoping prompt."""
    if isinstance(action, str):
        return {"action_name": action, "metadata": {}}

    if isinstance(action, Mapping):
        action_name = action.get("action_name") or action.get("name") or action.get("tool_name")
        if action_name is None:
            raise ValueError("Action mapping is missing an action name")

        metadata: dict[str, Any] = {}
        nested_metadata = action.get("metadata")
        if isinstance(nested_metadata, Mapping):
            metadata.update(nested_metadata)

        for key in ("description", "parameter_names", "required_parameters", "dependencies", "docstring", "schema"):
            value = action.get(key)
            if value is not None:
                metadata[key] = value

        return {"action_name": action_name, "metadata": metadata}

    action_name = getattr(action, "action_name", None) or getattr(action, "name", None)
    if action_name is None:
        raise ValueError(f"Unsupported action descriptor: {action!r}")

    metadata: dict[str, Any] = {}

    description = getattr(action, "description", None)
    if description is not None:
        metadata["description"] = description

    parameters = getattr(action, "parameters", None)
    if parameters is not None:
        if hasattr(parameters, "model_json_schema"):
            schema = parameters.model_json_schema()
            metadata["parameter_names"] = sorted(schema.get("properties", {}).keys())
            metadata["required_parameters"] = list(schema.get("required", []))
        elif hasattr(parameters, "schema"):
            schema = parameters.schema()
            metadata["parameter_names"] = sorted(schema.get("properties", {}).keys())
            metadata["required_parameters"] = list(schema.get("required", []))
        else:
            metadata["parameter_names"] = []
            metadata["required_parameters"] = []

    dependencies = getattr(action, "dependencies", None)
    if dependencies is not None:
        if isinstance(dependencies, Mapping):
            metadata["dependencies"] = sorted(dependencies.keys())
        elif isinstance(dependencies, Sequence) and not isinstance(dependencies, (str, bytes)):
            metadata["dependencies"] = list(dependencies)
        else:
            metadata["dependencies"] = str(dependencies)

    extra_metadata = getattr(action, "metadata", None)
    if isinstance(extra_metadata, Mapping):
        for key, value in extra_metadata.items():
            metadata.setdefault(key, value)

    return {"action_name": action_name, "metadata": metadata}

