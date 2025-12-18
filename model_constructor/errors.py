from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class SourceLocation:
    file: str
    line: int
    col: int

    def format(self) -> str:
        return f"{self.file}:{self.line}:{self.col}"


class ModelConstructorError(Exception):
    pass


class CompatibilityError(ModelConstructorError):
    pass


class ConfigError(ModelConstructorError):
    def __init__(
        self,
        message: str,
        *,
        config_path: tuple[Any, ...] | None = None,
        location: SourceLocation | None = None,
        include_stack: list[str] | None = None,
        suggestions: list[str] | None = None,
    ) -> None:
        self.message = message
        self.config_path = config_path
        self.location = location
        self.include_stack = include_stack or []
        self.suggestions = suggestions or []

        parts = [message]
        if config_path is not None:
            parts.append(f"path={_format_path(config_path)}")
        if location is not None:
            parts.append(f"loc={location.format()}")
        if self.include_stack:
            parts.append("includes=" + " -> ".join(self.include_stack))
        if self.suggestions:
            parts.append("suggestions=" + ", ".join(self.suggestions))
        super().__init__(" | ".join(parts))


class RegistryError(ModelConstructorError):
    pass


class InstantiationError(ModelConstructorError):
    pass


class GraphCompileError(ModelConstructorError):
    pass


class GraphExecutionError(ModelConstructorError):
    pass


def _format_path(path: tuple[Any, ...]) -> str:
    out = ""
    for seg in path:
        if isinstance(seg, int):
            out += f"[{seg}]"
        else:
            if out:
                out += "."
            out += str(seg)
    return out or "<root>"

