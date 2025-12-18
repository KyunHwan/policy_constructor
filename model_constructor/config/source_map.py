from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from ..errors import SourceLocation

Path = tuple[Any, ...]


class SourceMap:
    def __init__(self, mapping: dict[Path, SourceLocation] | None = None) -> None:
        self._map: dict[Path, SourceLocation] = dict(mapping or {})

    def get(self, path: Path) -> SourceLocation | None:
        return self._map.get(path)

    def items(self):
        return self._map.items()

    def submap(self, prefix: Path) -> "SourceMap":
        if not prefix:
            return SourceMap(self._map)
        n = len(prefix)
        return SourceMap({p: loc for p, loc in self._map.items() if p[:n] == prefix})

    def without_prefix(self, prefix: Path) -> "SourceMap":
        if not prefix:
            return SourceMap()
        n = len(prefix)
        return SourceMap({p: loc for p, loc in self._map.items() if p[:n] != prefix})

    def remap_prefix(self, old_prefix: Path, new_prefix: Path) -> "SourceMap":
        n = len(old_prefix)
        out: dict[Path, SourceLocation] = {}
        for p, loc in self._map.items():
            if p[:n] == old_prefix:
                out[new_prefix + p[n:]] = loc
        return SourceMap(out)

    def to_dict(self) -> dict[Path, SourceLocation]:
        return dict(self._map)


def delete_prefix(mapping: dict[Path, SourceLocation], prefix: Path) -> dict[Path, SourceLocation]:
    if not prefix:
        return {}
    n = len(prefix)
    return {p: loc for p, loc in mapping.items() if p[:n] != prefix}


def extract_prefix(mapping: dict[Path, SourceLocation], prefix: Path) -> dict[Path, SourceLocation]:
    n = len(prefix)
    return {p: loc for p, loc in mapping.items() if p[:n] == prefix}

