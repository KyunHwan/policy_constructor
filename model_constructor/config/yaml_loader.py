from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from ..errors import ConfigError, SourceLocation
from .source_map import Path as ConfigPath
from .source_map import SourceMap


_STR_TAG = "tag:yaml.org,2002:str"


class _StrictKeyLoader(yaml.SafeLoader):
    def construct_mapping(self, node, deep: bool = False):  # type: ignore[override]
        mapping = {}
        for key_node, value_node in node.value:
            if not isinstance(key_node, yaml.ScalarNode) or key_node.tag != _STR_TAG:
                mark = getattr(key_node, "start_mark", None)
                loc = None
                if mark is not None:
                    loc = SourceLocation(file="<yaml>", line=mark.line + 1, col=mark.column + 1)
                raise ConfigError(
                    "YAML mapping keys must be strings",
                    location=loc,
                )
            key = key_node.value
            if key in mapping:
                mark = getattr(key_node, "start_mark", None)
                loc = None
                if mark is not None:
                    loc = SourceLocation(file="<yaml>", line=mark.line + 1, col=mark.column + 1)
                raise ConfigError(
                    f"Duplicate mapping key {key!r}",
                    location=loc,
                )
            mapping[key] = self.construct_object(value_node, deep=deep)
        return mapping


def load_yaml_file(path: str | Path) -> tuple[dict[str, Any], SourceMap]:
    p = Path(path)
    text = p.read_text(encoding="utf-8")

    try:
        root = yaml.compose(text)
    except yaml.YAMLError as exc:
        loc = None
        mark = getattr(exc, "problem_mark", None)
        if mark is not None:
            loc = SourceLocation(file=str(p), line=mark.line + 1, col=mark.column + 1)
        raise ConfigError(f"Invalid YAML: {exc}", location=loc) from exc

    if root is None:
        raise ConfigError("Empty YAML document", location=SourceLocation(file=str(p), line=1, col=1))

    try:
        data = yaml.load(text, Loader=_StrictKeyLoader)
    except yaml.YAMLError as exc:
        loc = None
        mark = getattr(exc, "problem_mark", None)
        if mark is not None:
            loc = SourceLocation(file=str(p), line=mark.line + 1, col=mark.column + 1)
        raise ConfigError(f"Invalid YAML: {exc}", location=loc) from exc

    if not isinstance(data, dict):
        loc = _node_loc(root, str(p))
        raise ConfigError("Top-level YAML must be a mapping", location=loc)

    source_map = SourceMap(_build_source_map(root, file=str(p)))
    return data, source_map


def _node_loc(node: yaml.Node, file: str) -> SourceLocation:
    mark = getattr(node, "start_mark", None)
    if mark is None:  # pragma: no cover
        return SourceLocation(file=file, line=1, col=1)
    return SourceLocation(file=file, line=mark.line + 1, col=mark.column + 1)


def _build_source_map(node: yaml.Node, *, file: str) -> dict[ConfigPath, SourceLocation]:
    out: dict[ConfigPath, SourceLocation] = {}

    def walk(n: yaml.Node, path: ConfigPath) -> None:
        out[path] = _node_loc(n, file)

        if isinstance(n, yaml.MappingNode):
            for key_node, value_node in n.value:
                if isinstance(key_node, yaml.ScalarNode):
                    key = key_node.value
                else:  # pragma: no cover
                    key = str(key_node)
                walk(value_node, path + (key,))
        elif isinstance(n, yaml.SequenceNode):
            for i, child in enumerate(n.value):
                walk(child, path + (i,))

    walk(node, ())
    return out

