from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..errors import ConfigError
from .interpolate import interpolate_config
from .merge import merge_values
from .schema import validate_schema_v1
from .settings import Settings
from .source_map import Path as ConfigPath
from .source_map import SourceMap
from .templates import expand_templates
from .yaml_loader import load_yaml_file


@dataclass(frozen=True)
class ResolvedConfig:
    data: dict[str, Any]
    source_map: SourceMap
    settings: Settings
    root_file: str | None = None


def resolve_config(config_or_path: dict[str, Any] | str | Path) -> ResolvedConfig:
    if isinstance(config_or_path, (str, Path)):
        root_path = Path(config_or_path)
        data, sm = _resolve_from_file(root_path, include_stack=[])
        root_file = str(root_path)
    elif isinstance(config_or_path, dict):
        if "defaults" in config_or_path:
            raise ConfigError("defaults are only supported when loading from a file path", config_path=("defaults",))
        data = dict(config_or_path)
        sm = SourceMap()
        root_file = None
    else:
        raise ConfigError("config_or_path must be a dict or a file path", config_path=())

    # Consume templates
    data, sm = expand_templates(data, source_map=sm)

    # Interpolation
    data, sm = interpolate_config(data, source_map=sm)

    settings = validate_schema_v1(data, source_map=sm)
    return ResolvedConfig(data=data, source_map=sm, settings=settings, root_file=root_file)


def _resolve_from_file(path: Path, *, include_stack: list[str]) -> tuple[dict[str, Any], SourceMap]:
    p = path.resolve()
    ps = str(p)
    if ps in include_stack:
        raise ConfigError(f"defaults include cycle detected: {' -> '.join(include_stack + [ps])}", include_stack=include_stack + [ps])

    include_stack = include_stack + [ps]
    data, sm = load_yaml_file(p)

    defaults = data.get("defaults", [])
    if defaults is None:
        defaults = []
    if defaults and (not isinstance(defaults, list) or not all(isinstance(x, str) for x in defaults)):
        raise ConfigError("'defaults' must be a list of strings", config_path=("defaults",), location=sm.get(("defaults",)))

    merged_data: dict[str, Any] = {}
    merged_sm = SourceMap()

    for entry in defaults:
        child_path = (p.parent / entry).resolve()
        child_data, child_sm = _resolve_from_file(child_path, include_stack=include_stack)
        merged_data, merged_sm = merge_values(
            merged_data,
            child_data,
            base_sm=merged_sm,
            override_sm=child_sm,
            path=(),
        )

    # Overlay current file (without defaults key)
    current = {k: v for k, v in data.items() if k != "defaults"}
    merged_data, merged_sm = merge_values(merged_data, current, base_sm=merged_sm, override_sm=sm, path=())
    return merged_data, merged_sm

