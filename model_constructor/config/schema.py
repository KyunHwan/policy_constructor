from __future__ import annotations

from typing import Any

from ..errors import ConfigError
from .settings import Settings, parse_settings
from .source_map import SourceMap


def validate_schema_v1(config: Any, *, source_map: SourceMap) -> Settings:
    if not isinstance(config, dict):
        raise ConfigError("config must be a mapping", config_path=(), location=source_map.get(()))

    schema_version = config.get("schema_version")
    if schema_version != 1:
        raise ConfigError(
            "schema_version must be 1",
            config_path=("schema_version",),
            location=source_map.get(("schema_version",)) or source_map.get(()),
        )

    if "model" not in config:
        raise ConfigError("missing required key 'model'", config_path=("model",), location=source_map.get(()))

    if not isinstance(config.get("model"), dict):
        raise ConfigError("'model' must be a mapping", config_path=("model",), location=source_map.get(("model",)))

    settings = parse_settings(config.get("settings"))

    imports = config.get("imports", [])
    if imports is None:
        imports = []
        config["imports"] = imports
    if imports and (not isinstance(imports, list) or not all(isinstance(x, str) for x in imports)):
        raise ConfigError("'imports' must be a list of strings", config_path=("imports",), location=source_map.get(("imports",)))

    return settings
