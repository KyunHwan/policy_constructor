from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..errors import ConfigError


@dataclass(frozen=True)
class Settings:
    strict: bool = True
    allow_imports: bool = True
    allowed_import_prefixes: tuple[str, ...] = ("model_constructor.",)
    allow_target: bool = False
    error_context_lines: int = 2


def parse_settings(settings: Any) -> Settings:
    if settings is None:
        return Settings()
    if not isinstance(settings, dict):
        raise ConfigError("settings must be a mapping", config_path=("settings",))

    strict = _get_bool(settings, "strict", default=True)
    allow_imports = _get_bool(settings, "allow_imports", default=True)
    allow_target = _get_bool(settings, "allow_target", default=False)
    error_context_lines = _get_int(settings, "error_context_lines", default=2)

    allowed_import_prefixes_raw = settings.get("allowed_import_prefixes", ["model_constructor."])
    if not isinstance(allowed_import_prefixes_raw, list) or not all(
        isinstance(x, str) for x in allowed_import_prefixes_raw
    ):
        raise ConfigError(
            "settings.allowed_import_prefixes must be a list of strings",
            config_path=("settings", "allowed_import_prefixes"),
        )

    allowed_import_prefixes = tuple(allowed_import_prefixes_raw)
    return Settings(
        strict=strict,
        allow_imports=allow_imports,
        allowed_import_prefixes=allowed_import_prefixes,
        allow_target=allow_target,
        error_context_lines=error_context_lines,
    )


def _get_bool(d: dict[str, Any], key: str, *, default: bool) -> bool:
    val = d.get(key, default)
    if isinstance(val, bool):
        return val
    raise ConfigError(f"settings.{key} must be a bool", config_path=("settings", key))


def _get_int(d: dict[str, Any], key: str, *, default: int) -> int:
    val = d.get(key, default)
    if isinstance(val, int):
        return val
    raise ConfigError(f"settings.{key} must be an int", config_path=("settings", key))

