from __future__ import annotations

import importlib
from typing import Any

from ..config.settings import Settings
from ..config.source_map import SourceMap
from ..errors import ConfigError
from ..registry.registry import Registry


def apply_imports(
    imports: Any,
    *,
    settings: Settings,
    registry: Registry,
    source_map: SourceMap,
) -> None:
    if not imports:
        return

    if not settings.allow_imports:
        raise ConfigError(
            "imports are disabled by settings",
            config_path=("imports",),
            location=source_map.get(("imports",)),
        )

    if not isinstance(imports, list) or not all(isinstance(x, str) for x in imports):
        raise ConfigError(
            "imports must be a list of strings",
            config_path=("imports",),
            location=source_map.get(("imports",)),
        )

    for i, mod_name in enumerate(imports):
        path = ("imports", i)
        if not any(mod_name.startswith(prefix) for prefix in settings.allowed_import_prefixes):
            raise ConfigError(
                f"import {mod_name!r} is not allowed by settings.allowed_import_prefixes",
                config_path=path,
                location=source_map.get(path),
                suggestions=list(settings.allowed_import_prefixes),
            )

        try:
            mod = importlib.import_module(mod_name)
        except Exception as exc:
            raise ConfigError(
                f"failed to import {mod_name!r}: {exc}",
                config_path=path,
                location=source_map.get(path),
            ) from exc

        register_fn = getattr(mod, "register", None)
        if callable(register_fn):
            register_fn(registry)
