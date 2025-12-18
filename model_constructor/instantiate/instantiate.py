from __future__ import annotations

import importlib
from typing import Any, Callable

import torch

from ..config.settings import Settings
from ..config.source_map import SourceMap
from ..errors import ConfigError
from ..registry.registry import Registry
from .signature import validate_kwargs

_RESERVED = {"_type_", "_target_", "_args_", "_kwargs_", "_name_"}


def instantiate_module_spec(
    spec: Any,
    *,
    registry: Registry,
    settings: Settings,
    source_map: SourceMap,
    config_path: tuple[Any, ...],
) -> torch.nn.Module:
    obj = instantiate_value(
        spec,
        registry=registry,
        settings=settings,
        source_map=source_map,
        config_path=config_path,
    )
    if not isinstance(obj, torch.nn.Module):
        raise ConfigError(
            f"Spec did not produce a torch.nn.Module (got {type(obj).__name__})",
            config_path=config_path,
            location=source_map.get(config_path),
        )
    return obj


def instantiate_value(
    value: Any,
    *,
    registry: Registry,
    settings: Settings,
    source_map: SourceMap,
    config_path: tuple[Any, ...],
) -> Any:
    if _is_spec(value):
        return _instantiate_spec(
            value,
            registry=registry,
            settings=settings,
            source_map=source_map,
            config_path=config_path,
        )

    if isinstance(value, dict):
        return {
            k: instantiate_value(
                v,
                registry=registry,
                settings=settings,
                source_map=source_map,
                config_path=config_path + (k,),
            )
            for k, v in value.items()
        }

    if isinstance(value, list):
        return [
            instantiate_value(
                v,
                registry=registry,
                settings=settings,
                source_map=source_map,
                config_path=config_path + (i,),
            )
            for i, v in enumerate(value)
        ]

    return value


def _is_spec(obj: Any) -> bool:
    return isinstance(obj, dict) and ("_type_" in obj or "_target_" in obj)


def _instantiate_spec(
    spec: dict[str, Any],
    *,
    registry: Registry,
    settings: Settings,
    source_map: SourceMap,
    config_path: tuple[Any, ...],
) -> Any:
    if "_type_" in spec and "_target_" in spec:
        raise ConfigError("Spec may not contain both _type_ and _target_", config_path=config_path)

    args = spec.get("_args_", [])
    if args is None:
        args = []
    if not isinstance(args, list):
        raise ConfigError("_args_ must be a list", config_path=config_path + ("_args_",))

    kwargs_inline = {k: v for k, v in spec.items() if k not in _RESERVED and not k.startswith("__")}
    kwargs_extra = spec.get("_kwargs_", {})
    if kwargs_extra is None:
        kwargs_extra = {}
    if not isinstance(kwargs_extra, dict):
        raise ConfigError("_kwargs_ must be a mapping", config_path=config_path + ("_kwargs_",))
    overlap = set(kwargs_inline) & set(kwargs_extra)
    if overlap:
        raise ConfigError(f"Duplicate kwargs keys: {sorted(overlap)}", config_path=config_path)

    kwargs = dict(kwargs_extra)
    kwargs.update(kwargs_inline)

    if settings.strict:
        unknown_reserved = {k for k in spec.keys() if k.startswith("_") and k not in _RESERVED}
        if unknown_reserved:
            raise ConfigError(f"Unknown reserved keys: {sorted(unknown_reserved)}", config_path=config_path)

    inst_args = [
        instantiate_value(
            a,
            registry=registry,
            settings=settings,
            source_map=source_map,
            config_path=config_path + ("_args_", i),
        )
        for i, a in enumerate(args)
    ]
    inst_kwargs = {
        k: instantiate_value(
            v,
            registry=registry,
            settings=settings,
            source_map=source_map,
            config_path=config_path + (k,),
        )
        for k, v in kwargs.items()
    }

    target = _resolve_target(spec, registry=registry, settings=settings, source_map=source_map, config_path=config_path)
    validate_kwargs(
        target,
        kwargs=inst_kwargs,
        policy=_signature_policy(spec, registry, config_path=config_path),
        config_path=config_path,
    )
    try:
        return target(*inst_args, **inst_kwargs)
    except Exception as exc:
        raise ConfigError(f"Instantiation failed: {exc}", config_path=config_path, location=source_map.get(config_path)) from exc


def _signature_policy(spec: dict[str, Any], registry: Registry, *, config_path: tuple[Any, ...]) -> str:
    if "_type_" in spec:
        entry = registry.get_module(spec["_type_"], config_path=config_path + ("_type_",))
        return entry.signature_policy
    return "runtime_only"


def _resolve_target(
    spec: dict[str, Any],
    *,
    registry: Registry,
    settings: Settings,
    source_map: SourceMap,
    config_path: tuple[Any, ...],
) -> Callable[..., Any]:
    if "_type_" in spec:
        t = spec.get("_type_")
        if not isinstance(t, str) or not t:
            raise ConfigError("_type_ must be a non-empty string", config_path=config_path + ("_type_",))
        return registry.get_module(t, config_path=config_path + ("_type_",)).target

    target = spec.get("_target_")
    if not isinstance(target, str) or not target:
        raise ConfigError("_target_ must be a non-empty string", config_path=config_path + ("_target_",))
    if not settings.allow_target:
        raise ConfigError("_target_ is disabled by settings.allow_target", config_path=config_path + ("_target_",))

    module_name, _, attr = target.rpartition(".")
    if not module_name or not attr:
        raise ConfigError("Invalid _target_ import path", config_path=config_path + ("_target_",))

    if not any(module_name.startswith(prefix) for prefix in settings.allowed_import_prefixes):
        raise ConfigError(
            f"_target_ import {target!r} is not allowed by settings.allowed_import_prefixes",
            config_path=config_path + ("_target_",),
            suggestions=list(settings.allowed_import_prefixes),
        )

    try:
        mod = importlib.import_module(module_name)
    except Exception as exc:
        raise ConfigError(f"Failed to import module {module_name!r}: {exc}", config_path=config_path + ("_target_",)) from exc

    if not hasattr(mod, attr):
        raise ConfigError(f"Module {module_name!r} has no attribute {attr!r}", config_path=config_path + ("_target_",))
    return getattr(mod, attr)
