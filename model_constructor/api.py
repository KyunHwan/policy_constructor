from __future__ import annotations

from pathlib import Path
from typing import Any

from .compat import check_compatibility
from .config.resolve import ResolvedConfig, resolve_config as resolve_config_impl
from .graph.compiler import compile_ir as compile_graph_ir
from .graph.model import GraphModel
from .instantiate.instantiate import instantiate_module_spec
from .registry.default_registry import get_default_registry
from .registry.registry import Registry
from .util.imports import apply_imports


def build_model(config_or_path: dict[str, Any] | str | Path, *, registry: Registry | None = None) -> GraphModel:
    check_compatibility()

    active_registry = registry or get_default_registry()
    resolved = resolve_config_impl(config_or_path)

    imports = resolved.data.get("imports", [])
    settings = resolved.settings
    apply_imports(imports, settings=settings, registry=active_registry, source_map=resolved.source_map)

    ir = compile_graph_ir(resolved.data, source_map=resolved.source_map, registry=active_registry)

    modules = {
        name: instantiate_module_spec(
            module_ir.spec,
            registry=active_registry,
            settings=settings,
            source_map=resolved.source_map,
            config_path=module_ir.origin_path,
        )
        for name, module_ir in ir.modules.items()
    }

    return GraphModel(
        inputs=ir.inputs,
        modules=modules,
        steps=ir.steps,
        outputs=ir.outputs,
        return_policy=ir.return_policy,
        ops=ir.ops,
    )


def compile_ir(config_or_path: dict[str, Any] | str | Path, *, registry: Registry | None = None):
    active_registry = registry or get_default_registry()
    resolved = resolve_config_impl(config_or_path)

    imports = resolved.data.get("imports", [])
    settings = resolved.settings
    apply_imports(imports, settings=settings, registry=active_registry, source_map=resolved.source_map)

    return compile_graph_ir(resolved.data, source_map=resolved.source_map, registry=active_registry)


def resolve_config(config_or_path: dict[str, Any] | str | Path) -> ResolvedConfig:
    return resolve_config_impl(config_or_path)
