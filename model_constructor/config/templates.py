from __future__ import annotations

from copy import deepcopy
from typing import Any

from ..errors import ConfigError
from .merge import merge_values
from .source_map import Path, SourceMap


def expand_templates(config: dict[str, Any], *, source_map: SourceMap) -> tuple[dict[str, Any], SourceMap]:
    templates = config.get("templates", {})
    if templates is None:
        templates = {}
    if templates and not isinstance(templates, dict):
        raise ConfigError("templates must be a mapping", config_path=("templates",), location=source_map.get(("templates",)))

    template_cache: dict[str, tuple[Any, SourceMap]] = {}

    def expand_template(name: str, stack: tuple[str, ...]) -> tuple[Any, SourceMap]:
        if name in template_cache:
            return template_cache[name]
        if name in stack:
            raise ConfigError(
                f"Template cycle detected: {' -> '.join(stack + (name,))}",
                config_path=("templates", name),
                location=source_map.get(("templates", name)),
            )
        if not isinstance(templates, dict) or name not in templates:
            raise ConfigError(
                f"Unknown template {name!r}",
                config_path=("templates", name),
                location=source_map.get(("templates", name)),
            )

        tpl_raw = templates[name]
        tpl_sm = source_map.submap(("templates", name))
        expanded, expanded_sm = expand_obj(tpl_raw, ("templates", name), tpl_sm, stack + (name,))
        template_cache[name] = (expanded, expanded_sm)
        return expanded, expanded_sm

    def expand_obj(obj: Any, path: Path, sm: SourceMap, stack: tuple[str, ...]) -> tuple[Any, SourceMap]:
        if isinstance(obj, dict) and "_template_" in obj:
            tpl_name = obj.get("_template_")
            if not isinstance(tpl_name, str) or not tpl_name:
                raise ConfigError(
                    "_template_ must be a non-empty string",
                    config_path=path + ("_template_",),
                    location=sm.get(path + ("_template_",)) or sm.get(path),
                )
            base_data, base_sm = expand_template(tpl_name, stack)
            override_data = {k: v for k, v in obj.items() if k != "_template_"}

            remapped_base_sm = base_sm.remap_prefix(("templates", tpl_name), path)
            merged_data, merged_sm = merge_values(
                deepcopy(base_data),
                override_data,
                base_sm=remapped_base_sm,
                override_sm=sm,
                path=path,
            )
            return expand_obj(merged_data, path, merged_sm, stack)

        if isinstance(obj, dict):
            out: dict[str, Any] = {}
            out_sm_map: dict[Path, Any] = {}
            loc = sm.get(path)
            if loc is not None:
                out_sm_map[path] = loc
            for k, v in obj.items():
                child, child_sm = expand_obj(v, path + (k,), sm, stack)
                out[k] = child
                out_sm_map.update(child_sm.to_dict())
            return out, SourceMap(out_sm_map)

        if isinstance(obj, list):
            out_list: list[Any] = []
            out_sm_map: dict[Path, Any] = {}
            loc = sm.get(path)
            if loc is not None:
                out_sm_map[path] = loc
            for i, item in enumerate(obj):
                child, child_sm = expand_obj(item, path + (i,), sm, stack)
                out_list.append(child)
                out_sm_map.update(child_sm.to_dict())
            return out_list, SourceMap(out_sm_map)

        loc = sm.get(path)
        return obj, SourceMap({path: loc} if loc is not None else {})

    expanded_root, expanded_sm = expand_obj(config, (), source_map, ())
    if not isinstance(expanded_root, dict):  # pragma: no cover
        raise ConfigError("internal error: expanded config is not a mapping", config_path=())

    expanded_root.pop("templates", None)
    expanded_sm = SourceMap({p: loc for p, loc in expanded_sm.items() if not (p and p[0] == "templates")})
    return expanded_root, expanded_sm

