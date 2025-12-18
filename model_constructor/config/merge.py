from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from ..errors import ConfigError, SourceLocation
from .source_map import Path, SourceMap, delete_prefix, extract_prefix

MergeMode = Literal["replace", "append", "prepend", "keyed"]

_MERGE_KEY = "_merge_"
_VALUE_KEY = "_value_"


@dataclass(frozen=True)
class MergeDirective:
    mode: MergeMode
    key: str | None = None
    value: Any = None


def merge_values(
    base: Any,
    override: Any,
    *,
    base_sm: SourceMap,
    override_sm: SourceMap,
    path: Path,
) -> tuple[Any, SourceMap]:
    directive = _parse_merge_directive(override, override_sm, path)
    if directive is not None:
        return _apply_directive(base, directive, base_sm=base_sm, override_sm=override_sm, path=path)

    if isinstance(base, dict) and isinstance(override, dict):
        return _merge_dicts(base, override, base_sm=base_sm, override_sm=override_sm, path=path)

    if isinstance(override, list):
        return override, SourceMap(extract_prefix(override_sm.to_dict(), path))

    return override, SourceMap(extract_prefix(override_sm.to_dict(), path))


def _parse_merge_directive(obj: Any, sm: SourceMap, path: Path) -> MergeDirective | None:
    if not isinstance(obj, dict):
        return None
    if _MERGE_KEY not in obj:
        return None
    mode = obj.get(_MERGE_KEY)
    if mode not in ("replace", "append", "prepend", "keyed"):
        raise ConfigError(
            f"Invalid merge mode {mode!r}",
            config_path=path,
            location=sm.get(path),
            suggestions=["replace", "append", "prepend", "keyed"],
        )
    if _VALUE_KEY not in obj:
        raise ConfigError(
            f"Merge container missing {_VALUE_KEY!r}",
            config_path=path,
            location=sm.get(path),
        )
    key = obj.get("key")
    if mode == "keyed":
        if not isinstance(key, str) or not key:
            raise ConfigError(
                "keyed merge requires a non-empty string 'key' field",
                config_path=path,
                location=sm.get(path),
            )
    else:
        if key is not None:
            raise ConfigError(
                "'key' is only valid for keyed merge",
                config_path=path,
                location=sm.get(path),
            )

    return MergeDirective(mode=mode, key=key, value=obj[_VALUE_KEY])


def _apply_directive(
    base: Any,
    directive: MergeDirective,
    *,
    base_sm: SourceMap,
    override_sm: SourceMap,
    path: Path,
) -> tuple[Any, SourceMap]:
    mode = directive.mode
    override_value = directive.value
    value_path = path + (_VALUE_KEY,)
    override_value_sm = override_sm.submap(value_path).remap_prefix(value_path, path)
    override_effective_sm_dict = override_value_sm.to_dict()
    container_loc = override_sm.get(path)
    if container_loc is not None:
        override_effective_sm_dict[path] = container_loc
    override_effective_sm = SourceMap(override_effective_sm_dict)

    if mode == "replace":
        return override_value, SourceMap(extract_prefix(override_effective_sm.to_dict(), path))

    if mode in ("append", "prepend", "keyed"):
        if base is None:
            base_list: list[Any] = []
        elif isinstance(base, list):
            base_list = base
        else:
            raise ConfigError(
                f"Cannot {mode}-merge into non-list value",
                config_path=path,
                location=override_sm.get(path) or base_sm.get(path),
            )

        if not isinstance(override_value, list):
            raise ConfigError(
                f"{mode} merge requires a list in {_VALUE_KEY!r}",
                config_path=path,
                location=override_sm.get(path),
            )

        if mode == "append":
            return _merge_list_append(
                base_list, override_value, base_sm=base_sm, override_sm=override_effective_sm, path=path
            )
        if mode == "prepend":
            return _merge_list_prepend(
                base_list, override_value, base_sm=base_sm, override_sm=override_effective_sm, path=path
            )
        return _merge_list_keyed(
            base_list,
            override_value,
            key_field=directive.key or "",
            base_sm=base_sm,
            override_sm=override_effective_sm,
            path=path,
        )

    raise ConfigError("Unsupported merge mode", config_path=path, location=override_sm.get(path))  # pragma: no cover


def _merge_dicts(
    base: dict[str, Any],
    override: dict[str, Any],
    *,
    base_sm: SourceMap,
    override_sm: SourceMap,
    path: Path,
) -> tuple[dict[str, Any], SourceMap]:
    out: dict[str, Any] = {}
    out_sm: dict[Path, SourceLocation] = {}

    loc = override_sm.get(path) or base_sm.get(path)
    if loc is not None:
        out_sm[path] = loc

    base_keys = set(base.keys())
    override_keys = set(override.keys())

    for key in base_keys - override_keys:
        out[key] = base[key]
        out_sm.update(extract_prefix(base_sm.to_dict(), path + (key,)))

    for key in override_keys - base_keys:
        out[key] = override[key]
        out_sm.update(extract_prefix(override_sm.to_dict(), path + (key,)))

    for key in base_keys & override_keys:
        merged, merged_sm = merge_values(
            base[key],
            override[key],
            base_sm=base_sm,
            override_sm=override_sm,
            path=path + (key,),
        )
        out[key] = merged
        out_sm.update(merged_sm.to_dict())

    return out, SourceMap(out_sm)


def _shift_list_subtree(sm: SourceMap, *, path: Path, offset: int) -> dict[Path, SourceLocation]:
    out: dict[Path, SourceLocation] = {}
    n = len(path)
    for p, loc in sm.items():
        if p[:n] != path or p == path:
            continue
        if len(p) > n and isinstance(p[n], int):
            out[p[:n] + (p[n] + offset,) + p[n + 1 :]] = loc
        else:
            out[p] = loc
    return out


def _merge_list_append(
    base: list[Any],
    override: list[Any],
    *,
    base_sm: SourceMap,
    override_sm: SourceMap,
    path: Path,
) -> tuple[list[Any], SourceMap]:
    result = list(base) + list(override)
    out_sm = dict(extract_prefix(base_sm.to_dict(), path))
    out_sm.update(_shift_list_subtree(override_sm, path=path, offset=len(base)))

    loc = override_sm.get(path) or base_sm.get(path)
    if loc is not None:
        out_sm[path] = loc
    return result, SourceMap(out_sm)


def _merge_list_prepend(
    base: list[Any],
    override: list[Any],
    *,
    base_sm: SourceMap,
    override_sm: SourceMap,
    path: Path,
) -> tuple[list[Any], SourceMap]:
    result = list(override) + list(base)
    out_sm = dict(extract_prefix(override_sm.to_dict(), path))
    out_sm.update(_shift_list_subtree(base_sm, path=path, offset=len(override)))

    loc = override_sm.get(path) or base_sm.get(path)
    if loc is not None:
        out_sm[path] = loc
    return result, SourceMap(out_sm)


def _merge_list_keyed(
    base: list[Any],
    override: list[Any],
    *,
    key_field: str,
    base_sm: SourceMap,
    override_sm: SourceMap,
    path: Path,
) -> tuple[list[Any], SourceMap]:
    def item_key(item: Any, *, idx: int, origin: str) -> str:
        if not isinstance(item, dict):
            raise ConfigError(
                f"keyed merge requires list items to be mappings (got {type(item).__name__})",
                config_path=path + (idx,),
                location=(override_sm if origin == "override" else base_sm).get(path + (idx,)),
            )
        k = item.get(key_field)
        if not isinstance(k, str) or not k:
            raise ConfigError(
                f"keyed merge items must contain non-empty string key field {key_field!r}",
                config_path=path + (idx, key_field),
                location=(override_sm if origin == "override" else base_sm).get(path + (idx, key_field)),
            )
        return k

    base_index: dict[str, int] = {}
    for i, item in enumerate(base):
        k = item_key(item, idx=i, origin="base")
        if k in base_index:
            raise ConfigError(
                f"Duplicate keyed-merge key {k!r} in base list",
                config_path=path + (i, key_field),
                location=base_sm.get(path + (i, key_field)),
            )
        base_index[k] = i

    override_seen: set[str] = set()
    result = list(base)
    out_sm_map: dict[Path, SourceLocation] = dict(extract_prefix(base_sm.to_dict(), path))
    current_sm = SourceMap(out_sm_map)

    for j, item in enumerate(override):
        k = item_key(item, idx=j, origin="override")
        if k in override_seen:
            raise ConfigError(
                f"Duplicate keyed-merge key {k!r} in override list",
                config_path=path + (j, key_field),
                location=override_sm.get(path + (j, key_field)),
            )
        override_seen.add(k)

        if k in base_index:
            idx = base_index[k]
            base_item = result[idx]
            base_item_sm = current_sm.submap(path + (idx,))
            override_item_sm = override_sm.submap(path + (j,)).remap_prefix(path + (j,), path + (idx,))
            merged_item, merged_item_sm = merge_values(
                base_item,
                item,
                base_sm=base_item_sm,
                override_sm=override_item_sm,
                path=path + (idx,),
            )

            result[idx] = merged_item
            out_sm_map = delete_prefix(out_sm_map, path + (idx,))
            out_sm_map.update(merged_item_sm.to_dict())
            current_sm = SourceMap(out_sm_map)
        else:
            new_idx = len(result)
            result.append(item)
            remapped = override_sm.submap(path + (j,)).remap_prefix(path + (j,), path + (new_idx,))
            out_sm_map.update(remapped.to_dict())
            current_sm = SourceMap(out_sm_map)

    loc = override_sm.get(path) or base_sm.get(path)
    if loc is not None:
        out_sm_map[path] = loc

    return result, SourceMap(out_sm_map)
