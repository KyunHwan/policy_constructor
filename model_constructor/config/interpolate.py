from __future__ import annotations

import json
import os
import re
from copy import deepcopy
from typing import Any

from ..errors import ConfigError, SourceLocation
from .source_map import Path, SourceMap


_INTERP_RE = re.compile(r"\$\{([^{}]+)\}")


def interpolate_config(config: dict[str, Any], *, source_map: SourceMap) -> tuple[dict[str, Any], SourceMap]:
    interpolator = _Interpolator(config, source_map=source_map)
    value, sm = interpolator.resolve_path(())
    if not isinstance(value, dict):  # pragma: no cover
        raise ConfigError("interpolation produced non-mapping root", config_path=())
    return value, sm


class _Interpolator:
    def __init__(self, config: Any, *, source_map: SourceMap) -> None:
        self._config = config
        self._source_map = source_map
        self._cache: dict[Path, tuple[Any, SourceMap]] = {}
        self._resolving: list[Path] = []

    def resolve_path(self, path: Path) -> tuple[Any, SourceMap]:
        if path in self._cache:
            return self._cache[path]
        if path in self._resolving:
            cycle = " -> ".join(self._format_path(p) for p in self._resolving + [path])
            raise ConfigError(
                f"Interpolation cycle detected: {cycle}",
                config_path=path,
                location=self._source_map.get(path),
            )

        raw = self._get_raw(path)
        self._resolving.append(path)
        try:
            resolved_value, resolved_sm = self._resolve_value(raw, path)
        finally:
            self._resolving.pop()

        self._cache[path] = (resolved_value, resolved_sm)
        return resolved_value, resolved_sm

    def _resolve_value(self, value: Any, path: Path) -> tuple[Any, SourceMap]:
        loc = self._source_map.get(path)

        if isinstance(value, dict):
            out: dict[str, Any] = {}
            out_sm: dict[Path, SourceLocation] = {}
            if loc is not None:
                out_sm[path] = loc
            for k, v in value.items():
                child, child_sm = self.resolve_path(path + (k,))
                out[k] = child
                out_sm.update(child_sm.to_dict())
            return out, SourceMap(out_sm)

        if isinstance(value, list):
            out_list: list[Any] = []
            out_sm: dict[Path, SourceLocation] = {}
            if loc is not None:
                out_sm[path] = loc
            for i, _ in enumerate(value):
                child, child_sm = self.resolve_path(path + (i,))
                out_list.append(child)
                out_sm.update(child_sm.to_dict())
            return out_list, SourceMap(out_sm)

        if not isinstance(value, str):
            return value, SourceMap({path: loc} if loc is not None else {})

        if "${" not in value:
            return value, SourceMap({path: loc} if loc is not None else {})

        matches = list(_INTERP_RE.finditer(value))
        if not matches:
            return value, SourceMap({path: loc} if loc is not None else {})

        if len(matches) == 1 and matches[0].span() == (0, len(value)):
            expr = matches[0].group(1).strip()
            resolved, resolved_sm = self._eval_expr(expr, path)

            if isinstance(resolved, (dict, list)):
                # Preserve the interpolation site's location at `path`, but remap descendants for better errors.
                out_sm: dict[Path, SourceLocation] = {}
                if loc is not None:
                    out_sm[path] = loc
                for p, ploc in resolved_sm.items():
                    if p == ():  # skip referenced root marker
                        continue
                    out_sm[path + p] = ploc
                return deepcopy(resolved), SourceMap(out_sm)

            return resolved, SourceMap({path: loc} if loc is not None else {})

        rendered = value
        for m in reversed(matches):
            expr = m.group(1).strip()
            resolved, _ = self._eval_expr(expr, path)
            rendered = rendered[: m.start()] + _stringify(resolved) + rendered[m.end() :]
        return rendered, SourceMap({path: loc} if loc is not None else {})

    def _eval_expr(self, expr: str, current_path: Path) -> tuple[Any, SourceMap]:
        if expr.startswith("env:"):
            value = self._eval_env(expr, current_path)
            loc = self._source_map.get(current_path)
            if isinstance(value, (dict, list)):
                return value, _fill_sourcemap(value, (), loc)
            return value, SourceMap({(): loc} if loc is not None else {})

        ref_path = _parse_config_path(expr, current_path=current_path, source_map=self._source_map)
        resolved_value, resolved_sm = self.resolve_path(ref_path)
        return resolved_value, resolved_sm.remap_prefix(ref_path, ())

    def _eval_env(self, expr: str, current_path: Path) -> Any:
        # env:VAR[,default]
        # env:<type>:VAR[,default] where type in {int,float,bool,json}
        rest = expr[len("env:") :]
        cast: str | None = None

        for prefix in ("int:", "float:", "bool:", "json:"):
            if rest.startswith(prefix):
                cast = prefix[:-1]
                rest = rest[len(prefix) :]
                break

        if not rest:
            raise ConfigError("Invalid env interpolation", config_path=current_path, location=self._source_map.get(current_path))

        var, default = _split_var_default(rest)

        if cast is None:
            if var in os.environ:
                return os.environ[var]
            if default is not None:
                return default
            raise ConfigError(
                f"Missing env var {var!r}",
                config_path=current_path,
                location=self._source_map.get(current_path),
            )

        raw = os.environ.get(var, default)
        if raw is None:
            raise ConfigError(
                f"Missing env var {var!r}",
                config_path=current_path,
                location=self._source_map.get(current_path),
            )

        raw_stripped = raw.strip()
        if raw_stripped == "":
            raise ConfigError(
                f"Env var {var!r} is empty/whitespace",
                config_path=current_path,
                location=self._source_map.get(current_path),
            )

        try:
            if cast == "int":
                return _parse_int(raw_stripped)
            if cast == "float":
                return float(raw_stripped)
            if cast == "bool":
                return _parse_bool(raw_stripped)
            if cast == "json":
                return json.loads(raw_stripped)
        except Exception as exc:
            raise ConfigError(
                f"Failed to cast env var {var!r} as {cast}: {exc}",
                config_path=current_path,
                location=self._source_map.get(current_path),
            ) from exc

        raise ConfigError("Invalid env cast", config_path=current_path, location=self._source_map.get(current_path))  # pragma: no cover

    def _get_raw(self, path: Path) -> Any:
        cur = self._config
        for seg in path:
            if isinstance(seg, int):
                if not isinstance(cur, list) or seg < 0 or seg >= len(cur):
                    raise ConfigError(
                        f"Invalid config path (list index): {self._format_path(path)}",
                        config_path=path,
                        location=self._source_map.get(path),
                    )
                cur = cur[seg]
            else:
                if not isinstance(cur, dict) or seg not in cur:
                    raise ConfigError(
                        f"Invalid config path (missing key): {self._format_path(path)}",
                        config_path=path,
                        location=self._source_map.get(path),
                    )
                cur = cur[seg]
        return cur

    @staticmethod
    def _format_path(path: Path) -> str:
        out = ""
        for seg in path:
            if isinstance(seg, int):
                out += f"[{seg}]"
            else:
                if out:
                    out += "."
                out += str(seg)
        return out or "<root>"


def _split_var_default(rest: str) -> tuple[str, str | None]:
    if "," in rest:
        var, default = rest.split(",", 1)
        return var, default
    return rest, None


def _parse_int(s: str) -> int:
    s2 = s.strip()
    lower = s2.lower()
    if lower.startswith(("0x", "0o", "0b")):
        return int(s2, 0)
    return int(s2, 10)


def _parse_bool(s: str) -> bool:
    v = s.strip().lower()
    if v in {"true", "1", "yes", "on"}:
        return True
    if v in {"false", "0", "no", "off"}:
        return False
    raise ValueError(f"Invalid bool value: {s!r}")


def _parse_config_path(expr: str, *, current_path: Path, source_map: SourceMap) -> Path:
    parts = expr.split(".")
    if any(p == "" for p in parts):
        raise ConfigError(
            f"Invalid config path expression {expr!r}",
            config_path=current_path,
            location=source_map.get(current_path),
        )

    out: list[Any] = []
    for part in parts:
        while True:
            if "[" not in part:
                out.append(part)
                break

            base, rest = part.split("[", 1)
            if base:
                out.append(base)
            else:
                raise ConfigError(
                    f"Invalid config path expression {expr!r}",
                    config_path=current_path,
                    location=source_map.get(current_path),
                )

            if "]" not in rest:
                raise ConfigError(
                    f"Invalid config path expression {expr!r}",
                    config_path=current_path,
                    location=source_map.get(current_path),
                )
            idx_str, remainder = rest.split("]", 1)
            if not idx_str.isdigit():
                raise ConfigError(
                    f"Invalid list index in config path expression {expr!r}",
                    config_path=current_path,
                    location=source_map.get(current_path),
                )
            out.append(int(idx_str))
            part = remainder
            if part == "":
                break
            if not part.startswith("["):
                raise ConfigError(
                    f"Invalid config path expression {expr!r}",
                    config_path=current_path,
                    location=source_map.get(current_path),
                )

    return tuple(out)


def _stringify(value: Any) -> str:
    if isinstance(value, (dict, list)):
        return json.dumps(value)
    return str(value)


def _fill_sourcemap(value: Any, path: Path, loc: SourceLocation | None) -> SourceMap:
    out: dict[Path, SourceLocation] = {}

    def walk(v: Any, p: Path) -> None:
        if loc is not None:
            out[p] = loc
        if isinstance(v, dict):
            for k, child in v.items():
                walk(child, p + (k,))
        elif isinstance(v, list):
            for i, child in enumerate(v):
                walk(child, p + (i,))

    walk(value, path)
    return SourceMap(out)
