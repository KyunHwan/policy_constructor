from __future__ import annotations

import inspect
from difflib import get_close_matches
from typing import Any, Callable

from ..errors import ConfigError
from ..registry.registry import SignaturePolicy


def validate_kwargs(
    target: Callable[..., Any],
    *,
    kwargs: dict[str, Any],
    policy: SignaturePolicy,
    config_path: tuple[Any, ...],
) -> None:
    if policy == "runtime_only":
        return

    try:
        sig = inspect.signature(target)
    except Exception as exc:
        if policy == "strict":
            raise ConfigError(f"Unable to introspect signature: {exc}", config_path=config_path) from exc
        return

    params = sig.parameters
    accepts_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
    if accepts_var_kw:
        return

    allowed = {
        name
        for name, p in params.items()
        if p.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    }
    unknown = set(kwargs) - allowed
    if unknown:
        unk = sorted(unknown)
        suggestions = get_close_matches(unk[0], allowed, n=5) if len(unk) == 1 else []
        raise ConfigError(f"Unknown kwargs: {unk}", config_path=config_path, suggestions=suggestions)

