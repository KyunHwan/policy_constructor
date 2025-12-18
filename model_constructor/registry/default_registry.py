from __future__ import annotations

from .builtins import register_builtins
from .registry import Registry

_DEFAULT: Registry | None = None


def get_default_registry() -> Registry:
    global _DEFAULT
    if _DEFAULT is None:
        reg = Registry()
        register_builtins(reg)

        # Built-in blocks (registered explicitly into this registry)
        from ..blocks.register import register_blocks

        register_blocks(reg)
        _DEFAULT = reg
    return _DEFAULT

