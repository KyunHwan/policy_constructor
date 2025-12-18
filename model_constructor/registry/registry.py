from __future__ import annotations

from dataclasses import dataclass
from difflib import get_close_matches
from typing import Any, Callable, Literal

from ..errors import ConfigError

SignaturePolicy = Literal["strict", "best_effort", "runtime_only"]
EntryKind = Literal["module", "op"]


@dataclass(frozen=True)
class RegistryEntry:
    name: str
    kind: EntryKind
    target: Callable[..., Any]
    signature_policy: SignaturePolicy = "best_effort"
    tags: tuple[str, ...] = ()
    doc: str | None = None


class Registry:
    def __init__(self) -> None:
        self._modules: dict[str, RegistryEntry] = {}
        self._ops: dict[str, RegistryEntry] = {}

    def register_module(
        self,
        name: str,
        factory: Callable[..., Any],
        *,
        signature_policy: SignaturePolicy = "best_effort",
        tags: tuple[str, ...] = (),
        doc: str | None = None,
    ) -> None:
        self._register(name, kind="module", target=factory, signature_policy=signature_policy, tags=tags, doc=doc)

    def register_op(
        self,
        name: str,
        fn: Callable[..., Any],
        *,
        signature_policy: SignaturePolicy = "runtime_only",
        tags: tuple[str, ...] = (),
        doc: str | None = None,
    ) -> None:
        self._register(name, kind="op", target=fn, signature_policy=signature_policy, tags=tags, doc=doc)

    def _register(
        self,
        name: str,
        *,
        kind: EntryKind,
        target: Callable[..., Any],
        signature_policy: SignaturePolicy,
        tags: tuple[str, ...],
        doc: str | None,
    ) -> None:
        if not isinstance(name, str) or not name:
            raise ValueError("Registry key must be a non-empty string")

        entry = RegistryEntry(
            name=name,
            kind=kind,
            target=target,
            signature_policy=signature_policy,
            tags=tags,
            doc=doc,
        )
        store = self._modules if kind == "module" else self._ops
        if name in store:
            raise ValueError(f"Duplicate registry key {name!r} for kind {kind}")
        store[name] = entry

    def get_module(self, name: str, *, config_path: tuple[Any, ...] | None = None) -> RegistryEntry:
        if name in self._modules:
            return self._modules[name]
        suggestions = get_close_matches(name, self._modules.keys(), n=5)
        raise ConfigError(f"Unknown module type {name!r}", config_path=config_path, suggestions=suggestions)

    def get_op(self, name: str, *, config_path: tuple[Any, ...] | None = None) -> RegistryEntry:
        if name in self._ops:
            return self._ops[name]
        suggestions = get_close_matches(name, self._ops.keys(), n=5)
        raise ConfigError(f"Unknown op {name!r}", config_path=config_path, suggestions=suggestions)

    def list_modules(self) -> list[str]:
        return sorted(self._modules.keys())

    def list_ops(self) -> list[str]:
        return sorted(self._ops.keys())

