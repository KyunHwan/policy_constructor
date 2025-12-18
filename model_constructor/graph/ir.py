from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Literal


@dataclass(frozen=True)
class Ref:
    name: str


StepOut = str | list[str] | dict[str, str]
ReturnPolicy = Literal["single", "tuple", "dict"]


@dataclass(frozen=True)
class ModuleIR:
    spec: dict[str, Any]
    origin_path: tuple[Any, ...]


@dataclass(frozen=True)
class StepIR:
    call_kind: Literal["module", "op"]
    call_name: str
    args: list[Any]
    kwargs: dict[str, Any]
    out: StepOut
    origin_path: tuple[Any, ...]


@dataclass(frozen=True)
class GraphIR:
    inputs: list[str]
    modules: dict[str, ModuleIR]
    steps: list[StepIR]
    outputs: list[str] | dict[str, str]
    return_policy: ReturnPolicy
    ops: dict[str, Callable[..., Any]]
