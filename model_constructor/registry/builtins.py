from __future__ import annotations

import operator
from typing import Any

import torch

from .registry import Registry


def register_builtins(registry: Registry) -> None:
    # torch.nn modules (stable keys)
    nn = torch.nn
    for name in [
        "Identity",
        "Linear",
        "LazyLinear",
        "Conv1d",
        "Conv2d",
        "LazyConv2d",
        "BatchNorm1d",
        "BatchNorm2d",
        "LayerNorm",
        "GroupNorm",
        "ReLU",
        "GELU",
        "SiLU",
        "Dropout",
        "Dropout2d",
        "Flatten",
    ]:
        if hasattr(nn, name):
            registry.register_module(f"nn.{name}", getattr(nn, name), signature_policy="best_effort", tags=("torch.nn",))

    # Safe ops (runtime-only by default)
    registry.register_op("add", operator.add, tags=("math",))
    registry.register_op("mul", operator.mul, tags=("math",))
    registry.register_op("getitem", operator.getitem, tags=("util",))
    registry.register_op("cat", torch.cat, tags=("torch",))
    registry.register_op("stack", torch.stack, tags=("torch",))

    def identity(x: Any) -> Any:
        return x

    registry.register_op("identity", identity, tags=("util",))

