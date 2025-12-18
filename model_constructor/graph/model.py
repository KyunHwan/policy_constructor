from __future__ import annotations

from dataclasses import asdict
from typing import Any, Callable

import torch

from ..errors import GraphExecutionError
from .ir import Ref, ReturnPolicy, StepIR, StepOut


class GraphModel(torch.nn.Module):
    def __init__(
        self,
        *,
        inputs: list[str],
        modules: dict[str, torch.nn.Module],
        steps: list[StepIR],
        outputs: list[str] | dict[str, str],
        return_policy: ReturnPolicy,
        ops: dict[str, Callable[..., Any]] | None = None,
    ) -> None:
        super().__init__()
        self.inputs = list(inputs)
        self.graph_modules = torch.nn.ModuleDict(modules)
        self._steps = list(steps)
        self._outputs = outputs
        self._return_policy = return_policy
        self._ops = dict(ops or {})

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        ctx: dict[str, Any] = {}
        ctx.update(self._bind_inputs(args, kwargs))

        for i, step in enumerate(self._steps):
            try:
                resolved_args = [_resolve_runtime_value(v, ctx) for v in step.args]
                resolved_kwargs = {k: _resolve_runtime_value(v, ctx) for k, v in step.kwargs.items()}
                result = self._execute_step(step, resolved_args, resolved_kwargs)
                _write_outputs(ctx, step.out, result)
            except Exception as exc:
                raise GraphExecutionError(f"step[{i}] failed ({step.call_kind}:{step.call_name}): {exc}") from exc

        return self._pack_outputs(ctx)

    def _execute_step(self, step: StepIR, args: list[Any], kwargs: dict[str, Any]) -> Any:
        if step.call_kind == "module":
            mod = self.graph_modules[step.call_name]
            return mod(*args, **kwargs)
        fn = self._ops.get(step.call_name)
        if fn is None:
            raise GraphExecutionError(f"Unknown op {step.call_name!r}")
        return fn(*args, **kwargs)

    def _bind_inputs(self, args: tuple[Any, ...], kwargs: dict[str, Any]) -> dict[str, Any]:
        bound: dict[str, Any] = {}
        arg_i = 0
        used_kwargs: set[str] = set()

        for name in self.inputs:
            if name in kwargs:
                bound[name] = kwargs[name]
                used_kwargs.add(name)
            else:
                if arg_i >= len(args):
                    raise GraphExecutionError(f"Missing required input {name!r}")
                bound[name] = args[arg_i]
                arg_i += 1

        extra_args = len(args) - arg_i
        if extra_args:
            raise GraphExecutionError(f"Too many positional args: expected {len(self.inputs)}, got {len(args)}")

        extra_kwargs = set(kwargs.keys()) - used_kwargs
        if extra_kwargs:
            raise GraphExecutionError(f"Unexpected kwargs: {sorted(extra_kwargs)}")

        return bound

    def _pack_outputs(self, ctx: dict[str, Any]) -> Any:
        if self._return_policy == "dict":
            assert isinstance(self._outputs, dict)
            return {k: ctx[v] for k, v in self._outputs.items()}

        assert isinstance(self._outputs, list)
        if self._return_policy == "single":
            return ctx[self._outputs[0]]
        return tuple(ctx[n] for n in self._outputs)


def _resolve_runtime_value(v: Any, ctx: dict[str, Any]) -> Any:
    if isinstance(v, Ref):
        if v.name not in ctx:
            raise GraphExecutionError(f"Missing runtime value {v.name!r}")
        return ctx[v.name]
    if isinstance(v, str) and v.startswith("$") and not v.startswith("${"):
        name = v[1:]
        if name not in ctx:
            raise GraphExecutionError(f"Missing runtime value {name!r}")
        return ctx[name]
    if isinstance(v, dict):
        return {k: _resolve_runtime_value(val, ctx) for k, val in v.items()}
    if isinstance(v, list):
        return [_resolve_runtime_value(val, ctx) for val in v]
    return v


def _write_outputs(ctx: dict[str, Any], out: StepOut, result: Any) -> None:
    if isinstance(out, str):
        _assign(ctx, out, result)
        return
    if isinstance(out, list):
        if not isinstance(result, (tuple, list)):
            raise GraphExecutionError("Expected tuple/list result for list out mapping")
        if len(result) != len(out):
            raise GraphExecutionError(f"Output length mismatch: expected {len(out)}, got {len(result)}")
        for name, val in zip(out, result, strict=False):
            _assign(ctx, name, val)
        return
    if not isinstance(result, dict):
        raise GraphExecutionError("Expected dict result for dict out mapping")
    for returned_key, ctx_name in out.items():
        if returned_key not in result:
            raise GraphExecutionError(f"Missing dict key {returned_key!r} in op/module result")
        _assign(ctx, ctx_name, result[returned_key])


def _assign(ctx: dict[str, Any], name: str, value: Any) -> None:
    if name in ctx:
        raise GraphExecutionError(f"Context name collision: {name!r}")
    ctx[name] = value

