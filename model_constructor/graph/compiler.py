from __future__ import annotations

import re
from typing import Any, Callable, Iterable

from ..config.source_map import SourceMap
from ..errors import ConfigError
from ..registry.registry import Registry
from .ir import GraphIR, ModuleIR, Ref, ReturnPolicy, StepIR, StepOut

_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def compile_ir(config: dict[str, Any], *, source_map: SourceMap, registry: Registry) -> GraphIR:
    model = config.get("model")
    if not isinstance(model, dict):
        raise ConfigError("'model' must be a mapping", config_path=("model",), location=source_map.get(("model",)))

    if "sequential" in model:
        if "graph" in model:
            raise ConfigError("model may not contain both 'sequential' and 'graph'", config_path=("model",))
        return _compile_sequential(model["sequential"], source_map=source_map, registry=registry)

    if "graph" in model:
        return _compile_graph(model["graph"], source_map=source_map, registry=registry)

    raise ConfigError("model must contain either 'sequential' or 'graph'", config_path=("model",))


def _compile_sequential(seq: Any, *, source_map: SourceMap, registry: Registry) -> GraphIR:
    if not isinstance(seq, dict):
        raise ConfigError("'model.sequential' must be a mapping", config_path=("model", "sequential"))

    inputs = seq.get("inputs", ["x"])
    if not isinstance(inputs, list) or not all(isinstance(x, str) for x in inputs):
        raise ConfigError("'model.sequential.inputs' must be a list of strings", config_path=("model", "sequential", "inputs"))
    if len(inputs) != 1:
        raise ConfigError("sequential frontend supports exactly one input", config_path=("model", "sequential", "inputs"))
    for name in inputs:
        _validate_name(name, ("model", "sequential", "inputs"))

    layers = seq.get("layers")
    if not isinstance(layers, list) or not layers:
        raise ConfigError("'model.sequential.layers' must be a non-empty list", config_path=("model", "sequential", "layers"))

    modules: dict[str, ModuleIR] = {}
    steps: list[StepIR] = []
    prev = inputs[0]

    for i, layer_spec in enumerate(layers):
        mod_name = f"layer{i}"
        modules[mod_name] = ModuleIR(spec=layer_spec, origin_path=("model", "sequential", "layers", i))

        out_name = f"h{i}"
        steps.append(
            StepIR(
                call_kind="module",
                call_name=mod_name,
                args=[Ref(prev)],
                kwargs={},
                out=out_name,
                origin_path=("model", "sequential", "layers", i),
            )
        )
        prev = out_name

    outputs_spec = seq.get("outputs")
    return_policy = seq.get("return")
    outputs, policy = _parse_outputs(outputs_spec, return_policy, default_outputs=[prev], config_path=("model", "sequential"))

    ops: dict[str, Callable[..., Any]] = {}
    return GraphIR(inputs=inputs, modules=modules, steps=steps, outputs=outputs, return_policy=policy, ops=ops)


def _compile_graph(graph: Any, *, source_map: SourceMap, registry: Registry) -> GraphIR:
    if not isinstance(graph, dict):
        raise ConfigError("'model.graph' must be a mapping", config_path=("model", "graph"))

    inputs = graph.get("inputs")
    if not isinstance(inputs, list) or not inputs or not all(isinstance(x, str) for x in inputs):
        raise ConfigError("'model.graph.inputs' must be a non-empty list of strings", config_path=("model", "graph", "inputs"))
    for name in inputs:
        _validate_name(name, ("model", "graph", "inputs"))

    modules_raw = graph.get("modules", {})
    if modules_raw is None:
        modules_raw = {}
    if not isinstance(modules_raw, dict) or not all(isinstance(k, str) for k in modules_raw.keys()):
        raise ConfigError("'model.graph.modules' must be a mapping", config_path=("model", "graph", "modules"))
    for name in modules_raw.keys():
        _validate_name(name, ("model", "graph", "modules", name))

    modules: dict[str, ModuleIR] = {
        name: ModuleIR(spec=spec, origin_path=("model", "graph", "modules", name)) for name, spec in modules_raw.items()
    }

    if "nodes" in graph and "steps" in graph:
        raise ConfigError("model.graph may not contain both 'nodes' and 'steps'", config_path=("model", "graph"))

    if "nodes" in graph:
        steps = _compile_nodes_to_steps(
            graph,
            config_path=("model", "graph"),
        )
    elif "steps" in graph:
        steps_raw = graph.get("steps")
        steps = _compile_steps_list(steps_raw, config_path=("model", "graph", "steps"))
    else:
        raise ConfigError("model.graph requires either 'nodes' or 'steps'", config_path=("model", "graph"))

    outputs_spec = graph.get("outputs")
    if outputs_spec is None:
        raise ConfigError("model.graph.outputs is required", config_path=("model", "graph", "outputs"))

    return_policy = graph.get("return")
    outputs, policy = _parse_outputs(outputs_spec, return_policy, default_outputs=None, config_path=("model", "graph"))

    ops = _collect_ops(steps, registry=registry)

    ir = GraphIR(inputs=inputs, modules=modules, steps=steps, outputs=outputs, return_policy=policy, ops=ops)
    _validate_ir(ir, registry=registry)
    return ir


def _compile_nodes_to_steps(graph: dict[str, Any], *, config_path: tuple[Any, ...]) -> list[StepIR]:
    nodes = graph.get("nodes")
    if isinstance(nodes, list):
        return [_compile_node(n, config_path=config_path + ("nodes", i)) for i, n in enumerate(nodes)]

    if isinstance(nodes, dict):
        order = graph.get("order")
        if not isinstance(order, list) or not all(isinstance(x, str) for x in order):
            raise ConfigError(
                "nodes mapping form requires 'order: [..]'",
                config_path=config_path + ("order",),
            )
        out: list[StepIR] = []
        for name in order:
            if name not in nodes:
                raise ConfigError(
                    f"order references unknown node {name!r}",
                    config_path=config_path + ("order",),
                )
            body = nodes[name]
            if not isinstance(body, dict):
                raise ConfigError(
                    "node body must be a mapping",
                    config_path=config_path + ("nodes", name),
                )
            node = dict(body)
            node["name"] = name
            out.append(_compile_node(node, config_path=config_path + ("nodes", name)))
        return out

    raise ConfigError("'nodes' must be a list or a mapping", config_path=config_path + ("nodes",))


def _compile_node(node: Any, *, config_path: tuple[Any, ...]) -> StepIR:
    if not isinstance(node, dict):
        raise ConfigError("node must be a mapping", config_path=config_path)
    name = node.get("name")
    if not isinstance(name, str) or not name:
        raise ConfigError("node.name must be a non-empty string", config_path=config_path + ("name",))
    _validate_name(name, config_path + ("name",))

    call = node.get("call")
    if not isinstance(call, str) or not call:
        raise ConfigError("node.call must be a non-empty string", config_path=config_path + ("call",))

    args = node.get("args", [])
    kwargs = node.get("kwargs", {})
    if args is None:
        args = []
    if kwargs is None:
        kwargs = {}
    if not isinstance(args, list):
        raise ConfigError("node.args must be a list", config_path=config_path + ("args",))
    if not isinstance(kwargs, dict):
        raise ConfigError("node.kwargs must be a mapping", config_path=config_path + ("kwargs",))

    out_spec = node.get("out", name)
    out = _parse_step_out(out_spec, config_path=config_path + ("out",))

    call_kind, call_name = _parse_call(call, config_path=config_path + ("call",))

    return StepIR(
        call_kind=call_kind,
        call_name=call_name,
        args=[_parse_runtime_value(v, config_path=config_path + ("args", i)) for i, v in enumerate(args)],
        kwargs={k: _parse_runtime_value(v, config_path=config_path + ("kwargs", k)) for k, v in kwargs.items()},
        out=out,
        origin_path=config_path,
    )


def _compile_steps_list(steps_raw: Any, *, config_path: tuple[Any, ...]) -> list[StepIR]:
    if not isinstance(steps_raw, list) or not steps_raw:
        raise ConfigError("steps must be a non-empty list", config_path=config_path)
    out: list[StepIR] = []
    for i, step in enumerate(steps_raw):
        if not isinstance(step, dict):
            raise ConfigError("step must be a mapping", config_path=config_path + (i,))
        call = step.get("call")
        if not isinstance(call, str) or not call:
            raise ConfigError("step.call must be a non-empty string", config_path=config_path + (i, "call"))
        args = step.get("args", [])
        kwargs = step.get("kwargs", {})
        if args is None:
            args = []
        if kwargs is None:
            kwargs = {}
        if not isinstance(args, list):
            raise ConfigError("step.args must be a list", config_path=config_path + (i, "args"))
        if not isinstance(kwargs, dict):
            raise ConfigError("step.kwargs must be a mapping", config_path=config_path + (i, "kwargs"))
        out_spec = step.get("out")
        if out_spec is None:
            raise ConfigError("step.out is required", config_path=config_path + (i, "out"))
        out_parsed = _parse_step_out(out_spec, config_path=config_path + (i, "out"))
        call_kind, call_name = _parse_call(call, config_path=config_path + (i, "call"))
        out.append(
            StepIR(
                call_kind=call_kind,
                call_name=call_name,
                args=[_parse_runtime_value(v, config_path=config_path + (i, "args", j)) for j, v in enumerate(args)],
                kwargs={k: _parse_runtime_value(v, config_path=config_path + (i, "kwargs", k)) for k, v in kwargs.items()},
                out=out_parsed,
                origin_path=config_path + (i,),
            )
        )
    return out


def _parse_call(call: str, *, config_path: tuple[Any, ...]) -> tuple[str, str]:
    if call.startswith("module:"):
        name = call[len("module:") :]
        if not name:
            raise ConfigError("module call missing name", config_path=config_path)
        _validate_name(name, config_path)
        return "module", name
    if call.startswith("op:"):
        name = call[len("op:") :]
        if not name:
            raise ConfigError("op call missing name", config_path=config_path)
        return "op", name
    raise ConfigError("call must start with 'module:' or 'op:'", config_path=config_path)


def _parse_step_out(out: Any, *, config_path: tuple[Any, ...]) -> StepOut:
    if isinstance(out, str):
        _validate_name(out, config_path)
        return out
    if isinstance(out, list) and out and all(isinstance(x, str) for x in out):
        for x in out:
            _validate_name(x, config_path)
        return out
    if isinstance(out, dict) and out and all(isinstance(k, str) for k in out.keys()) and all(isinstance(v, str) for v in out.values()):
        for v in out.values():
            _validate_name(v, config_path)
        return out  # type: ignore[return-value]
    raise ConfigError("Invalid out spec (str | list[str] | dict[str,str])", config_path=config_path)


def _parse_runtime_value(v: Any, *, config_path: tuple[Any, ...]) -> Any:
    if isinstance(v, str) and v.startswith("$$"):
        return v[1:]
    if isinstance(v, str) and v.startswith("$") and not v.startswith("${"):
        name = v[1:]
        if not name:
            raise ConfigError("Invalid runtime ref '$' (empty name)", config_path=config_path)
        _validate_name(name, config_path)
        return Ref(name)
    if isinstance(v, dict):
        return {k: _parse_runtime_value(val, config_path=config_path + (k,)) for k, val in v.items()}
    if isinstance(v, list):
        return [_parse_runtime_value(val, config_path=config_path + (i,)) for i, val in enumerate(v)]
    return v


def _collect_ops(steps: list[StepIR], *, registry: Registry) -> dict[str, Callable[..., Any]]:
    ops: dict[str, Callable[..., Any]] = {}
    for i, step in enumerate(steps):
        if step.call_kind != "op":
            continue
        name = step.call_name
        if name not in ops:
            entry = registry.get_op(name, config_path=step.origin_path + ("call",))
            ops[name] = entry.target
    return ops


def _validate_ir(ir: GraphIR, *, registry: Registry) -> None:
    available = set(ir.inputs)

    for i, step in enumerate(ir.steps):
        step_path = step.origin_path or ("model", "graph", "steps", i)
        if step.call_kind == "module":
            if step.call_name not in ir.modules:
                raise ConfigError(
                    f"Step references unknown module {step.call_name!r}",
                    config_path=step_path + ("call",),
                )
        else:
            registry.get_op(step.call_name, config_path=step_path + ("call",))

        refs = _collect_refs(step.args) | _collect_refs(step.kwargs)
        missing = sorted(r.name for r in refs if r.name not in available)
        if missing:
            raise ConfigError(
                f"Forward reference(s) not allowed: {missing}",
                config_path=step_path,
            )

        produced = _out_names(step.out)
        collisions = sorted(n for n in produced if n in available)
        if collisions:
            raise ConfigError(
                f"Output name collision(s): {collisions}",
                config_path=step_path + ("out",),
            )
        available.update(produced)

    if ir.return_policy == "dict":
        if not isinstance(ir.outputs, dict):
            raise ConfigError("return_policy=dict requires outputs mapping", config_path=("model", "graph", "outputs"))
        missing_out = sorted(v for v in ir.outputs.values() if v not in available)
        if missing_out:
            raise ConfigError(f"outputs reference missing keys: {missing_out}", config_path=("model", "graph", "outputs"))
    else:
        if not isinstance(ir.outputs, list) or not ir.outputs:
            raise ConfigError("outputs must be a non-empty list", config_path=("model", "graph", "outputs"))
        missing_out = sorted(v for v in ir.outputs if v not in available)
        if missing_out:
            raise ConfigError(f"outputs reference missing keys: {missing_out}", config_path=("model", "graph", "outputs"))
        if ir.return_policy == "single" and len(ir.outputs) != 1:
            raise ConfigError("return_policy=single requires exactly one output", config_path=("model", "graph", "return"))


def _collect_refs(obj: Any) -> set[Ref]:
    out: set[Ref] = set()

    def walk(v: Any) -> None:
        if isinstance(v, Ref):
            out.add(v)
            return
        if isinstance(v, dict):
            for vv in v.values():
                walk(vv)
        elif isinstance(v, list):
            for vv in v:
                walk(vv)

    walk(obj)
    return out


def _out_names(out: StepOut) -> list[str]:
    if isinstance(out, str):
        return [out]
    if isinstance(out, list):
        return list(out)
    return list(out.values())


def _parse_outputs(
    outputs_spec: Any,
    return_policy_raw: Any,
    *,
    default_outputs: list[str] | None,
    config_path: tuple[Any, ...],
) -> tuple[list[str] | dict[str, str], ReturnPolicy]:
    if outputs_spec is None:
        if default_outputs is None:
            raise ConfigError("outputs is required", config_path=config_path + ("outputs",))
        outputs = default_outputs
    else:
        outputs = _coerce_outputs(outputs_spec, config_path=config_path + ("outputs",))

    policy = _coerce_return_policy(return_policy_raw, outputs, config_path=config_path + ("return",))
    return outputs, policy


def _coerce_outputs(outputs_spec: Any, *, config_path: tuple[Any, ...]) -> list[str] | dict[str, str]:
    if isinstance(outputs_spec, list):
        if not outputs_spec or not all(isinstance(x, str) for x in outputs_spec):
            raise ConfigError("outputs must be a non-empty list of strings", config_path=config_path)
        out_names: list[str] = []
        for x in outputs_spec:
            if x.startswith("$") and not x.startswith("${"):
                x = x[1:]
            _validate_name(x, config_path)
            out_names.append(x)
        return out_names

    if isinstance(outputs_spec, dict):
        if not outputs_spec:
            raise ConfigError("outputs mapping must be non-empty", config_path=config_path)
        out_map: dict[str, str] = {}
        for k, v in outputs_spec.items():
            if not isinstance(k, str) or not k:
                raise ConfigError("outputs mapping keys must be non-empty strings", config_path=config_path)
            if not isinstance(v, str) or not v:
                raise ConfigError("outputs mapping values must be non-empty strings", config_path=config_path + (k,))
            if v.startswith("$") and not v.startswith("${"):
                v = v[1:]
            _validate_name(v, config_path + (k,))
            out_map[k] = v
        return out_map

    raise ConfigError("outputs must be a list or a mapping", config_path=config_path)


def _coerce_return_policy(policy_raw: Any, outputs: list[str] | dict[str, str], *, config_path: tuple[Any, ...]) -> ReturnPolicy:
    if policy_raw is None:
        if isinstance(outputs, dict):
            return "dict"
        if isinstance(outputs, list) and len(outputs) == 1:
            return "single"
        return "tuple"

    if policy_raw not in ("single", "tuple", "dict"):
        raise ConfigError("return must be one of: single | tuple | dict", config_path=config_path)

    if policy_raw == "dict" and not isinstance(outputs, dict):
        raise ConfigError("return=dict requires outputs mapping", config_path=config_path)
    if policy_raw != "dict" and not isinstance(outputs, list):
        raise ConfigError("return=single/tuple requires outputs list", config_path=config_path)
    if policy_raw == "single" and isinstance(outputs, list) and len(outputs) != 1:
        raise ConfigError("return=single requires exactly one output", config_path=config_path)

    return policy_raw


def _validate_name(name: str, config_path: tuple[Any, ...]) -> None:
    if not _NAME_RE.match(name):
        raise ConfigError(f"Invalid name {name!r} (must match {_NAME_RE.pattern})", config_path=config_path)
