# `model_constructor.graph`

This folder defines the architecture-agnostic model backend:

- `GraphIR`: a small, explicit representation of a model as **modules + runtime steps**
- `GraphModel`: a generic `torch.nn.Module` that executes `GraphIR`

## GraphIR in one paragraph

A graph model is:
- a set of named **modules** instantiated once and stored in a `ModuleDict`
- a list of **steps** that call either:
  - `module:<name>` (a module in the ModuleDict), or
  - `op:<name>` (a registered runtime callable)
- a set of **outputs** to return

All YAML frontends compile into this representation.

## Runtime references

Graph steps read inputs from a runtime context by using `$name` references:
- `$x` reads the model input named `x`
- `$h1` reads a value produced by a previous step

`$$literal` escapes a literal string starting with `$`.

## DAG-only

`GraphIR` forbids forward references and cycles. Any `$ref` must refer to:
- a graph input, or
- a value produced by a previous step/node.

If you need recurrence, implement it inside a block module and call that module from the graph.

## Common errors

### `ConfigError: Forward reference(s) not allowed`

Graph execution is DAG-only: any `$name` reference must refer to a graph input or a value produced by an earlier step/node.

Fix:
- reorder nodes/steps so producers come before consumers, or
- move recurrence/iteration inside a block module and call that module from the graph

## Outputs

`outputs` can be:
- list of names -> returned as `single` (len 1) or `tuple`
- mapping -> returned as `dict`

Output collisions and missing refs are errors in strict mode.

## Where YAML compiles

Compilation happens in `model_constructor/graph/compiler.py`:
- `model.sequential` -> linear `GraphIR`
- `model.graph` -> `GraphIR` with DAG checks and deterministic execution ordering
