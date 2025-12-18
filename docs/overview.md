# Overview

`model_constructor` is a **construct-only** PyTorch model architecture constructor.

It takes a YAML configuration file, resolves composition/interpolation/templates deterministically, compiles the config into a small graph IR, and returns a `torch.nn.Module` that executes that graph.

## What this repo is (and is not)

**Is**
- A YAML-first architecture playground (swap blocks/params in YAML, no training code).
- Architecture/input/output agnostic (models are graphs of modules + ops; your parent repo decides the task).
- Submodule-friendly (intended to be embedded into a training repo or an inference engine without coupling).

**Is not**
- A trainer, dataset loader, optimizer config system, checkpointing framework, metrics runner, or inference server.

## End-to-end data flow

1. `build_model(path_or_dict)` (`model_constructor/api.py:16`)
2. `resolve_config(...)` (`model_constructor/config/resolve.py:1`)
   - YAML parse with a `SourceMap` (best-effort file/line/col tracking)
   - `defaults` includes (file-based composition)
   - `_merge_` list directives
   - `_template_` expansion
   - `${...}` interpolation (including typed `${env:int:...}`)
   - schema v1 checks
3. Compile to `GraphIR` (`model_constructor/graph/compiler.py:14`)
   - Supports `model.sequential` and `model.graph`
   - Enforces DAG-only (no forward references)
4. Instantiate modules from specs (`model_constructor/instantiate/instantiate.py:1`)
   - Module specs use registry keys via `_type_` (safe default)
5. Execute via `GraphModel` (`model_constructor/graph/model.py:1`)

The normative semantic contract is documented in `model_constructor/config/schema_v1.md`.

## Core abstractions

### ResolvedConfig
- A fully-resolved config dict plus settings and a source map.
- Produced by `model_constructor/config/resolve.py`.

### Registry
- A single registry for both **modules** (construct-time) and **ops** (runtime functions).
- Implemented in `model_constructor/registry/registry.py`.
- Default registry is created in `model_constructor/registry/default_registry.py` and includes:
  - common `torch.nn` modules under keys like `nn.Linear`
  - safe ops under keys used as `op:<name>` (e.g. `op:add`)
  - built-in blocks like `conv_bn_act`, `mlp`, `residual_block`

### GraphIR
`GraphIR` is the single backend representation of any model:
- `inputs`: names bound from `forward(*args, **kwargs)`
- `modules`: named module specs (instantiated once and stored in a `ModuleDict`)
- `steps`: runtime calls to `module:<name>` or `op:<name>` with explicit outputs
- `outputs` + `return` policy

Implemented in `model_constructor/graph/ir.py` and compiled by `model_constructor/graph/compiler.py`.

### GraphModel
`GraphModel` is a generic `torch.nn.Module` executor for `GraphIR`:
- binds inputs
- executes steps
- writes named outputs into a context dict
- returns outputs as `single`, `tuple`, or `dict`

Implemented in `model_constructor/graph/model.py`.

## Extensibility model

You can add custom building blocks without changing core logic:

1. **In this repo**: implement a new `torch.nn.Module` under `model_constructor/blocks/` and register it in `model_constructor/blocks/register.py`.
2. **In a parent repo** (recommended when embedding as a submodule):
   - Create a module (e.g. `my_project/model_blocks.py`) that exposes `register(registry)`.
   - Use YAML `imports` + `settings.allowed_import_prefixes` to load it, then reference your blocks via `_type_`.
   - Or register programmatically and pass `registry=` to `build_model`.

Concrete walkthroughs are in:
- `model_constructor/config/authoring_yaml.md`
- `model_constructor/blocks/README.md`
- `docs/integration.md`
