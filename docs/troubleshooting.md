# Troubleshooting

## `ModuleNotFoundError: No module named 'model_constructor'`

This repo is not installed as a package by default. Add the repo root to your import path:
- set `PYTHONPATH` to the repo root (or submodule root), or
- insert the path into `sys.path` in the parent repo (see `docs/integration.md`)

## `ConfigError: nodes mapping form requires 'order: [..]'`

When `model.graph.nodes` is a mapping, `order` is required for deterministic execution.

Fix:
- add `order: [name1, name2, ...]`, or
- switch to the canonical list form for `nodes`.

## `ConfigError: Forward reference(s) not allowed`

GraphIR is DAG-only and forbids forward references. Any `$ref` must refer to:
- an input, or
- a value produced by a previous node/step.

Fix:
- reorder nodes/steps, or
- move recurrence/iteration inside a block module.

## `ConfigError: Unknown module type '...'/Unknown op '...'`

You used an unregistered `_type_` (module) or `op:<name>` (op).

Fix:
- check for typos
- ensure your custom block registration is executed (via `imports` or parent-code registration)
- list available built-ins with:
  - `from model_constructor.registry.default_registry import get_default_registry`
  - `get_default_registry().list_modules()` / `.list_ops()`

## `ConfigError: imports are disabled / import ... is not allowed`

`imports` are controlled by settings:
- `settings.allow_imports` (bool)
- `settings.allowed_import_prefixes` (list of prefixes)

If importing a parent repo module, add your prefix, e.g.:
```yaml
settings:
  allowed_import_prefixes: ["model_constructor.", "my_project."]
imports:
  - my_project.model_blocks
```

## `_target_ is disabled by settings.allow_target`

This repo is safe-by-default: `_target_` is off unless explicitly enabled.

Fix:
- prefer `_type_` registry keys, or
- enable `_target_` in `settings` (expert mode) and allow the module prefix.

## PyTorch version error

If you see an error about unsupported torch version, see `docs/compatibility.md` and upgrade PyTorch.

## Running tests

From the repo root:
```bash
python -m pytest -q
```
