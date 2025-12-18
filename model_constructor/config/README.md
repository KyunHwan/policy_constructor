# `model_constructor.config`

This folder contains the **deterministic YAML resolution pipeline**. All other parts of the system consume the *resolved* config and do not re-implement parsing/merging/interpolation.

## Key entrypoints

- `model_constructor/config/resolve.py`: top-level resolver
- `model_constructor/config/schema_v1.md`: normative schema/semantics contract
- `model_constructor/config/authoring_yaml.md`: practical YAML authoring guide

## Resolver stages (high level)

`resolve_config(path)` performs:

1. **YAML parse** + build a `SourceMap` (best-effort file/line/col) (`model_constructor/config/yaml_loader.py`)
2. **`defaults` include resolution** (file-based composition; cycle-checked) (`model_constructor/config/resolve.py`)
3. **Merge** with explicit list directives via `_merge_` (`model_constructor/config/merge.py`)
4. **Templates** via `_template_` (`model_constructor/config/templates.py`)
5. **Interpolation** `${...}` with cycle detection (`model_constructor/config/interpolate.py`)
6. **Schema checks** (`model_constructor/config/schema.py`)

The result is a `ResolvedConfig` which then compiles to `GraphIR` in `model_constructor/graph/`.

## Merge directives (`_merge_`)

Lists **replace by default**. To control list merges, use a merge container:

```yaml
imports:
  _merge_: append
  _value_:
    - my_project.model_blocks
```

Supported modes: `replace`, `append`, `prepend`, `keyed`.

## Templates (`_template_`)

Templates live under `templates:` and are expanded before interpolation. Templates are deep-merged into the node (node wins).

## Interpolation

- `${path.to.value}`: references values in the merged config
- `${env:VAR}`: raw string (no trimming)
- typed env: `${env:int:VAR}`, `${env:bool:VAR}`, `${env:json:VAR}` (whitespace-trimmed before casting)

Full-scalar interpolation preserves type; embedded interpolation stringifies.

## Settings

Settings are parsed by `model_constructor/config/settings.py`:

- `strict` (default true): stricter schema/spec enforcement
- `allow_imports` (default true): enables top-level `imports`
- `allowed_import_prefixes` (default `["model_constructor."]`): guardrail for `imports` (and `_target_` if enabled)
- `allow_target` (default false): enables `_target_` import-by-string (expert mode)

