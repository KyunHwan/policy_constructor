# `model_constructor.util`

This folder contains small utilities used by the core pipeline.

## `imports` plugin mechanism

`model_constructor/util/imports.py` implements a simple plugin hook:

- YAML can contain `imports: ["some.module", ...]`
- each imported module may define `register(registry)`
- if present, `register(registry)` is called with the active registry

This enables parent repositories to add custom blocks in a YAML-first workflow, without modifying `model_constructor`.

Safety controls are enforced via settings:
- `settings.allow_imports` (default true)
- `settings.allowed_import_prefixes` (default `["model_constructor."]`)

See `../README.md` (Integration section) and `model_constructor/blocks/README.md` for examples.

## Common errors

### `ConfigError: imports are disabled`

`imports` was turned off via `settings.allow_imports: false`.

Fix:
- remove `imports` if you don’t need it, or
- enable it explicitly:

```yaml
settings:
  allow_imports: true
```

### `ConfigError: import ... is not allowed`

The import prefix guard rejected the module you tried to load.

Fix: allow your parent repo’s package prefix:

```yaml
settings:
  allowed_import_prefixes: ["model_constructor.", "my_project."]
imports:
  - my_project.model_blocks
```
