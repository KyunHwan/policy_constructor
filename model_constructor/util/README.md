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

See `docs/integration.md` and `model_constructor/blocks/README.md` for examples.

