# `model_constructor.instantiate`

This folder contains the single instantiation engine that turns YAML specs into Python objects.

## Specs

Any mapping containing `_type_` (preferred) or `_target_` (expert mode) is treated as a construct-time spec.

Common keys:
- `_type_`: registry key to a module factory (recommended)
- `_args_`: positional args list (rare; prefer keyword args)
- `_kwargs_`: extra keyword args mapping (rare; prefer inline kwargs)

The engine recursively instantiates nested specs inside kwargs.

## Safety defaults

- `_type_` resolves through the registry (deterministic + discoverable).
- `_target_` import-by-string is disabled unless `settings.allow_target: true`.
- `settings.allowed_import_prefixes` gates both `imports` and `_target_` module import prefixes.

## Kwarg validation

Each registry entry has a `signature_policy`:
- `strict`: reject unknown kwargs when signature is introspectable
- `best_effort`: validate when possible; otherwise wrap runtime errors with config context
- `runtime_only`: do not pre-validate; always wrap runtime errors

See `model_constructor/instantiate/signature.py` for the kwarg checking rules.

