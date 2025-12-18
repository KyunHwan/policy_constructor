# `model_constructor.registry`

The registry is the single source of truth for:
- **modules**: construct-time callables that create `torch.nn.Module` instances (used by YAML `_type_`)
- **ops**: runtime callables used in graphs (used by `call: op:<name>`)

## Default registry

`model_constructor/registry/default_registry.py` builds the default registry by:

1. registering built-in `torch.nn` modules and safe ops (`model_constructor/registry/builtins.py`)
2. registering built-in blocks (`model_constructor/blocks/register.py`)

## Registry keys

### Modules (`_type_`)

Examples:
- `nn.Linear`
- `nn.Conv2d`
- `conv_bn_act` (built-in block)
- `my_project.token_mixer` (custom block from a parent repo)

### Ops (`call: op:<name>`)

Examples:
- `op:add`
- `op:cat`

Ops are functions (or callables) executed during `GraphModel.forward`.

## Extending from a parent repository

Preferred pattern:

1. In the parent repo, create a module that exports `register(registry)` and registers your blocks.
2. In YAML, use:
   - `settings.allowed_import_prefixes` to allow your package prefix
   - `imports` to import your module (which triggers `register(registry)`)
   - `_type_` keys that refer to your registry entries

See `model_constructor/blocks/README.md` for a full example.

## Listing available entries

In a Python REPL:

```python
from model_constructor.registry.default_registry import get_default_registry

reg = get_default_registry()
print(reg.list_modules())
print(reg.list_ops())
```

## Common errors

### `ConfigError: Unknown module type '...'` / `ConfigError: Unknown op '...'`

You used an unregistered `_type_` (module) or `op:<name>` (op).

Fix:
- check for typos
- ensure your custom block registration is executed (via YAML `imports` or parent-code registration)
- list available built-ins with the snippet above
