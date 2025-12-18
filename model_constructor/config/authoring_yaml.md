# Authoring YAML Configs

This repo is designed so that **all model architecture experiments are expressed in YAML**, while Python code is only needed to define *new primitives* (building blocks).

The normative contract is `model_constructor/config/schema_v1.md`; this file is a practical guide with examples.

## Minimal config

```yaml
schema_version: 1
model:
  sequential:
    layers:
      - {_type_: nn.Identity}
```

## Object specs (`_type_`, `_args_`, `_kwargs_`)

Any dict containing `_type_` is treated as a construct-time spec and will be instantiated recursively.

### `_type_` (recommended)
Use a registry key:

```yaml
_type_: conv_bn_act
in_channels: null
out_channels: 32
kernel_size: 3
padding: 1
```

### Nested specs
You can override nested modules with specs:

```yaml
_type_: conv_bn_act
in_channels: 16
out_channels: 16
act: {_type_: nn.Identity}
```

### `_args_` and `_kwargs_`
Most specs should use keyword args directly, but `_args_` and `_kwargs_` exist for edge cases:

```yaml
_type_: nn.Linear
_args_: [128, 256]
_kwargs_: {bias: false}
```

### `_target_` (expert mode)
Import-by-string is disabled by default. If you enable it:

```yaml
settings:
  allow_target: true
  allowed_import_prefixes: ["model_constructor.", "torch."]
```

Prefer `_type_` whenever possible for reproducibility and safety.

## Frontend 1: sequential

Sequential is optimized for quick stacks and intentionally minimal:

```yaml
schema_version: 1
model:
  sequential:
    inputs: [x]      # exactly one input in v1
    layers:
      - {_type_: nn.Flatten}
      - _type_: mlp
        dims: [null, 256, 10]
    return: single   # inferred if omitted
```

Notes:
- In v1, `model.sequential.inputs` must be a list with exactly one name.
- Outputs default to the last layer output.

## Frontend 2: graph

Graph is the general architecture form (DAG-only).

### Modules + nodes + order (mapping form)
If `nodes` is a mapping, `order` is required (do not rely on YAML key order):

```yaml
schema_version: 1
model:
  graph:
    inputs: [x]
    modules:
      stem: {_type_: conv_bn_act, in_channels: null, out_channels: 16, kernel_size: 3, padding: 1}
      main: {_type_: conv_bn_act, in_channels: 16, out_channels: 16, kernel_size: 3, padding: 1, act: {_type_: nn.Identity}}
    nodes:
      h1: {call: module:stem, args: [$x]}
      h2: {call: module:main, args: [$h1]}
      h3: {call: op:add, args: [$h1, $h2]}
    order: [h1, h2, h3]
    outputs: [$h3]
    return: single
```

### Nodes as an ordered list (canonical form)
```yaml
nodes:
  - {name: h1, call: module:stem, args: [$x]}
  - {name: h2, call: module:main, args: [$h1]}
  - {name: h3, call: op:add, args: [$h1, $h2]}
```

### Explicit steps (power users)
The `steps` format is already canonical; each step must include `out`:

```yaml
steps:
  - {call: module:stem, args: [$x], out: h1}
  - {call: op:add, args: [$h1, $h1], out: h2}
```

## Runtime references (`$x`) and escaping (`$$`)

- `$name` means: look up a runtime value named `name` in the graph context.
- `$$literal` is an escape for a literal string starting with `$` (becomes `$literal`).

Examples:
```yaml
args: [$x]
kwargs: {dim: 1}
```

## Outputs and return policy

`outputs` can be:
- a list of names (often written as `[$h3]`)
- a mapping for dict returns (e.g. `{logits: $h3, features: $h2}`)

`return` can be:
- `single` (requires exactly one output)
- `tuple`
- `dict` (requires outputs mapping)

If `return` is omitted, it is inferred:
- outputs mapping -> `dict`
- outputs list of length 1 -> `single`
- otherwise -> `tuple`

## Composition with `defaults`

`defaults` are only supported when loading from a file path (not when passing an in-memory dict).

`configs/base/backbone.yaml`
```yaml
schema_version: 1
params: {width: 32}
model:
  graph:
    inputs: [x]
    modules:
      stem: {_type_: conv_bn_act, in_channels: null, out_channels: ${params.width}, kernel_size: 3, padding: 1}
    nodes:
      h1: {call: module:stem, args: [$x]}
    order: [h1]
    outputs: [$h1]
```

`configs/experiments/w64.yaml`
```yaml
defaults: [../base/backbone.yaml]
schema_version: 1
params: {width: 64}
```

## List merge directives (`_merge_`)

Lists replace by default. To append/prepend/keyed-merge, wrap the list in a merge container:

```yaml
imports:
  _merge_: append
  _value_:
    - my_project.model_blocks
```

## Templates (`templates` + `_template_`)

```yaml
schema_version: 1
templates:
  conv3x3:
    _type_: conv_bn_act
    kernel_size: 3
    padding: 1
model:
  graph:
    inputs: [x]
    modules:
      stem:
        _template_: conv3x3
        in_channels: null
        out_channels: 16
    nodes: [{name: h1, call: module:stem, args: [$x]}]
    outputs: [$h1]
```

Templates are expanded before interpolation.

## Interpolation (`${...}`)

### Config references
- Full scalar `${params.width}` preserves type (e.g. int).
- Embedded `"w_${params.width}"` is stringified.

### Env vars
- `${env:VAR}` returns the raw string with no trimming.
- Typed forms trim whitespace then cast:
  - `${env:int:VAR}`, `${env:float:VAR}`, `${env:bool:VAR}`, `${env:json:VAR}`

Example:
```yaml
params:
  width: ${env:int:WIDTH,32}
  tag: "run_${env:RUN_ID,local}"
```

## Settings

Supported settings (see `model_constructor/config/settings.py`):
- `strict` (default true): errors on unknown reserved keys in specs and various schema violations
- `allow_imports` (default true): enables `imports`
- `allowed_import_prefixes` (default `["model_constructor."]`): allowed module prefixes for `imports` (and `_target_` if enabled)
- `allow_target` (default false): allows `_target_` import-by-string

## Common errors

### `ConfigError: nodes mapping form requires 'order: [..]'`

When `model.graph.nodes` is written as a mapping, `order` is required for deterministic execution (do not rely on YAML key order).

Fix:
- add `order: [name1, name2, ...]`, or
- switch to the canonical list form for `nodes`.

### `_target_ is disabled by settings.allow_target`

This repo is safe-by-default: `_target_` is off unless explicitly enabled.

Fix:
- prefer `_type_` registry keys (recommended), or
- enable `_target_` in settings (expert mode) and allow the module prefix.
