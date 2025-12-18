# Schema v1

This document is normative: behavior is defined here and regression-locked by tests.

## Resolution pipeline (deterministic)

1. Parse YAML with source locations
2. Resolve `defaults` includes (ordered, cycle-checked)
3. Merge (dict deep-merge, list replace by default, `_merge_` directives)
4. Expand `_template_` (cycle-checked)
5. Interpolate `${...}` (cycle-checked)
6. Validate schema v1
7. Compile to `GraphIR`
8. Validate `GraphIR` (DAG-only, no forward refs)

## Top-level keys

- `schema_version: 1` (required)
- `settings` (optional)
- `defaults` (optional, consumed by resolver)
- `imports` (optional, list of modules to import; see `settings`)
- `templates` (optional, consumed by resolver)
- `model` (required)

## Merge directives (`_merge_`)

Lists replace by default. To control list merge behavior, wrap the list in a merge container:

```yaml
imports:
  _merge_: append
  _value_:
    - my_project.blocks
```

Supported merge modes:
- `replace`
- `append`
- `prepend`
- `keyed` (requires `key: <field_name>`, enforces uniqueness)

## Templates (`_template_`)

```yaml
templates:
  conv3x3:
    _type_: conv_bn_act
    kernel_size: 3
    padding: 1

model:
  ...
```

Any dict node can reference a template:

```yaml
stem:
  _template_: conv3x3
  in_channels: 3
  out_channels: 64
```

Template values are deep-merged into the node (node wins).

## Interpolation (`${...}`)

### Config path

- `${path.to.value}` references another value in the resolved config.
- If the entire scalar equals the interpolation expression, the referenced value's **type is preserved**.
- If interpolation appears inside a larger string, it is **stringified**.

### Environment

- `${env:VAR}` / `${env:VAR,default}` yields the raw env string (no trimming).
- Typed forms trim whitespace before casting:
  - `${env:int:VAR}` / `${env:int:VAR,42}`
  - `${env:float:VAR}` / `${env:float:VAR,0.1}`
  - `${env:bool:VAR}` / `${env:bool:VAR,false}` (true/false, 1/0, yes/no, on/off)
  - `${env:json:VAR}` parses JSON via `json.loads`

## Graph semantics

- `GraphIR` is **DAG-only**.
- Forward references are forbidden: any `$ref` must refer to a model input or a value produced by a previous node/step.
- Recurrence must live inside a module/block, not in the graph.

See `model_constructor/config/authoring_yaml.md` for practical examples.
