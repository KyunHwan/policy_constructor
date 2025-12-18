# `model_constructor/`

This directory is the **entire construct-only model builder**. It is designed to be vendored as a submodule inside training repositories and inference repositories without coupling to either.

If you are new, start with:
- `../README.md` (overview + integration)
- `model_constructor/config/schema_v1.md` (normative YAML contract)
- `model_constructor/config/authoring_yaml.md` (practical YAML examples)

## Directory map (what each component does)

### Top-level files

- `model_constructor/__init__.py`
  - Re-exports the public API: `build_model`, `compile_ir`, `resolve_config`, and `Registry`.
- `model_constructor/api.py`
  - The main integration surface. Parent repositories should call `build_model(...)` and treat the returned module as a normal `torch.nn.Module`.
  - Internally orchestrates: config resolution → optional imports/registration → IR compilation → module instantiation → `GraphModel`.
- `model_constructor/compat.py`
  - Enforces minimum supported PyTorch version (see `../README.md`).
- `model_constructor/errors.py`
  - Defines `ConfigError` and other error types. Most user-facing failures should be raised as `ConfigError` with a helpful path and (best-effort) source location.

### Subdirectories

- `model_constructor/config/`
  - Deterministic YAML resolution pipeline (composition, merge directives, templates, interpolation) and schema checks.
  - Start here:
    - `model_constructor/config/README.md`
    - `model_constructor/config/schema_v1.md`
    - `model_constructor/config/authoring_yaml.md`
- `model_constructor/registry/`
  - Single registry for:
    - construct-time module factories referenced by YAML `_type_`
    - runtime ops referenced by `call: op:<name>`
  - Start here: `model_constructor/registry/README.md`
- `model_constructor/instantiate/`
  - The single instantiation engine that turns YAML specs into Python objects with strict/best-effort kwarg validation.
  - Start here: `model_constructor/instantiate/README.md`
- `model_constructor/graph/`
  - The architecture-agnostic backend.
  - `GraphIR` is the single internal representation; `GraphModel` is the single runtime executor.
  - Start here: `model_constructor/graph/README.md`
- `model_constructor/blocks/`
  - Built-in reusable building blocks (pure `torch.nn.Module`s), plus the block registration entrypoint.
  - Start here: `model_constructor/blocks/README.md`
- `model_constructor/util/`
  - Small utilities, including the YAML-driven imports plugin mechanism.
  - Start here: `model_constructor/util/README.md`

## End-to-end build pipeline (from YAML to `nn.Module`)

At a high level, `build_model("path/to/model.yaml")` does:

1. Resolve config (`model_constructor/config/resolve.py`)
   - `defaults` includes (file-based composition)
   - list merge directives (`_merge_`)
   - templates (`_template_`)
   - interpolation (`${...}`), including typed env interpolation
   - schema checks (v1)
2. Apply `imports` (optional)
   - imports modules and calls `register(registry)` if present (see `model_constructor/util/imports.py`)
3. Compile to `GraphIR` (`model_constructor/graph/compiler.py`)
   - supports `model.sequential` and `model.graph`
   - enforces DAG-only graphs (no forward references)
4. Instantiate named modules (`model_constructor/instantiate/instantiate.py`)
5. Return a `GraphModel` (`model_constructor/graph/model.py`)

## Complete example: add a custom building block *inside this repo*

This is the most direct workflow for first-time users experimenting in this codebase. It keeps:
- block implementation in Python
- all hyperparameters and model wiring in YAML

### Step 1: create a new block module

Create `model_constructor/blocks/residual_mlp.py`:

```python
import torch

class ResidualMLP(torch.nn.Module):
    def __init__(self, *, width: int, hidden_mult: int = 4, dropout: float = 0.0):
        super().__init__()
        hidden = width * hidden_mult
        self.norm = torch.nn.LayerNorm(width)
        self.fc1 = torch.nn.Linear(width, hidden)
        self.act = torch.nn.GELU()
        self.drop = torch.nn.Dropout(dropout)
        self.fc2 = torch.nn.Linear(hidden, width)

    def forward(self, x):
        y = self.norm(x)
        y = self.fc1(y)
        y = self.act(y)
        y = self.drop(y)
        y = self.fc2(y)
        return x + y
```

Notes:
- This block is input-shape agnostic across leading dims (PyTorch `Linear` applies to the last dim).
- All parameters (`width`, `hidden_mult`, `dropout`) are YAML-modifiable.

### Step 2: register the block

Edit `model_constructor/blocks/register.py` and add:

```python
from .residual_mlp import ResidualMLP
registry.register_module("residual_mlp", ResidualMLP, signature_policy="strict", tags=("blocks", "mlp"))
```

Now YAML can reference `_type_: residual_mlp`.

### Step 3: write a model YAML that uses your block

Create `configs/examples/custom_residual_mlp.yaml`:

```yaml
schema_version: 1

params:
  width: 128
  hidden_mult: 4
  dropout: 0.1

model:
  sequential:
    inputs: [x]
    layers:
      - _type_: residual_mlp
        width: ${params.width}
        hidden_mult: ${params.hidden_mult}
        dropout: ${params.dropout}
      - _type_: residual_mlp
        width: ${params.width}
        hidden_mult: ${params.hidden_mult}
        dropout: ${params.dropout}
```

### Step 4: build and run a quick smoke check

From the repo root:

```python
import torch
from model_constructor import build_model

model = build_model("configs/examples/custom_residual_mlp.yaml")
x = torch.randn(2, 8, 128)   # (batch, tokens, width)
y = model(x)
print(y.shape)
```

## Complete example: custom blocks + `model.graph` (skip connections + weight sharing)

This example shows why `model.graph` exists:
- you can wire **explicit skip connections** (and other DAG topologies)
- you can **reuse the same module instance multiple times** for weight sharing (call the same `module:<name>` more than once)

### Step 1: create two small block modules

Create `model_constructor/blocks/token_mlp.py`:

```python
import torch


class TokenMLP(torch.nn.Module):
    def __init__(self, *, width: int, hidden_mult: int = 4, dropout: float = 0.0):
        super().__init__()
        hidden = width * hidden_mult
        self.norm = torch.nn.LayerNorm(width)
        self.fc1 = torch.nn.Linear(width, hidden)
        self.act = torch.nn.GELU()
        self.drop = torch.nn.Dropout(dropout)
        self.fc2 = torch.nn.Linear(hidden, width)

    def forward(self, x):
        y = self.norm(x)
        y = self.fc1(y)
        y = self.act(y)
        y = self.drop(y)
        y = self.fc2(y)
        return y
```

Create `model_constructor/blocks/gated_add.py`:

```python
import torch


class GatedAdd(torch.nn.Module):
    def __init__(self, *, init: float = 0.0):
        super().__init__()
        self.gate = torch.nn.Parameter(torch.tensor(float(init)))

    def forward(self, x, y):
        return x + self.gate * y
```

Notes:
- `TokenMLP` is an intentionally small, reusable primitive: it transforms `x` but does not bake in residual wiring.
- `GatedAdd` is a tiny “wiring” block: it makes residual connections explicit and learnable (common for stable deep residual stacks).

### Step 2: register the blocks

Edit `model_constructor/blocks/register.py` and add:

```python
from .gated_add import GatedAdd
from .token_mlp import TokenMLP

registry.register_module("token_mlp", TokenMLP, signature_policy="strict", tags=("blocks", "mlp"))
registry.register_module("gated_add", GatedAdd, signature_policy="strict", tags=("blocks", "util"))
```

Now YAML can reference `_type_: token_mlp` and `_type_: gated_add`.

### Step 3: write a graph YAML that uses custom blocks + built-ins

Create `configs/examples/custom_graph_token_mlp.yaml`:

```yaml
schema_version: 1

params:
  width: 128
  hidden_mult: 4
  dropout: 0.1
  num_classes: 10

model:
  graph:
    inputs: [x]
    modules:
      stem: {_type_: nn.LazyLinear, out_features: ${params.width}}
      block_shared: {_type_: token_mlp, width: ${params.width}, hidden_mult: ${params.hidden_mult}, dropout: ${params.dropout}}
      fuse: {_type_: gated_add, init: 0.0}
      head: {_type_: mlp, dims: [${params.width}, ${params.width}, ${params.num_classes}], dropout: ${params.dropout}}
    nodes:
      - {name: h0, call: module:stem, args: [$x]}
      - {name: h1, call: module:block_shared, args: [$h0]}
      - {name: h2, call: module:block_shared, args: [$h1]}
      - {name: h3, call: module:fuse, args: [$h0, $h2]}
      - {name: logits, call: module:head, args: [$h3]}
    outputs: [$logits]
    return: single
```

Key takeaways:
- `block_shared` is called twice, so those two calls **share weights**. If you want independent blocks, define `block1` and `block2` in `modules`.
- This model mixes:
  - your custom blocks (`token_mlp`, `gated_add`)
  - torch modules (`nn.LazyLinear`)
  - built-in repo blocks (`mlp`)

### Step 4: build and run a quick smoke check

From the repo root:

```python
import torch
from model_constructor import build_model

model = build_model("configs/examples/custom_graph_token_mlp.yaml")
x = torch.randn(2, 8, 64)  # (batch, tokens, in_width); LazyLinear infers in_width=64 on first call
y = model(x)
print(y.shape)
```

## Using custom blocks from a *parent repository* (recommended for submodules)

If you intend to embed `model_constructor` as a submodule, prefer defining your custom blocks in the parent repo and registering them via YAML `imports`.

Start with:
- `../README.md` (Integration section)
- `model_constructor/blocks/README.md` (Option B)
- `examples/end_to_end.md`
