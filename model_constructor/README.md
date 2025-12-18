# `model_constructor/`

This directory is the **entire construct-only model builder**. It is designed to be vendored as a submodule inside training repositories and inference repositories without coupling to either.

If you are new, start with:
- `docs/overview.md`
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
  - Enforces minimum supported PyTorch version (see `docs/compatibility.md`).
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

## Using custom blocks from a *parent repository* (recommended for submodules)

If you intend to embed `model_constructor` as a submodule, prefer defining your custom blocks in the parent repo and registering them via YAML `imports`.

Start with:
- `docs/integration.md`
- `model_constructor/blocks/README.md` (Option B)
- `examples/end_to_end.md`

