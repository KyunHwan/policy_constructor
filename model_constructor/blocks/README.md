# Custom Building Blocks

This repo is designed so that:
- **YAML controls composition + hyperparameters**
- **Python defines reusable primitives (blocks)**

You can add blocks either by modifying this repository (fork/submodule changes) or by extending it from a parent repository (recommended).

## Built-in blocks in this repository

These blocks are registered into the default registry (see `model_constructor/blocks/register.py`).

### `conv_bn_act`

2D convolution followed by normalization and activation.

Example:
```yaml
_type_: conv_bn_act
in_channels: null        # null -> LazyConv2d
out_channels: 32
kernel_size: 3
padding: 1
act: {_type_: nn.ReLU}   # override activation (optional)
```

### `mlp`

Feedforward MLP specified by a `dims` list.

Example:
```yaml
_type_: mlp
dims: [null, 256, 10]    # null -> LazyLinear for the first layer
dropout: 0.1
```

### `residual_block`

A simple 2-layer residual block with an optional projection skip if `stride != 1` or channels change.

Example:
```yaml
_type_: residual_block
in_channels: 64
out_channels: 128
stride: 2
```

## Option A: Add blocks inside this repo

1. Create a new module under `model_constructor/blocks/`, e.g. `model_constructor/blocks/my_block.py`:

```python
import torch

class MyBlock(torch.nn.Module):
    def __init__(self, *, width: int, dropout: float = 0.0):
        super().__init__()
        self.fc = torch.nn.Linear(width, width)
        self.drop = torch.nn.Dropout(dropout)

    def forward(self, x):
        return self.drop(self.fc(x))
```

2. Register it in `model_constructor/blocks/register.py`:

```python
from .my_block import MyBlock
registry.register_module("my_block", MyBlock, signature_policy="strict")
```

3. Use it from YAML:

```yaml
schema_version: 1
model:
  sequential:
    layers:
      - _type_: my_block
        width: 128
        dropout: 0.1
```

## Option B (recommended): Add blocks from a parent repository

This lets a training repo and an inference repo share the same `model_constructor` submodule without forking it.

### B1) Define a registration module in the parent repo

Create `my_project/model_blocks.py`:

```python
import torch

class MyBlock(torch.nn.Module):
    def __init__(self, *, width: int):
        super().__init__()
        self.fc = torch.nn.Linear(width, width)

    def forward(self, x):
        return self.fc(x)

def register(registry):
    registry.register_module("my_project.my_block", MyBlock, signature_policy="strict", tags=("my_project",))
```

Key point: `model_constructor` will import this module and call `register(registry)` if it exists.

### B2) Allow importing the parent module (settings)

By default, `imports` are restricted to `model_constructor.*`. In your model YAML (stored in the parent repo), allow your prefix:

```yaml
schema_version: 1
settings:
  allowed_import_prefixes: ["model_constructor.", "my_project."]
imports:
  - my_project.model_blocks
model:
  sequential:
    layers:
      - _type_: my_project.my_block
        width: 128
```

This keeps model definitions YAML-first while still allowing custom primitives.

### B3) Use the same YAML in training and inference repos

As long as both repos:
- include the same `my_project.model_blocks` module (or equivalent)
- share the same `model_constructor` submodule revision

then the model built from YAML will be identical.

## Parameterization and “architecture playground” workflows

Common patterns:
- Put reusable structures in `defaults` base configs.
- Override only the small “params” section in experiment configs.
- Use templates for repeated block patterns.
- Use `${env:int:...}` to control sizes from environment variables when launching runs.

For YAML features (defaults/templates/merge/interpolation), see `model_constructor/config/authoring_yaml.md` and `model_constructor/config/schema_v1.md`.
