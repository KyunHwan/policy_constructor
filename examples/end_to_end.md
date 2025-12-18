# End-to-End Example (Custom Blocks + Training Repo + Inference Repo)

This walkthrough shows a practical “two parent repos” setup:
- a **training** repository that trains the model
- an **inference** repository that loads the same model architecture

Both repos reuse:
- the same `model_constructor` submodule revision
- the same YAML config(s)
- the same custom block registration module (or equivalent)

See also:
- `docs/integration.md`
- `model_constructor/blocks/README.md`

## 1) Training repository layout

```
train_repo/
  third_party/
    model_constructor/        # this repo as a git submodule
  my_project/
    __init__.py
    model_blocks.py           # your custom building blocks + register()
  configs/
    models/
      my_model.yaml
  train.py
```

### `my_project/model_blocks.py`

```python
import torch

class TokenMixer(torch.nn.Module):
    def __init__(self, *, width: int, dropout: float = 0.0):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(width, width),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

def register(registry):
    registry.register_module("my_project.token_mixer", TokenMixer, signature_policy="strict")
```

### `configs/models/my_model.yaml`

This YAML:
- imports your block registry module
- composes a model with built-ins + your custom blocks
- keeps all hyperparameters YAML-modifiable

```yaml
schema_version: 1
settings:
  allowed_import_prefixes: ["model_constructor.", "my_project."]
imports:
  - my_project.model_blocks

params:
  width: 128
  dropout: 0.1

model:
  graph:
    inputs: [x]
    modules:
      proj: {_type_: nn.Linear, in_features: ${params.width}, out_features: ${params.width}}
      mix: {_type_: my_project.token_mixer, width: ${params.width}, dropout: ${params.dropout}}
    nodes:
      h1: {call: module:proj, args: [$x]}
      h2: {call: module:mix, args: [$h1]}
    order: [h1, h2]
    outputs: [$h2]
    return: single
```

### `train.py` (skeleton)

```python
import sys
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent / "third_party" / "model_constructor"))
from model_constructor import build_model

def main():
    model = build_model("configs/models/my_model.yaml")
    # training repo owns all training logic:
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    ...

if __name__ == "__main__":
    main()
```

## 2) Inference repository layout

```
infer_repo/
  third_party/
    model_constructor/        # same submodule revision as train_repo
  my_project/
    __init__.py
    model_blocks.py           # same registry keys as train_repo
  configs/
    models/
      my_model.yaml           # identical YAML (copied or shared)
  infer.py
```

### `infer.py` (skeleton)

```python
import sys
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent / "third_party" / "model_constructor"))
from model_constructor import build_model

def main():
    model = build_model("configs/models/my_model.yaml")
    model.eval()
    with torch.no_grad():
        x = torch.randn(2, 128)
        y = model(x)
        print(y.shape)

if __name__ == "__main__":
    main()
```

## 3) How to keep training and inference consistent

To guarantee the same architecture:
- pin the `model_constructor` submodule commit in both repos
- keep your registry keys stable (e.g. `my_project.token_mixer`)
- keep the YAML config identical (or versioned)

If you change a block implementation but keep the same key, both repos must update together.
