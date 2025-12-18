# Integration in Parent Repositories (Training / Inference)

This repo is intentionally “construct-only”. A parent repo owns:
- data loading / preprocessing
- training loops
- checkpoint loading/saving
- inference serving / batching / device placement policies

`model_constructor` only builds a `torch.nn.Module` from YAML.

## Adding as a git submodule

Example:
```bash
git submodule add <this-repo-url> third_party/model_constructor
```

Your parent repo should pin the submodule commit for reproducibility.

## Importing without installing a package

Because there is no pip packaging requirement, you typically add the submodule root to Python’s import path.

### Option 1: `PYTHONPATH`
```bash
export PYTHONPATH="$PWD/third_party/model_constructor:$PYTHONPATH"
```

### Option 2: `sys.path` in the parent repo
```python
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "third_party" / "model_constructor"))

from model_constructor import build_model
```

## Training repo usage (example)

```python
import torch
from model_constructor import build_model

model = build_model("configs/models/my_model.yaml")
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

# your training loop here...
```

## Inference repo usage (example)

```python
import torch
from model_constructor import build_model

model = build_model("configs/models/my_model.yaml")
model.eval()

with torch.no_grad():
    y = model(x)  # your inference code decides how to produce x
```

## Registering custom blocks from the parent repo (YAML-first)

If you want *only YAML* to control which blocks exist, use:
- `imports` to import a parent module
- `register(registry)` inside that module to add blocks into the registry

See `model_constructor/blocks/README.md` for a complete example.

## Using `defaults` includes in a parent repo

`defaults` resolution is file-path based and uses paths relative to the YAML file, so parent repos typically keep a config tree like:

```
configs/
  base/
  experiments/
  models/
```

Then build via:
```python
model = build_model("configs/experiments/exp_001.yaml")
```

Note: `defaults` are not supported when passing an in-memory dict to `build_model`; use file paths when you want composition.

## Reproducibility recommendations

- Prefer `_type_` registry keys over `_target_` imports.
- Keep all registry registrations deterministic (no global side effects).
- Pin:
  - the `model_constructor` submodule commit
  - your parent repo’s registration module code
  - the YAML config used to build the model
