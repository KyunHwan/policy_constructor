# model_constructor

YAML-first, construct-only PyTorch model architecture constructor.

This repository builds `torch.nn.Module` architectures from `.yaml` files and intentionally **does not** include any training or inference engine logic. It is designed to be embedded as a submodule inside other codebases.

## Quick start

```python
from model_constructor import build_model

model = build_model("configs/examples/sequential_mlp.yaml")
```

## What this repo is (and isn't)

**Is**
- A YAML-first architecture playground (swap blocks/params in YAML, no training code).
- Architecture/input/output agnostic (models are graphs of modules + ops; your parent repo decides the task).
- Submodule-friendly (embed it into a training repo or an inference engine without coupling).

**Is not**
- A trainer, dataset loader, optimizer config system, checkpointing framework, metrics runner, or inference server.

## Architecture: `YAML -> ResolvedConfig -> GraphIR -> nn.Module`

The canonical build pipeline is:

1. `build_model(path_or_dict)` (`model_constructor/api.py`)
2. `resolve_config(...)` (`model_constructor/config/resolve.py`)
   - `defaults` includes (file-based composition)
   - `_merge_` list directives
   - `_template_` expansion
   - `${...}` interpolation (including typed `${env:int:...}`)
   - schema v1 checks
3. Compile to `GraphIR` (`model_constructor/graph/compiler.py`)
   - supports `model.sequential` and `model.graph`
   - enforces DAG-only (no forward references)
4. Instantiate modules from specs (`model_constructor/instantiate/instantiate.py`)
   - module specs use registry keys via `_type_` (safe default)
   - runtime ops are referenced as `op:<name>`
5. Execute via `GraphModel` (`model_constructor/graph/model.py`)

The normative semantic contract is documented in `model_constructor/config/schema_v1.md`.

## Integration (training / inference repositories)

This repo is intentionally “construct-only”. A parent repo owns:
- data loading / preprocessing
- training loops
- checkpoint loading/saving
- inference serving / batching / device placement policies

`model_constructor` only builds a `torch.nn.Module` from YAML.

### Add as a git submodule

```bash
git submodule add <this-repo-url> third_party/model_constructor
```

Pin the submodule commit in your parent repo for reproducibility.

### Importing without installing a package

Because there is no pip packaging requirement, parent repos typically add the submodule root to Python’s import path.

**Option 1: `PYTHONPATH`**
```bash
export PYTHONPATH="$PWD/third_party/model_constructor:$PYTHONPATH"
```

**Option 2: `sys.path` in the parent repo**
```python
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "third_party" / "model_constructor"))

from model_constructor import build_model
```

### Training repo usage (example)

```python
import torch
from model_constructor import build_model

model = build_model("configs/models/my_model.yaml")
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

# your training loop here...
```

### Inference repo usage (example)

```python
import torch
from model_constructor import build_model

model = build_model("configs/models/my_model.yaml")
model.eval()

with torch.no_grad():
    y = model(x)  # your inference code decides how to produce x
```

### Registering custom blocks from the parent repo (YAML-first)

If you want *only YAML* to control which blocks exist, use:
- `imports` to import a parent module
- `register(registry)` inside that module to add blocks into the registry

Start with:
- `model_constructor/blocks/README.md`
- `model_constructor/util/README.md`
- `examples/end_to_end.md`

### Using `defaults` includes in a parent repo

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

### Reproducibility recommendations

- Prefer `_type_` registry keys over `_target_` imports.
- Keep all registry registrations deterministic (no global side effects).
- Pin:
  - the `model_constructor` submodule commit
  - your parent repo’s registration module code
  - the YAML config used to build the model

## Compatibility

- Python: 3.10+
- PyTorch: >= 2.2 (enforced by a small runtime check)

## Troubleshooting

- `ModuleNotFoundError: No module named 'model_constructor'`: ensure the submodule root is on `PYTHONPATH` or `sys.path` (see Integration above).
- `ConfigError: Forward reference(s) not allowed`: see `model_constructor/graph/README.md`.
- `ConfigError: Unknown module type '...'/Unknown op '...'`: see `model_constructor/registry/README.md`.
- `ConfigError: imports are disabled / import ... is not allowed`: see `model_constructor/util/README.md`.
- `_target_ is disabled by settings.allow_target`: see `model_constructor/config/authoring_yaml.md`.
- PyTorch version error: upgrade torch (see Compatibility above).

## Documentation map

**Start here**
- `model_constructor/README.md` (directory map + internal architecture)
- `model_constructor/config/schema_v1.md` (normative contract)

**Guides**
- `model_constructor/config/authoring_yaml.md` (how to write YAML)
- `examples/end_to_end.md` (two-parent-repo submodule workflow)

**Component docs (near the code)**
- `model_constructor/blocks/README.md`
- `model_constructor/config/README.md`
- `model_constructor/graph/README.md`
- `model_constructor/instantiate/README.md`
- `model_constructor/registry/README.md`
- `model_constructor/util/README.md`
