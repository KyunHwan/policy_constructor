# Quickstart — Your First Hour

A strictly linear, copy-paste path from a fresh `git clone` to a working model + forward pass + one custom block. No detours. Two variants:

- **Variant 1**: build a model from a Python dict, then from a YAML file. Read first.
- **Variant 2**: define a tiny custom block, register it via parent-repo `imports:`, and use it from YAML.

If anything fails along the way, jump to [TROUBLESHOOTING.md](TROUBLESHOOTING.md) and come back.

A few terms you'll see below — full definitions in [GLOSSARY.md](GLOSSARY.md), but for now:

- **registry** — a lookup table that maps a string key like `"nn.Linear"` to a Python callable that builds a `torch.nn.Module`.
- **spec** — any YAML dict containing `_type_`. The constructor reads the spec and calls the registry to build that module.
- **`GraphModel`** — the `torch.nn.Module` subclass produced by `build_model()`. You call `model(x)` on it like any other PyTorch model.

---

## Prerequisites

You need:

- Python 3.10 or newer
- PyTorch 2.2 or newer
- Git

A working `pytest` install is helpful but not required for the Quickstart itself.

---

## Variant 1 — Build and run an existing model

### Step 1. Clone and verify Python/torch versions

```bash
git clone <this-repo-url> policy_constructor
cd policy_constructor
python --version          # expect: Python 3.10+
python -c "import torch; print(torch.__version__)"   # expect: 2.2.0 or newer
```

If `torch` is not installed:

```bash
pip install torch
```

### Step 2. Put the repo on Python's import path

`policy_constructor` is not pip-installable — it's an import-path-only package. Set `PYTHONPATH` so `import model_constructor` resolves:

```bash
export PYTHONPATH="$PWD:$PYTHONPATH"
```

You'll need to re-export this in any new terminal. If you forget, you'll see `ModuleNotFoundError: No module named 'model_constructor'` — see [TROUBLESHOOTING.md](TROUBLESHOOTING.md).

### Step 3. Run the test suite

```bash
pytest -q
```

You should see all 4+ tests pass. If they don't, stop here and figure out why before moving on — the rest of the Quickstart assumes the suite is green.

### Step 4. Build a tiny model from a Python dict (no YAML yet)

Start a Python REPL (`python`) in the repo root and paste:

```python
import torch
from model_constructor import build_model

# Smallest possible model: identity layer
cfg = {
    "schema_version": 1,
    "model": {
        "sequential": {
            "layers": [{"_type_": "nn.Identity"}],
        },
    },
}
model = build_model(cfg)

x = torch.randn(2, 3)        # (batch=2, features=3)
y = model(x)
print(type(model).__name__)  # GraphModel
print(y.shape)               # torch.Size([2, 3])
print(torch.allclose(y, x))  # True
```

What just happened:

1. `build_model()` resolved the dict, compiled it to a `GraphIR`, instantiated one `nn.Identity` module, and returned a `GraphModel` (a `torch.nn.Module` subclass).
2. `nn.Identity` is one of several built-in `torch.nn` types pre-registered by [`model_constructor/registry/builtins.py`](../model_constructor/registry/builtins.py) (~line 14). You can also use `nn.Linear`, `nn.Conv2d`, `nn.LayerNorm`, `nn.ReLU`, `nn.Flatten`, etc.

### Step 5. Build a slightly bigger model with a skip connection

This uses the `graph` frontend (a DAG) instead of `sequential` (a stack):

```python
import torch
from model_constructor import build_model

cfg = {
    "schema_version": 1,
    "model": {
        "graph": {
            "inputs": ["x"],
            "modules": {"m": {"_type_": "nn.Identity"}},
            "nodes": [
                {"name": "h1", "call": "module:m", "args": ["$x"]},
                {"name": "h2", "call": "op:add", "args": ["$h1", "$h1"]},
            ],
            "outputs": ["$h2"],
            "return": "single",
        },
    },
}
model = build_model(cfg)

x = torch.randn(2, 3)
y = model(x)
print(torch.allclose(y, x + x))   # True — y == 2*x
```

What just happened:

1. `inputs: ["x"]` named the model's single input.
2. `modules: {m: ...}` instantiated one `nn.Identity` and stored it under the name `m`.
3. The two `nodes` each produce a named runtime value (`h1`, `h2`). The string `"$x"` is a **runtime reference** to the input named `x`; `"$h1"` references the output of the previous step.
4. `op:add` is a built-in op (Python's `operator.add`) registered by [`model_constructor/registry/builtins.py`](../model_constructor/registry/builtins.py) (~line 36).
5. `outputs: ["$h2"]` + `return: single` makes `forward()` return the single tensor `h2`.

This matches `test_build_model_graph_skip_add_runs` in [`tests/test_build_and_run.py`](../tests/test_build_and_run.py) — you can verify by reading that test.

### Step 6. List every available block type

Want to know what other `_type_` keys you can write? Ask the registry:

```python
from model_constructor.registry.default_registry import get_default_registry
reg = get_default_registry()
print("Modules:")
for name in reg.list_modules():
    print(" ", name)
print("Ops:")
for name in reg.list_ops():
    print(" ", name)
```

You'll see entries from three groups:

- `nn.*` — built-in `torch.nn` modules (from [`builtins.py`](../model_constructor/registry/builtins.py))
- experimental research blocks like `cfg_vqvae_action_decoder`, `vfp_moe`, `dsrl_q_function`, `resfit_residual_actor` — see [`model_constructor/blocks/experiments/README.md`](../model_constructor/blocks/experiments/README.md)
- ops: `add`, `mul`, `cat`, `stack`, `getitem`, `identity`

### Step 7. Build from a YAML file

Create `quickstart.yaml` in the repo root:

```yaml
schema_version: 1
model:
  graph:
    inputs: [x]
    modules:
      proj:    {_type_: nn.Linear, in_features: 64, out_features: 32}
      act:     {_type_: nn.ReLU}
    nodes:
      - {name: h1, call: module:proj, args: [$x]}
      - {name: h2, call: module:act,  args: [$h1]}
    outputs: [$h2]
    return: single
```

Run it:

```python
import torch
from model_constructor import build_model
model = build_model("quickstart.yaml")
x = torch.randn(4, 64)        # (batch=4, in_features=64)
y = model(x)
print(y.shape)                # torch.Size([4, 32])
```

### Step 8. Modify the YAML, see the output change

Edit `quickstart.yaml`: change `out_features: 32` to `out_features: 16`. Re-run the same Python snippet. The output shape should now be `torch.Size([4, 16])`. Nothing else needs to change.

This is the whole point of the repo: change architecture by editing YAML, not by editing Python.

### Step 9. (Optional) Read the existing example YAMLs

The repo ships two example YAMLs under [`configs/examples/`](../configs/examples/):

- [`sequential_mlp.yaml`](../configs/examples/sequential_mlp.yaml) — uses `_type_: mlp`
- [`graph_skip_add.yaml`](../configs/examples/graph_skip_add.yaml) — uses `_type_: conv_bn_act`

> **Note:** As of this writing, the `mlp`, `conv_bn_act`, and `residual_block` types referenced in those YAMLs are defined in [`model_constructor/blocks/basic_blocks/`](../model_constructor/blocks/basic_blocks/) **but are not registered** in [`model_constructor/blocks/register.py`](../model_constructor/blocks/register.py). Building those two example YAMLs without first registering the basic blocks will raise `ConfigError: Unknown module type 'mlp'` (or `'conv_bn_act'`). See [TROUBLESHOOTING.md](TROUBLESHOOTING.md#unknown-module-type-mlp--conv_bn_act--residual_block) for two ways to make them work, and see [Open questions](../docs/MENTAL_MODEL.md#open-questions-for-the-maintainer) for the broader gap.

If you want to learn YAML-based composition right now without dealing with that, use your `quickstart.yaml` from Step 7 — it uses only the `nn.*` types that are always registered.

---

## Variant 2 — Add a custom block, register it, use it from YAML

You'll do this without modifying any code inside `model_constructor/`. The mechanism (`imports:` in YAML) is the same one a parent training/inference repo uses in production.

### Step 1. Pick a parent directory

Anywhere outside `model_constructor/` is fine. We'll use the repo root for the Quickstart. Make sure it's on `PYTHONPATH`:

```bash
export PYTHONPATH="$PWD:$PYTHONPATH"
```

### Step 2. Create the custom-block module

Create `my_quickstart_blocks.py` in the repo root with this exact content:

```python
# my_quickstart_blocks.py
import torch


class GatedResidual(torch.nn.Module):
    """y = x + gate * f(x). gate is a learnable scalar parameter."""

    def __init__(self, *, width: int, init: float = 0.0):
        super().__init__()
        self.fc = torch.nn.Linear(width, width)
        self.act = torch.nn.SiLU()
        self.gate = torch.nn.Parameter(torch.tensor(float(init)))

    def forward(self, x):
        return x + self.gate * self.act(self.fc(x))


def register(registry):
    registry.register_module(
        "my_quickstart_blocks.gated_residual",
        GatedResidual,
        signature_policy="strict",
        tags=("custom",),
    )
```

Key points:

- The constructor uses **keyword-only arguments** (`*, width, init`). The instantiator validates kwargs against this signature; missing or extra kwargs in YAML will raise a clear `ConfigError`.
- The module name (`my_quickstart_blocks`) and the registry key (`my_quickstart_blocks.gated_residual`) share a prefix. That prefix needs to be in `settings.allowed_import_prefixes` in the YAML (Step 3).
- `register(registry)` is the well-known hook name. When `build_model()` processes `imports: [my_quickstart_blocks]`, it imports the module and calls this function automatically — see [`util/imports.py`](../model_constructor/util/imports.py) line 54.

### Step 3. Write a YAML that uses your block

Create `quickstart_custom.yaml` in the repo root:

```yaml
schema_version: 1

settings:
  allowed_import_prefixes: ["model_constructor.", "my_quickstart_blocks"]

imports:
  - my_quickstart_blocks

params:
  width: 32

model:
  graph:
    inputs: [x]
    modules:
      proj:  {_type_: nn.Linear, in_features: ${params.width}, out_features: ${params.width}}
      block: {_type_: my_quickstart_blocks.gated_residual, width: ${params.width}, init: 0.1}
    nodes:
      - {name: h1, call: module:proj,  args: [$x]}
      - {name: h2, call: module:block, args: [$h1]}
    outputs: [$h2]
    return: single
```

What each part is doing:

- `settings.allowed_import_prefixes` adds `my_quickstart_blocks` to the import allowlist. By default only `model_constructor.*` is allowed (safety guardrail).
- `imports: [my_quickstart_blocks]` runs the registration hook.
- `params.width: 32` is a value referenced via `${params.width}` interpolation. The interpolation is type-preserving when it's the entire scalar value (so `width: ${params.width}` becomes `width: 32`, an int — not the string `"32"`).
- The two `nodes` chain `proj → block`.

### Step 4. Build and run

```python
import torch
from model_constructor import build_model

model = build_model("quickstart_custom.yaml")
x = torch.randn(8, 32)        # (batch=8, width=32)
y = model(x)
print(y.shape)                # torch.Size([8, 32])

# Confirm the parameter you defined is there:
print("Trainable params:")
for name, p in model.named_parameters():
    print(f"  {name:50s} shape={tuple(p.shape)}")
```

You should see entries that include `graph_modules.block.gate` (the scalar gate) and `graph_modules.block.fc.weight`, `graph_modules.block.fc.bias`.

### Step 5. Tweak and observe

Edit `quickstart_custom.yaml`: change `params.width: 32` to `params.width: 64`. Re-run. Both modules (`proj` and `block`) pick up the new width automatically via `${params.width}` — you changed one number, not three.

---

## You finished. What to read next.

If everything above worked, here's the recommended next-read order:

1. **[MENTAL_MODEL.md](MENTAL_MODEL.md)** — the conceptual map. Read this before any deeper component doc; it'll save you time.
2. **[../model_constructor/config/authoring_yaml.md](../model_constructor/config/authoring_yaml.md)** — every YAML feature (`defaults` composition, `_template_`, `_merge_`, `${...}` interpolation in full detail).
3. **[../model_constructor/blocks/README.md](../model_constructor/blocks/README.md)** — Option A vs Option B for adding blocks. You did Option B above; Option A is for forks/submodule edits.

If your goal is to use experimental policy blocks (CFG-VQVAE, VFP, DSRL, ResFit, OpenPI), read [../model_constructor/blocks/experiments/README.md](../model_constructor/blocks/experiments/README.md) next.

If your goal is to embed this repo in a training or inference codebase, read [../examples/end_to_end.md](../examples/end_to_end.md).
