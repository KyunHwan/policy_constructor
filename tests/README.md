# `tests/`

`pytest` test suite for the `model_constructor` package.

## Running tests

```bash
# From the repository root
pytest

# Verbose output
pytest -v

# Single file
pytest tests/test_build_and_run.py

# Single test
pytest tests/test_build_and_run.py::test_build_model_sequential_identity_runs
```

Configuration is in [`pyproject.toml`](../pyproject.toml):

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["."]
```

The `pythonpath = ["."]` line means `pytest` automatically prepends the repo root to `sys.path`, so you don't need to set `PYTHONPATH` when running tests (but you do need it for ad-hoc Python scripts — see [QUICKSTART.md](../docs/QUICKSTART.md) Step 2).

## Existing test files — quick map (use these as templates)

When you're writing a new test, find the closest match in the table and read it before writing.

| File | What it covers | Pattern to imitate when... |
|---|---|---|
| [`test_build_and_run.py`](test_build_and_run.py) | End-to-end `build_model()` + forward pass on a tiny model | You want a smoke test for a new block — build a minimal graph using your block and assert the output shape. |
| [`test_graph_compiler_contracts.py`](test_graph_compiler_contracts.py) | `compile_ir()` compile-time contracts (forward-reference rejection, mapping `nodes:` requires `order:`) | You want to assert that an invalid YAML raises a specific `ConfigError`. |
| [`test_defaults_and_merge.py`](test_defaults_and_merge.py) | `resolve_config()` `defaults:` composition and cycle detection | You want to test composition behavior (parent/child YAML files). Uses `tmp_path`. |
| [`test_templates_and_interpolation.py`](test_templates_and_interpolation.py) | `_template_` expansion, `${...}` interpolation, type preservation, typed env vars | You're adding a new YAML feature (template, merge directive, interpolation form). Uses `tmp_path` + `monkeypatch.setenv`. |

### Existing test inventory

`test_build_and_run.py`:

- `test_build_model_sequential_identity_runs` — Builds a `sequential` model with `nn.Identity`; verifies output shape matches input.
- `test_build_model_graph_skip_add_runs` — Builds a `graph` model with `nn.Identity` + `op:add`; verifies `y == x + x`.

`test_graph_compiler_contracts.py`:

- `test_nodes_mapping_requires_order` — `model.graph.nodes` as a mapping without `order:` must raise `ConfigError`.
- `test_forward_reference_forbidden` — A node referencing a not-yet-produced value must raise `ConfigError`.

`test_defaults_and_merge.py`:

- `test_defaults_merge_and_cycle` — Child config inherits and overrides parent `params:`; circular `defaults:` includes raise `ConfigError`.
- `test_merge_list_append` — `_merge_: append` on `imports:` appends child entries onto base entries.

`test_templates_and_interpolation.py`:

- `test_template_expansion` — `_template_` expansion with local overrides (template values merged, local wins).
- `test_interpolation_type_preservation` — `${...}` interpolation preserves types for full-scalar references and stringifies for embedded references. Tests typed env vars (`${env:int:...}`).
- `test_interpolation_cycle_error` — Cyclic `${a}` ↔ `${b}` references raise `ConfigError`.

## Key imports used in tests

| Import | Source |
|---|---|
| `build_model` | `model_constructor` ([`model_constructor/api.py`](../model_constructor/api.py)) |
| `compile_ir` | `model_constructor` ([`model_constructor/api.py`](../model_constructor/api.py)) |
| `resolve_config` | `model_constructor.config.resolve` ([`model_constructor/config/resolve.py`](../model_constructor/config/resolve.py)) |
| `ConfigError` | `model_constructor.errors` ([`model_constructor/errors.py`](../model_constructor/errors.py)) |

---

## How to add a test for your new block

Below is a copy-pasteable template that exercises a new block end-to-end. Replace `my_project.my_block.MyBlock` and the registry key `my_project.my_block` with your own values; the rest is intentionally generic.

### Template — file `tests/test_my_block.py`

```python
from __future__ import annotations

from pathlib import Path

import pytest
import torch

from model_constructor import build_model
from model_constructor.errors import ConfigError


# --- Helpers ------------------------------------------------------------

def _register_my_block() -> None:
    """Register MyBlock into the default registry.

    Call this from any test that needs MyBlock available. Idempotent
    in practice (a duplicate-key error is fine if you guard against it,
    or just use a fresh Registry per test — see _make_registry below).
    """
    from model_constructor.registry.default_registry import get_default_registry

    reg = get_default_registry()
    if "my_project.my_block" in reg.list_modules():
        return  # already registered
    from my_project.my_block import MyBlock
    reg.register_module(
        "my_project.my_block",
        MyBlock,
        signature_policy="strict",
        tags=("custom",),
    )


def _make_registry():
    """Build a fresh registry for tests that want isolation."""
    from model_constructor.registry.builtins import register_builtins
    from model_constructor.registry.registry import Registry

    reg = Registry()
    register_builtins(reg)
    from my_project.my_block import MyBlock
    reg.register_module("my_project.my_block", MyBlock, signature_policy="strict")
    return reg


# --- Happy-path forward pass --------------------------------------------

def test_my_block_forward_pass_shape() -> None:
    _register_my_block()
    cfg = {
        "schema_version": 1,
        "model": {
            "sequential": {
                "inputs": ["x"],
                "layers": [
                    {"_type_": "my_project.my_block", "width": 32, "init": 0.1},
                ],
            }
        },
    }
    model = build_model(cfg)

    x = torch.randn(4, 32)        # (batch=4, width=32)
    y = model(x)

    assert y.shape == x.shape


# --- Numerical correctness against a hand-built reference ---------------

def test_my_block_matches_reference() -> None:
    _register_my_block()
    cfg = {
        "schema_version": 1,
        "model": {
            "sequential": {
                "inputs": ["x"],
                "layers": [
                    {"_type_": "my_project.my_block", "width": 8, "init": 0.0},
                ],
            }
        },
    }
    torch.manual_seed(0)
    model = build_model(cfg)

    # Build a direct reference from the same class, with the same weights.
    from my_project.my_block import MyBlock
    torch.manual_seed(0)
    reference = MyBlock(width=8, init=0.0)
    reference.load_state_dict(model.graph_modules["layer0"].state_dict())

    x = torch.randn(2, 8)
    y_model = model(x)
    y_ref = reference(x)

    torch.testing.assert_close(y_model, y_ref)


# --- Error path: missing required kwarg ---------------------------------

def test_my_block_missing_kwarg_raises() -> None:
    _register_my_block()
    cfg = {
        "schema_version": 1,
        "model": {
            "sequential": {
                "inputs": ["x"],
                # MyBlock requires `width` — omit it on purpose.
                "layers": [{"_type_": "my_project.my_block", "init": 0.1}],
            }
        },
    }
    with pytest.raises(ConfigError, match="width"):
        build_model(cfg)


# --- YAML round-trip via tmp_path ---------------------------------------

def test_my_block_from_yaml(tmp_path: Path) -> None:
    _register_my_block()
    cfg = tmp_path / "model.yaml"
    cfg.write_text(
        """
schema_version: 1
settings:
  allowed_import_prefixes: ["model_constructor.", "my_project."]
imports:
  - my_project.my_block_register   # only if you have a parent-repo register()
params:
  width: 16
model:
  sequential:
    inputs: [x]
    layers:
      - _type_: my_project.my_block
        width: ${params.width}
        init: 0.0
""".lstrip()
    )
    model = build_model(cfg)
    x = torch.randn(1, 16)
    y = model(x)
    assert y.shape == x.shape
```

### What each test demonstrates

- **`test_my_block_forward_pass_shape`** — the bare minimum. Builds the model, runs one batch through, checks the shape. If this passes, the block instantiates and its `forward()` runs.
- **`test_my_block_matches_reference`** — bit-exact numerical check against a hand-built instance of the same class with the same weights. Use `torch.testing.assert_close` (which honors relative and absolute tolerances by default for floating-point). Use `torch.manual_seed` before each `nn.Module.__init__` to keep weights deterministic.
- **`test_my_block_missing_kwarg_raises`** — error-path test. Use `pytest.raises(ConfigError, match=<substring>)` to assert both the exception class and a substring of the message. Substring matching protects against accidental message rewording masking a real regression.
- **`test_my_block_from_yaml`** — round-trip via a real YAML file written to `tmp_path`. Use this when your block uses `${params.X}` interpolation, `defaults:` composition, or templates — features only fully exercised by reading from disk.

### Conventions used by existing tests

- **Inline Python dicts** when possible. They produce smaller, more focused failure messages.
- **`tmp_path`** for YAML files. Don't commit test fixtures under `configs/`.
- **`pytest.raises(ConfigError, match=...)`** for error paths. Always include a `match=` substring — bare `pytest.raises(ConfigError)` accepts any `ConfigError` and won't catch regressions where a different error message replaces yours.
- **`monkeypatch.setenv`** for tests that read env vars (see `test_interpolation_type_preservation`).
- **No global state mutation.** If a test needs a custom registry entry, prefer building a fresh `Registry` (the `_make_registry` helper above) or guarding with `if name in reg.list_modules(): return`.

### Picking the right `build_model` / `compile_ir` / `resolve_config` entry point

| Use case | Function |
|---|---|
| Full end-to-end including instantiation + forward pass | `build_model(cfg)` |
| Compile-time IR-level checks (don't instantiate modules) | `compile_ir(cfg)` |
| Resolution-only checks (defaults, templates, interpolation, schema) | `resolve_config(cfg)` |

For compile-time-only tests, `compile_ir()` skips instantiation, which means a registry key referenced by the YAML doesn't need a real factory class — it just needs to be registered. Useful for testing graph topology rules without worrying about block constructors.
