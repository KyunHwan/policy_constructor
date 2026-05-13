# `third_party/` — Vendored OpenPI codebase

This directory contains **vendored upstream code** from the Physical Intelligence [OpenPI](https://github.com/Physical-Intelligence/openpi) project. Treat the contents of [`openpi/`](openpi/) and [`openpi_client/`](openpi_client/) as **out-of-scope for editing** — we don't modify them, we don't redocument them in place. This README explains what's here, what to touch, and what to leave alone.

If you're new to the term **OpenPI / PI0**, see [docs/GLOSSARY.md](../../../../docs/GLOSSARY.md#openpi--pi0).

---

## What lives here

```
third_party/
├── __init__.py
├── openpi/                       # vendored upstream package
│   ├── conftest.py               #   upstream pytest config (NOT used by this repo's tests)
│   ├── models/                   #   upstream model definitions
│   ├── models_pytorch/           #   upstream PyTorch port of PI0
│   ├── policies/                 #   upstream policy classes
│   ├── serving/                  #   upstream serving utilities
│   ├── shared/                   #   upstream shared helpers (normalize.py, image_tools.py, etc.)
│   ├── training/                 #   upstream training scaffolding (NOT used here — see below)
│   ├── transforms.py
│   ├── openpi_batched_wrapper.py          # ← the ONE file this repo authored
│   └── openpi_batched_wrapper_test.py     # ← upstream-style test for the wrapper
└── openpi_client/                # vendored upstream client package
    ├── __init__.py
    ├── action_chunk_broker.py
    ├── base_policy.py
    ├── image_tools.py
    ├── msgpack_numpy.py
    ├── runtime/                  # upstream runtime/agent/subscriber/environment helpers
    └── websocket_client_policy.py
```

Inside `openpi/` and `openpi_client/`, every file except `openpi_batched_wrapper.py` and `openpi_batched_wrapper_test.py` is upstream code carried verbatim.

---

## Why it's vendored (not pip-installed)

Best guess from reading the wrapper: **vendoring controls the exact version used by `openpi_batched`**. The OpenPI APIs (`openpi.models`, `openpi.training.config`, `openpi.policies.igris_policy`) are evolving research-grade interfaces, and tying our wrapper to a specific commit avoids surprise breakage when the upstream refactors.

The wrapper also injects `sys.path` manipulation at import time so that `from openpi... import ...` resolves to *this* vendored copy rather than any system-installed `openpi` ([`openpi/openpi_batched_wrapper.py`](openpi/openpi_batched_wrapper.py) ~line 26):

```python
_THIRD_PARTY = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _THIRD_PARTY not in sys.path:
    sys.path.insert(0, _THIRD_PARTY)
```

> **Note:** Verify with maintainer: the exact rationale for vendoring (was it version pinning, license requirements, or build-system avoidance?). Until confirmed, treat "version pinning" as the working hypothesis.

---

## The only piece a normal user touches

`OpenPiBatchedWrapper` ([`openpi/openpi_batched_wrapper.py`](openpi/openpi_batched_wrapper.py) line 83) is the only thing in this directory exposed via the registry — it's registered in [`../../register.py`](../../register.py) as `_type_: openpi_batched`.

The wrapper:

- Takes the same input fields the upstream `Pi05IgrisVlaAdapter` expects, plus a batch dimension.
- Loads `PI0Pytorch` from `openpi.models_pytorch.pi0_pytorch`, optionally with pretrained weights from a `model.safetensors` checkpoint.
- Loads normalization stats from a checkpoint's `assets/<asset_id>/norm_stats.json` and applies them via the upstream normalization transforms.
- Exposes a `forward(...)` that returns batched action predictions.

### Constructor signature

```python
OpenPiBatchedWrapper(
    train_config_name: str = "pi05_igris",
    ckpt_dir: str | None = None,            # path to checkpoint dir; None → random weights
    default_prompt: str | None = None,      # language instruction if obs has none
    camera_names: list[str] | None = None,  # default: ["head", "left", "right"]
    action_dim: int = 24,
    action_horizon: int = 50,
    num_inference_steps: int = 10,
    gradient_checkpointing: bool = False,
)
```

If `ckpt_dir` is set, the wrapper looks for `ckpt_dir/model.safetensors` and `ckpt_dir/assets/<asset_id>/norm_stats.json`. If either is missing, the wrapper logs a warning and proceeds with whatever it found (random weights, no norm stats).

### Optional dependencies — when they're needed

| Dependency | Required when... |
|---|---|
| `safetensors` | `ckpt_dir` is set (used to load `model.safetensors`). |
| `jax` (CPU) | OpenPI's training-config machinery imports JAX. The wrapper sets `JAX_PLATFORMS=cpu` at import time to avoid GPU contention; `XLA_PYTHON_CLIENT_PREALLOCATE=false` to avoid eager VRAM allocation. You don't need a GPU JAX install. |
| Upstream OpenPI deps | Whatever `openpi/pyproject.toml` (inside the vendored copy) declares. |

If you import `OpenPiBatchedWrapper` and JAX isn't installed, instantiation will fail with a clear `ImportError` from the OpenPI side — install JAX CPU and retry.

### Minimal YAML usage

```yaml
schema_version: 1
model:
  graph:
    inputs: [obs]                     # whatever shape the OpenPI policy expects
    modules:
      pi0:
        _type_: openpi_batched
        train_config_name: pi05_igris
        ckpt_dir: /path/to/checkpoint
        action_dim: 24
        action_horizon: 50
        num_inference_steps: 10
    nodes:
      - {name: action, call: module:pi0, args: [$obs]}
    outputs: [$action]
    return: single
```

---

## Caution — upstream tests are not part of this repo's `pytest` run

Several `*_test.py` files exist inside the vendored tree (e.g., `openpi/transforms_test.py`, `openpi/shared/normalize_test.py`, `openpi_client/image_tools_test.py`). They are **upstream tests**, written for OpenPI's own test infrastructure. They are not collected by this repo's `pytest` run because:

- [`pyproject.toml`](../../../../pyproject.toml) sets `testpaths = ["tests"]`, scoping discovery to the top-level `tests/` directory.
- `third_party/openpi/conftest.py` is upstream-specific configuration unrelated to our test suite.

Don't try to run them via `pytest` from the repo root; they may fail on imports or environmental assumptions that the OpenPI project's own CI provides. If you need them, run them from inside the OpenPI submodule's own setup.

`openpi/openpi_batched_wrapper_test.py` is a wrapper-specific test that *might* work from the repo root, but it depends on JAX, safetensors, and likely a checkpoint path — treat it as a developer aid, not part of the canonical test suite.

---

## Don't edit files in `openpi/` or `openpi_client/`

For the same reason we don't edit upstream code in the [`backbones/vision/externals/depth_anything_3/`](../backbones/vision/externals/depth_anything_3/) submodule: those directories belong to upstream projects with their own licenses, release cadences, and maintainers. Local edits will:

- Drift from upstream and become hard to update.
- Conflict with future re-vendoring.
- Potentially violate license terms.

The only repo-authored file in this whole tree is [`openpi/openpi_batched_wrapper.py`](openpi/openpi_batched_wrapper.py). All other modifications should happen there, in a parent-repo wrapper, or in a fork.

---

## Upstream

- OpenPI project: [github.com/Physical-Intelligence/openpi](https://github.com/Physical-Intelligence/openpi)
- Bug reports about anything other than the wrapper should go upstream.
- The vendored snapshot's commit hash is not tracked separately by this repo; the contents are what they are.

---

## What about `checkpoints/`?

`third_party/checkpoints/` exists as a placeholder for downloaded weights / norm-stats assets used by `OpenPiBatchedWrapper`. Don't commit large weight files there. The directory is intentionally outside this repo's documentation scope — its layout is whatever the upstream OpenPI checkpoint format specifies (typically `<asset_id>/model.safetensors` + `<asset_id>/assets/<asset_id>/norm_stats.json`).
