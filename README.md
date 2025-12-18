# model_constructor

YAML-first, construct-only PyTorch model architecture constructor.

This repository builds `torch.nn.Module` architectures from `.yaml` files and **does not** include any training or inference engine logic. It is designed to be used as a submodule inside other codebases.

## Quick start

```python
from model_constructor import build_model

model = build_model("configs/examples/sequential_mlp.yaml")
```

## Key ideas

- A single canonical resolution pipeline: `YAML -> ResolvedConfig -> GraphIR -> nn.Module`.
- Safe-by-default configs: use registry keys (`_type_`) rather than import strings.
- Architecture/input/output agnostic: graph execution is defined via a small IR and a generic `GraphModel`.

## Docs

- `docs/README.md`
- `docs/overview.md`
- `model_constructor/config/schema_v1.md`
- `docs/compatibility.md`
