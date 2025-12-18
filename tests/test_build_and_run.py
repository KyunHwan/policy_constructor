from __future__ import annotations

import torch

from model_constructor import build_model


def test_build_model_sequential_identity_runs() -> None:
    cfg = {
        "schema_version": 1,
        "model": {"sequential": {"layers": [{"_type_": "nn.Identity"}]}},
    }
    m = build_model(cfg)
    x = torch.randn(2, 3)
    y = m(x)
    assert y.shape == x.shape


def test_build_model_graph_skip_add_runs() -> None:
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
            }
        },
    }
    m = build_model(cfg)
    x = torch.randn(2, 3)
    y = m(x)
    assert torch.allclose(y, x + x)

