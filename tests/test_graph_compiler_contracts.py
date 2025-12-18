from __future__ import annotations

from pathlib import Path

import pytest

from model_constructor import compile_ir
from model_constructor.errors import ConfigError


def test_nodes_mapping_requires_order() -> None:
    cfg = {
        "schema_version": 1,
        "model": {
            "graph": {
                "inputs": ["x"],
                "modules": {"m": {"_type_": "nn.Identity"}},
                "nodes": {"h1": {"call": "module:m", "args": ["$x"]}},
                "outputs": ["$h1"],
                "return": "single",
            }
        },
    }
    with pytest.raises(ConfigError, match="requires 'order"):
        compile_ir(cfg)


def test_forward_reference_forbidden() -> None:
    cfg = {
        "schema_version": 1,
        "model": {
            "graph": {
                "inputs": ["x"],
                "modules": {"m": {"_type_": "nn.Identity"}},
                "nodes": [
                    {"name": "h2", "call": "module:m", "args": ["$h1"]},
                    {"name": "h1", "call": "module:m", "args": ["$x"]},
                ],
                "outputs": ["$h2"],
                "return": "single",
            }
        },
    }
    with pytest.raises(ConfigError, match="Forward reference"):
        compile_ir(cfg)

