from __future__ import annotations

import os
from pathlib import Path

import pytest

from model_constructor.config.resolve import resolve_config
from model_constructor.errors import ConfigError


def test_template_expansion(tmp_path: Path) -> None:
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(
        """
schema_version: 1
templates:
  lin:
    _type_: nn.Linear
    in_features: 4
    out_features: 8
model:
  sequential:
    layers:
      - _template_: lin
        out_features: 16
""".lstrip()
    )

    resolved = resolve_config(cfg)
    layer0 = resolved.data["model"]["sequential"]["layers"][0]
    assert layer0["_type_"] == "nn.Linear"
    assert layer0["in_features"] == 4
    assert layer0["out_features"] == 16
    assert "templates" not in resolved.data


def test_interpolation_type_preservation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MC_INT", " 08 ")
    monkeypatch.setenv("MC_RAW", "  x ")

    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(
        """
schema_version: 1
params:
  width: 32
model:
  graph:
    inputs: [x]
    modules:
      m:
        _type_: conv_bn_act
        in_channels: null
        out_channels: ${params.width}
        kernel_size: 3
        padding: 1
    nodes:
      h1: {call: module:m, args: [$x]}
    order: [h1]
    outputs: [$h1]
    return: single
env_tests:
  a: ${env:int:MC_INT}
  b: ${env:MC_RAW}
  c: "hello_${params.width}"
""".lstrip()
    )

    resolved = resolve_config(cfg)
    assert resolved.data["env_tests"]["a"] == 8
    assert resolved.data["env_tests"]["b"] == "  x "
    assert resolved.data["env_tests"]["c"] == "hello_32"

    # Full-scalar interpolation preserves type (int)
    assert resolved.data["model"]["graph"]["modules"]["m"]["out_channels"] == 32


def test_interpolation_cycle_error(tmp_path: Path) -> None:
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(
        """
schema_version: 1
a: ${b}
b: ${a}
model:
  sequential:
    layers:
      - {_type_: nn.Identity}
""".lstrip()
    )

    with pytest.raises(ConfigError, match="Interpolation cycle detected"):
        resolve_config(cfg)
