from __future__ import annotations

from pathlib import Path

import pytest

from model_constructor.config.resolve import resolve_config
from model_constructor.errors import ConfigError


def test_defaults_merge_and_cycle(tmp_path: Path) -> None:
    base = tmp_path / "base.yaml"
    child = tmp_path / "child.yaml"

    base.write_text(
        """
schema_version: 1
params:
  a: 1
model:
  sequential:
    layers:
      - {_type_: nn.Identity}
""".lstrip()
    )

    child.write_text(
        f"""
defaults:
  - {base.name}
schema_version: 1
params:
  b: 2
model:
  sequential:
    layers:
      - {{_type_: nn.Identity}}
""".lstrip()
    )

    resolved = resolve_config(child)
    assert resolved.data["params"] == {"a": 1, "b": 2}

    a = tmp_path / "a.yaml"
    b = tmp_path / "b.yaml"
    a.write_text("defaults: [b.yaml]\nschema_version: 1\nmodel: {sequential: {layers: [{_type_: nn.Identity}]}}\n")
    b.write_text("defaults: [a.yaml]\nschema_version: 1\nmodel: {sequential: {layers: [{_type_: nn.Identity}]}}\n")

    with pytest.raises(ConfigError, match="defaults include cycle detected"):
        resolve_config(a)


def test_merge_list_append(tmp_path: Path) -> None:
    base = tmp_path / "base.yaml"
    child = tmp_path / "child.yaml"

    base.write_text(
        """
schema_version: 1
imports:
  - model_constructor.blocks.register
model:
  sequential:
    layers:
      - {_type_: nn.Identity}
""".lstrip()
    )

    child.write_text(
        f"""
defaults:
  - {base.name}
schema_version: 1
imports:
  _merge_: append
  _value_:
    - model_constructor.blocks.register
model:
  sequential:
    layers:
      - {{_type_: nn.Identity}}
""".lstrip()
    )

    resolved = resolve_config(child)
    assert resolved.data["imports"] == ["model_constructor.blocks.register", "model_constructor.blocks.register"]

