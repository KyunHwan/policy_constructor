from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model_constructor import build_model


def main() -> None:
    model = build_model("configs/examples/graph_skip_add.yaml")
    x = torch.randn(2, 3, 32, 32)
    y = model(x)
    print(type(model).__name__, "->", tuple(y.shape))


if __name__ == "__main__":
    main()
