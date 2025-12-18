from __future__ import annotations

from typing import Any

import torch


class MLP(torch.nn.Module):
    def __init__(
        self,
        *,
        dims: list[int | None],
        activation: torch.nn.Module | None = None,
        dropout: float = 0.0,
        bias: bool = True,
        final_activation: torch.nn.Module | None = None,
    ) -> None:
        super().__init__()

        if not isinstance(dims, list) or len(dims) < 2:
            raise ValueError("dims must be a list of length >= 2")

        if activation is None:
            activation = torch.nn.ReLU()

        layers: list[torch.nn.Module] = []
        for i in range(len(dims) - 1):
            in_f = dims[i]
            out_f = dims[i + 1]
            if out_f is None:
                raise ValueError("dims[-1] and intermediate out features must be integers (not None)")

            if in_f is None:
                linear: torch.nn.Module = torch.nn.LazyLinear(out_f, bias=bias)
            else:
                linear = torch.nn.Linear(in_f, out_f, bias=bias)
            layers.append(linear)

            is_last = i == (len(dims) - 2)
            if not is_last:
                layers.append(activation)
                if dropout and dropout > 0:
                    layers.append(torch.nn.Dropout(dropout))
            elif final_activation is not None:
                layers.append(final_activation)

        self.net = torch.nn.Sequential(*layers)

    def forward(self, x: Any) -> Any:
        return self.net(x)

