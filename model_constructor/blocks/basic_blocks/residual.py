from __future__ import annotations

from typing import Any

import torch

from .conv import ConvBnAct


class ResidualBlock(torch.nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: int | None = None,
        stride: int = 1,
        act: torch.nn.Module | None = None,
    ) -> None:
        super().__init__()

        if out_channels is None:
            out_channels = in_channels
        if act is None:
            act = torch.nn.ReLU(inplace=True)

        self.main1 = ConvBnAct(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
        )
        self.main2 = ConvBnAct(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            act=torch.nn.Identity(),
        )

        if in_channels == out_channels and stride == 1:
            self.skip: torch.nn.Module = torch.nn.Identity()
        else:
            self.skip = ConvBnAct(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=stride,
                padding=0,
                act=torch.nn.Identity(),
            )

        self.act = act

    def forward(self, x: Any) -> Any:
        y = self.main1(x)
        y = self.main2(y)
        y = y + self.skip(x)
        return self.act(y)

