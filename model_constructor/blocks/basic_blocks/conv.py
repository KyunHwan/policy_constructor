from __future__ import annotations

from typing import Any

import torch


class ConvBnAct(torch.nn.Module):
    def __init__(
        self,
        *,
        in_channels: int | None,
        out_channels: int,
        kernel_size: int | tuple[int, int] = 3,
        stride: int | tuple[int, int] = 1,
        padding: int | tuple[int, int] = 0,
        dilation: int | tuple[int, int] = 1,
        groups: int = 1,
        bias: bool | None = None,
        norm: torch.nn.Module | None = None,
        act: torch.nn.Module | None = None,
    ) -> None:
        super().__init__()

        if bias is None:
            bias = norm is None

        if in_channels is None:
            conv: torch.nn.Module = torch.nn.LazyConv2d(
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias=bias,
            )
        else:
            conv = torch.nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias=bias,
            )

        if norm is None:
            norm = torch.nn.BatchNorm2d(out_channels)
        if act is None:
            act = torch.nn.ReLU(inplace=True)

        self.conv = conv
        self.norm = norm
        self.act = act

    def forward(self, x: Any) -> Any:
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x

