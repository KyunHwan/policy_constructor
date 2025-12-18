from __future__ import annotations

from ..registry.registry import Registry


def register_blocks(registry: Registry) -> None:
    from .conv import ConvBnAct
    from .mlp import MLP
    from .residual import ResidualBlock

    registry.register_module("conv_bn_act", ConvBnAct, signature_policy="strict", tags=("blocks", "conv"))
    registry.register_module("mlp", MLP, signature_policy="strict", tags=("blocks", "mlp"))
    registry.register_module("residual_block", ResidualBlock, signature_policy="strict", tags=("blocks", "residual"))
