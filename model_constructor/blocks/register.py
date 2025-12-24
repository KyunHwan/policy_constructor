from __future__ import annotations

from ..registry.registry import Registry


def register_blocks(registry: Registry) -> None:
    from .basic_blocks.conv import ConvBnAct
    from .basic_blocks.mlp import MLP
    from .basic_blocks.residual import ResidualBlock

    from .experiments.backbones.vision import 
    from .experiments.body.flow_matching.naive_flow_matching import 


    registry.register_module("conv_bn_act", ConvBnAct, signature_policy="strict", tags=("blocks", "conv"))
    registry.register_module("mlp", MLP, signature_policy="strict", tags=("blocks", "mlp"))
    registry.register_module("residual_block", ResidualBlock, signature_policy="strict", tags=("blocks", "residual"))
