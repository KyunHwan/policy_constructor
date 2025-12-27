from __future__ import annotations

from ..registry.registry import Registry


def register_blocks(registry: Registry) -> None:
    from .basic_blocks.conv import ConvBnAct
    from .basic_blocks.mlp import MLP
    from .basic_blocks.residual import ResidualBlock

    from .experiments.cfg_vqvae_flow_matching import (
        action_decoder,
        conditioning_info_encoder,
        vq_vae_multimodal_prior,
        vq_vae_multimodal_posterior,
    )

    registry.register_module("cfg_vqvae_action_decoder", action_decoder.ActionDecoder, signature_policy="strict", tags=("experimental", "decoder"))
    registry.register_module("cfg_vqvae_info_encoder", conditioning_info_encoder.ConditioningInfoEncoder, signature_policy="strict", tags=("experimental", "encoder"))
    registry.register_module("cfg_vqvae_prior", vq_vae_multimodal_prior.VQVAE_Prior, signature_policy="strict", tags=("experimental", "prior"))
    registry.register_module("cfg_vqvae_posterior", vq_vae_multimodal_posterior.VQVAE_Posterior, signature_policy="strict", tags=("experimental", "posterior"))

    # registry.register_module("conv_bn_act", ConvBnAct, signature_policy="strict", tags=("blocks", "conv"))
    # registry.register_module("mlp", MLP, signature_policy="strict", tags=("blocks", "mlp"))
    # registry.register_module("residual_block", ResidualBlock, signature_policy="strict", tags=("blocks", "residual"))
