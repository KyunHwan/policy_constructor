from __future__ import annotations

from ..registry.registry import Registry


def register_blocks(registry: Registry) -> None:

    """ Backbones """
    # Vision 
    from .experiments.backbones.vision.radiov3 import RadioV3
    from .experiments.backbones.vision.depth_anything_3 import DepthAnything3Bridge

    registry.register_module("radiov3", RadioV3, signature_policy="strict", tags=("experimental", "backbone"))
    registry.register_module("da3", DepthAnything3Bridge, signature_policy="strict", tags=("experimental", "backbone"))

    """ CFG-VQVAE """
    from .experiments.cfg_vqvae_flow_matching import (
        action_decoder as cfg_vqvae_action_decoder,
        conditioning_info_encoder,
        vq_vae_multimodal_prior,
        vq_vae_multimodal_posterior,
        vq_vae_codebook_manager,
        proprio_projector as cfg_vqvae_proprio_projector
    )
    
    registry.register_module("cfg_vqvae_action_decoder", cfg_vqvae_action_decoder.ActionDecoder, signature_policy="strict", tags=("experimental", "decoder"))
    registry.register_module("cfg_vqvae_info_encoder", conditioning_info_encoder.ConditioningInfoEncoder, signature_policy="strict", tags=("experimental", "encoder"))
    registry.register_module("cfg_vqvae_prior", vq_vae_multimodal_prior.VQVAE_Prior, signature_policy="strict", tags=("experimental", "prior"))
    registry.register_module("cfg_vqvae_posterior", vq_vae_multimodal_posterior.VQVAE_Posterior, signature_policy="strict", tags=("experimental", "posterior"))
    registry.register_module("cfg_vqvae_codebook", vq_vae_codebook_manager.VQCodebookManager, signature_policy="strict", tags=("experimental", "vqcodebook"))
    registry.register_module("cfg_vqvae_proprio_projector", cfg_vqvae_proprio_projector.ProprioProjector, signature_policy="strict", tags=("experimental", "projection"))


    """ Naive Flow Matching Policy """
    from .experiments.naive_flow_matching import (
        action_decoder as naive_action_decoder,
        proprio_projector as naive_proprio_projector
    )

    registry.register_module("naive_action_decoder", naive_action_decoder.ActionDecoder, signature_policy="strict", tags=("experimental", "decoder"))
    registry.register_module("naive_proprio_projector", naive_proprio_projector.ProprioProjector, signature_policy="strict", tags=("experimental", "projection"))
    
