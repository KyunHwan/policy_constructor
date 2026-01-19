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
        conditioning_info_encoder as cfg_vqvae_conditioning_info_encoder,
        vq_vae_multimodal_prior as cfg_vqvae_prior,
        vq_vae_multimodal_posterior as cfg_vqvae_posterior,
        vq_vae_codebook_manager as cfg_vqvae_codebook_manager,
        proprio_projector as cfg_vqvae_proprio_projector
    )
    
    registry.register_module("cfg_vqvae_action_decoder", cfg_vqvae_action_decoder.ActionDecoder, signature_policy="strict", tags=("experimental", "decoder"))
    registry.register_module("cfg_vqvae_info_encoder", cfg_vqvae_conditioning_info_encoder.ConditioningInfoEncoder, signature_policy="strict", tags=("experimental", "encoder"))
    registry.register_module("cfg_vqvae_prior", cfg_vqvae_prior.VQVAE_Prior, signature_policy="strict", tags=("experimental", "prior"))
    registry.register_module("cfg_vqvae_posterior", cfg_vqvae_posterior.VQVAE_Posterior, signature_policy="strict", tags=("experimental", "posterior"))
    registry.register_module("cfg_vqvae_codebook", cfg_vqvae_codebook_manager.VQCodebookManager, signature_policy="strict", tags=("experimental", "vqcodebook"))
    registry.register_module("cfg_vqvae_proprio_projector", cfg_vqvae_proprio_projector.ProprioProjector, signature_policy="strict", tags=("experimental", "projection"))


    """ Naive Flow Matching Policy """
    from .experiments.naive_flow_matching_policy import (
        action_decoder as naive_action_decoder,
        proprio_projector as naive_proprio_projector
    )

    registry.register_module("naive_action_decoder", naive_action_decoder.ActionDecoder, signature_policy="strict", tags=("experimental", "decoder"))
    registry.register_module("naive_proprio_projector", naive_proprio_projector.ProprioProjector, signature_policy="strict", tags=("experimental", "projection"))
    
    """ VFP """
    from .experiments.variational_flow_matching_policy import (
        experts as vfp_experts,
        gate as vfp_gate,
        posterior as vfp_posterior,
        prior as vfp_prior,
        vq_vae_codebook_manager as vfp_vqvae_codebook,
        proprio_projector as vfp_proprio_projector
    )

    registry.register_module("vfp_moe", vfp_experts.MoE, signature_policy="strict", tags=("experimental", "moe"))
    registry.register_module("vfp_gate", vfp_gate.Gate, signature_policy="strict", tags=("experimental", "gate"))
    registry.register_module("vfp_posterior", vfp_posterior.VQVAE_Posterior, signature_policy="strict", tags=("experimental", "posterior"))
    registry.register_module("vfp_prior", vfp_prior.VQVAE_Prior, signature_policy="strict", tags=("experimental", "prior"))
    registry.register_module("vfp_vqvae_codebook", vfp_vqvae_codebook.VQCodebookManager, signature_policy="strict", tags=("experimental", "vqcodebook"))
    registry.register_module("vfp_proprio_projector", vfp_proprio_projector.ProprioProjector, signature_policy="strict", tags=("experimental", "projection"))

    """ VFP """
    from .experiments.mutual_inf_est import (
        action_decoder as a_decoder,
        action_encoder as a_encoder,
        state_resnet34_decoder as state_decoder,
        state_resnet34_encoder as state_encoder,
    )

    registry.register_module("a_decoder", a_decoder.ActionDecoder, signature_policy="strict", tags=("experimental", "action_decoder"))
    registry.register_module("a_encoder", a_encoder.ActionEncoder, signature_policy="strict", tags=("experimental", "action_encoder"))
    registry.register_module("state_decoder", state_decoder.ResNet34DecoderGroup, signature_policy="strict", tags=("experimental", "state_decoder"))
    registry.register_module("state_encoder", state_encoder.ResNet34EncoderGroup, signature_policy="strict", tags=("experimental", "state_encoder"))