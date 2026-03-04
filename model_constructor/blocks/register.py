from __future__ import annotations

from ..registry.registry import Registry


def register_blocks(registry: Registry) -> None:

    """ Backbones """
    # Vision 
    from .experiments.backbones.vision.radiov3 import RadioV3
    from .experiments.backbones.vision.depth_anything_3 import DepthAnything3Bridge
    from.experiments.backbones.vision.resnet34 import Resnet34

    registry.register_module("resnet34", Resnet34, signature_policy="strict", tags=("experimental", "backbone"))
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
        proprio_projector as naive_proprio_projector,
        info_embedder as naive_info_embedder,
        hand_extractor as naive_hand_extractor,
    )

    registry.register_module("naive_hand_extractor", naive_hand_extractor.ResNet34Encoder, signature_policy="strict", tags=("experimental", "embedding"))
    registry.register_module("naive_info_embedder", naive_info_embedder.InfoEmbedder, signature_policy="strict", tags=("experimental", "embedding"))
    registry.register_module("naive_action_decoder", naive_action_decoder.ActionDecoder, signature_policy="strict", tags=("experimental", "decoder"))
    registry.register_module("naive_proprio_projector", naive_proprio_projector.ProprioProjector, signature_policy="strict", tags=("experimental", "projection"))
    
    """ VFP """
    from .experiments.variational_flow_matching_policy import (
        experts as vfp_experts,
        gate as vfp_gate,
        posterior as vfp_posterior,
        prior as vfp_prior,
        vq_vae_codebook_manager as vfp_vqvae_codebook,
        proprio_projector as vfp_proprio_projector,
        info_embedder as vfp_info_embedder,
        hand_extractor as vfp_hand_extractor,
    )

    registry.register_module("vfp_hand_extractor", vfp_hand_extractor.ResNet34Encoder, signature_policy="strict", tags=("experimental", "embedding"))
    registry.register_module("vfp_info_embedder", vfp_info_embedder.InfoEmbedder, signature_policy="strict", tags=("experimental", "embedding"))
    registry.register_module("vfp_moe", vfp_experts.MoE, signature_policy="strict", tags=("experimental", "moe"))
    registry.register_module("vfp_gate", vfp_gate.Gate, signature_policy="strict", tags=("experimental", "gate"))
    registry.register_module("vfp_posterior", vfp_posterior.VQVAE_Posterior, signature_policy="strict", tags=("experimental", "posterior"))
    registry.register_module("vfp_prior", vfp_prior.VQVAE_Prior, signature_policy="strict", tags=("experimental", "prior"))
    registry.register_module("vfp_vqvae_codebook", vfp_vqvae_codebook.VQCodebookManager, signature_policy="strict", tags=("experimental", "vqcodebook"))
    registry.register_module("vfp_proprio_projector", vfp_proprio_projector.ProprioProjector, signature_policy="strict", tags=("experimental", "projection"))

    """ Mutual Information Encoder """
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

    """ VFP """
    from .experiments.vfp_single_expert import (
        action_decoder as vfp_single_action_decoder,
        posterior as vfp_single_posterior,
        prior as vfp_single_prior,
        info_embedder as vfp_single_info_embedder,
        multimodal_bridge as vfp_single_mmb
    )

    registry.register_module("vfp_single_action_decoder", vfp_single_action_decoder.ActionDecoder, signature_policy="strict", tags=("experimental", "decoder"))
    registry.register_module("vfp_single_info_embedder", vfp_single_info_embedder.InfoEmbedder, signature_policy="strict", tags=("experimental", "embedding"))
    registry.register_module("vfp_single_multimodal_bridge", vfp_single_mmb.MultiModalBridge, signature_policy="strict", tags=("experimental", "embedding"))
    registry.register_module("vfp_single_posterior", vfp_single_posterior.VAE_Posterior, signature_policy="strict", tags=("experimental", "posterior"))
    registry.register_module("vfp_single_prior", vfp_single_prior.VAE_Prior, signature_policy="strict", tags=("experimental", "prior"))

    """ OpenPI Batched Wrapper """
    from .experiments.third_party.openpi.openpi_batched_wrapper import OpenPiBatchedWrapper

    registry.register_module("openpi_batched", OpenPiBatchedWrapper, signature_policy="strict", tags=("experimental", "openpi", "vla"))

    """ DSRL """
    from .experiments.dsrl.q_function import Q_Function
    from .experiments.dsrl.noise_latent_actor import Noise_Latent_Actor

    from .experiments.dsrl.q_function_img_depth_proprio_processor import (
        QFunctionImgDepthProprioProcessor,
    )
    from .experiments.dsrl.noise_actor_img_depth_proprio_processor import (
        NoiseActorImgDepthProprioProcessor as NoiseActorImgDepthProprioProcessor_,
    )
    
    registry.register_module("dsrl_q_function", Q_Function, signature_policy="strict", tags=("experimental", "dsrl"))
    registry.register_module("dsrl_noise_latent_actor", Noise_Latent_Actor, signature_policy="strict", tags=("experimental", "dsrl"))
    registry.register_module("dsrl_q_function_processor", QFunctionImgDepthProprioProcessor, signature_policy="strict", tags=("experimental", "dsrl"))
    registry.register_module("dsrl_noise_actor_processor", NoiseActorImgDepthProprioProcessor_, signature_policy="strict", tags=("experimental", "dsrl"))
