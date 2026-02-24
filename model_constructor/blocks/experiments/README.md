# `model_constructor/blocks/experiments/`

Experimental `torch.nn.Module` components for robot policy architectures. All blocks in this directory are registered in the default registry via [`../register.py`](../register.py) with the `"experimental"` tag.

These components implement flow matching, VQ-VAE, mixture-of-experts, vision backbones, and third-party model integrations used in research on robot manipulation policies.

## Subdirectory map

```
experiments/
├── backbones/                     # Vision and VLM feature extractors
│   ├── vision/                    # RadioV3, DepthAnything3, ResNet34, DUNE
│   │   ├── radiov3.py
│   │   ├── depth_anything_3.py
│   │   ├── resnet34.py
│   │   ├── dune.py
│   │   └── externals/             # Git submodule (depth_anything_3)
│   └── vlm/                       # Vision-language model backbones
│       └── cosmos_reason2.py
├── cfg_vqvae_flow_matching/       # CFG-VQVAE + flow matching policy
├── variational_flow_matching_policy/  # Variational flow matching (MoE)
├── vfp_single_expert/             # VFP single-expert variant
├── naive_flow_matching_policy/    # Naive (non-variational) flow matching
├── mutual_inf_est/                # Mutual information estimator
├── templates/                     # Abstract base classes for policy components
├── utils/                         # Shared utilities (positional encoding, time embedding)
└── third_party/                   # Third-party integrations (OpenPI)
```

## Registry keys

All blocks below are registered in [`../register.py`](../register.py) with `signature_policy="strict"`.

### Vision backbones

| Registry key | Class | File | Tags |
|-------------|-------|------|------|
| `radiov3` | `RadioV3` | [`backbones/vision/radiov3.py`](backbones/vision/radiov3.py) | `experimental`, `backbone` |
| `da3` | `DepthAnything3Bridge` | [`backbones/vision/depth_anything_3.py`](backbones/vision/depth_anything_3.py) | `experimental`, `backbone` |

The `depth_anything_3` backbone depends on the external git submodule at `backbones/vision/externals/depth_anything_3`.

### CFG-VQVAE flow matching

Conditional flow matching policy with VQ-VAE latent space.

| Registry key | Class | File | Tags |
|-------------|-------|------|------|
| `cfg_vqvae_action_decoder` | `ActionDecoder` | [`cfg_vqvae_flow_matching/action_decoder.py`](cfg_vqvae_flow_matching/action_decoder.py) | `experimental`, `decoder` |
| `cfg_vqvae_info_encoder` | `ConditioningInfoEncoder` | [`cfg_vqvae_flow_matching/conditioning_info_encoder.py`](cfg_vqvae_flow_matching/conditioning_info_encoder.py) | `experimental`, `encoder` |
| `cfg_vqvae_prior` | `VQVAE_Prior` | [`cfg_vqvae_flow_matching/vq_vae_multimodal_prior.py`](cfg_vqvae_flow_matching/vq_vae_multimodal_prior.py) | `experimental`, `prior` |
| `cfg_vqvae_posterior` | `VQVAE_Posterior` | [`cfg_vqvae_flow_matching/vq_vae_multimodal_posterior.py`](cfg_vqvae_flow_matching/vq_vae_multimodal_posterior.py) | `experimental`, `posterior` |
| `cfg_vqvae_codebook` | `VQCodebookManager` | [`cfg_vqvae_flow_matching/vq_vae_codebook_manager.py`](cfg_vqvae_flow_matching/vq_vae_codebook_manager.py) | `experimental`, `vqcodebook` |
| `cfg_vqvae_proprio_projector` | `ProprioProjector` | [`cfg_vqvae_flow_matching/proprio_projector.py`](cfg_vqvae_flow_matching/proprio_projector.py) | `experimental`, `projection` |

Example config: [`configs/experiments/cfg_vqvae_flow_matching.yaml`](../../../configs/experiments/cfg_vqvae_flow_matching.yaml)

### Variational flow matching policy (MoE)

Mixture-of-experts variant with gate network, VQ-VAE codebook, and variational prior/posterior.

| Registry key | Class | File | Tags |
|-------------|-------|------|------|
| `vfp_hand_extractor` | `ResNet34Encoder` | [`variational_flow_matching_policy/hand_extractor.py`](variational_flow_matching_policy/hand_extractor.py) | `experimental`, `embedding` |
| `vfp_info_embedder` | `InfoEmbedder` | [`variational_flow_matching_policy/info_embedder.py`](variational_flow_matching_policy/info_embedder.py) | `experimental`, `embedding` |
| `vfp_moe` | `MoE` | [`variational_flow_matching_policy/experts.py`](variational_flow_matching_policy/experts.py) | `experimental`, `moe` |
| `vfp_gate` | `Gate` | [`variational_flow_matching_policy/gate.py`](variational_flow_matching_policy/gate.py) | `experimental`, `gate` |
| `vfp_posterior` | `VQVAE_Posterior` | [`variational_flow_matching_policy/posterior.py`](variational_flow_matching_policy/posterior.py) | `experimental`, `posterior` |
| `vfp_prior` | `VQVAE_Prior` | [`variational_flow_matching_policy/prior.py`](variational_flow_matching_policy/prior.py) | `experimental`, `prior` |
| `vfp_vqvae_codebook` | `VQCodebookManager` | [`variational_flow_matching_policy/vq_vae_codebook_manager.py`](variational_flow_matching_policy/vq_vae_codebook_manager.py) | `experimental`, `vqcodebook` |
| `vfp_proprio_projector` | `ProprioProjector` | [`variational_flow_matching_policy/proprio_projector.py`](variational_flow_matching_policy/proprio_projector.py) | `experimental`, `projection` |

### VFP single expert

Single-expert variant of VFP with a multimodal bridge instead of MoE.

| Registry key | Class | File | Tags |
|-------------|-------|------|------|
| `vfp_single_action_decoder` | `ActionDecoder` | [`vfp_single_expert/action_decoder.py`](vfp_single_expert/action_decoder.py) | `experimental`, `decoder` |
| `vfp_single_info_embedder` | `InfoEmbedder` | [`vfp_single_expert/info_embedder.py`](vfp_single_expert/info_embedder.py) | `experimental`, `embedding` |
| `vfp_single_multimodal_bridge` | `MultiModalBridge` | [`vfp_single_expert/multimodal_bridge.py`](vfp_single_expert/multimodal_bridge.py) | `experimental`, `embedding` |
| `vfp_single_posterior` | `VAE_Posterior` | [`vfp_single_expert/posterior.py`](vfp_single_expert/posterior.py) | `experimental`, `posterior` |
| `vfp_single_prior` | `VAE_Prior` | [`vfp_single_expert/prior.py`](vfp_single_expert/prior.py) | `experimental`, `prior` |

### Naive flow matching policy

Simpler flow matching variant without variational inference or VQ-VAE.

| Registry key | Class | File | Tags |
|-------------|-------|------|------|
| `naive_hand_extractor` | `ResNet34Encoder` | [`naive_flow_matching_policy/hand_extractor.py`](naive_flow_matching_policy/hand_extractor.py) | `experimental`, `embedding` |
| `naive_info_embedder` | `InfoEmbedder` | [`naive_flow_matching_policy/info_embedder.py`](naive_flow_matching_policy/info_embedder.py) | `experimental`, `embedding` |
| `naive_action_decoder` | `ActionDecoder` | [`naive_flow_matching_policy/action_decoder.py`](naive_flow_matching_policy/action_decoder.py) | `experimental`, `decoder` |
| `naive_proprio_projector` | `ProprioProjector` | [`naive_flow_matching_policy/proprio_projector.py`](naive_flow_matching_policy/proprio_projector.py) | `experimental`, `projection` |

### Mutual information estimator

Encoder-decoder pairs for estimating mutual information between actions and states.

| Registry key | Class | File | Tags |
|-------------|-------|------|------|
| `a_decoder` | `ActionDecoder` | [`mutual_inf_est/action_decoder.py`](mutual_inf_est/action_decoder.py) | `experimental`, `action_decoder` |
| `a_encoder` | `ActionEncoder` | [`mutual_inf_est/action_encoder.py`](mutual_inf_est/action_encoder.py) | `experimental`, `action_encoder` |
| `state_decoder` | `ResNet34DecoderGroup` | [`mutual_inf_est/state_resnet34_decoder.py`](mutual_inf_est/state_resnet34_decoder.py) | `experimental`, `state_decoder` |
| `state_encoder` | `ResNet34EncoderGroup` | [`mutual_inf_est/state_resnet34_encoder.py`](mutual_inf_est/state_resnet34_encoder.py) | `experimental`, `state_encoder` |

### Third-party: OpenPI

Batched inference wrapper for ByteDance's OpenPI vision-language-action model.

| Registry key | Class | File | Tags |
|-------------|-------|------|------|
| `openpi_batched` | `OpenPiBatchedWrapper` | [`third_party/openpi/openpi_batched_wrapper.py`](third_party/openpi/openpi_batched_wrapper.py) | `experimental`, `openpi`, `vla` |

The `third_party/openpi/` directory contains a vendored copy of the OpenPI codebase with PyTorch model implementations, tokenizers, policies, and transforms. The `OpenPiBatchedWrapper` wraps `PI0Pytorch` for batched multi-camera inference with normalization and weight loading from safetensors checkpoints.

## Templates (abstract base classes)

[`templates/`](templates/) provides abstract base classes for policy component interfaces:

| Template | File | Key abstract method |
|----------|------|-------------------|
| `FlowMatchingBodyTemplate` | [`templates/flow_matching.py`](templates/flow_matching.py) | `forward(time, noise, memory_input, discrete_semantic_input, **kwargs)` |
| `MultiModalEncoderTemplate` | [`templates/multimodal_encoder.py`](templates/multimodal_encoder.py) | `forward(cond_proprio, cond_visual, cond_semantic, action, **kwargs)` |
| `InformationEncoderTemplate` | [`templates/information_encoder.py`](templates/information_encoder.py) | `forward(...)` — information encoding interface |

These templates define the expected forward signatures. Concrete implementations (action decoders, info embedders, etc.) inherit from them.

## Utilities

[`utils/`](utils/) provides shared helpers:

| Module | File | Purpose |
|--------|------|---------|
| `pos_encoding` | [`utils/pos_encoding.py`](utils/pos_encoding.py) | Positional encoding for transformer inputs |
| `time_embedding` | [`utils/time_embedding.py`](utils/time_embedding.py) | Sinusoidal time step embeddings for flow matching |

## How to use experimental blocks

All experimental blocks are available in YAML via their registry keys:

```yaml
schema_version: 1
model:
  graph:
    inputs: [cond_proprio, cond_visual, action, time, noise]
    modules:
      posterior:
        _type_: cfg_vqvae_posterior
        # ... constructor kwargs
      encoder:
        _type_: cfg_vqvae_info_encoder
        # ... constructor kwargs
      decoder:
        _type_: cfg_vqvae_action_decoder
        # ... constructor kwargs
    nodes:
      - {name: semantic, call: module:posterior, args: [$cond_proprio, $cond_visual, $action]}
      - {name: cond, call: module:encoder, args: [$cond_proprio, $cond_visual, $semantic]}
      - {name: out, call: module:decoder, args: [$time, $noise, $cond, $semantic]}
    outputs: [$out]
    return: single
```

See [`configs/experiments/`](../../../configs/experiments/) for complete example configs.

## Listing all registered experimental blocks

```python
from model_constructor.registry.default_registry import get_default_registry

reg = get_default_registry()
for name in reg.list_modules():
    entry = reg.get_module(name)
    if "experimental" in entry.tags:
        print(f"{name:40s} tags={entry.tags}")
```
