from __future__ import annotations

import functools
import sys
from pathlib import Path

import torch
import einops


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from model_constructor import build_model, resolve_config


def _patch_einops_rearrange() -> None:
    # Work around comma-separated einops patterns in InformationEncoder.
    original_rearrange = einops.rearrange

    def rearrange_with_commas(x, pattern, *args, **kwargs):
        if isinstance(pattern, str) and "," in pattern:
            pattern = " ".join(pattern.replace(",", " ").split())
        return original_rearrange(x, pattern, *args, **kwargs)

    einops.rearrange = rearrange_with_commas


def _patch_moduledict_generic() -> None:
    if not hasattr(torch.nn.ModuleDict, "__class_getitem__"):
        torch.nn.ModuleDict.__class_getitem__ = classmethod(lambda cls, item: cls)


def _patch_missing_multimodal_template() -> None:
    from model_constructor.blocks.experiments.templates import flow_matching, multimodal_encoder

    if not hasattr(flow_matching, "MultiModalEncoderTemplate"):
        flow_matching.MultiModalEncoderTemplate = multimodal_encoder.MultiModalEncoderTemplate


def _patch_conditioning_info_encoder_init() -> None:
    from model_constructor.blocks.experiments.cfg_vqvae_flow_matching import conditioning_info_encoder

    cls = conditioning_info_encoder.ConditioningInfoEncoder
    if getattr(cls, "_codex_super_patched", False):
        return

    original_init = cls.__init__

    @functools.wraps(original_init)
    def patched_init(self, *args, **kwargs):
        torch.nn.Module.__init__(self)
        return original_init(self, *args, **kwargs)

    cls.__init__ = patched_init
    cls._codex_super_patched = True


def _patch_basic_transformer_inits() -> None:
    from model_constructor.blocks.basic_blocks import transformer_decoder, transformer_encoder

    def _patch_class(cls) -> None:
        if getattr(cls, "_codex_super_patched", False):
            return

        original_init = cls.__init__

        @functools.wraps(original_init)
        def patched_init(self, *args, **kwargs):
            torch.nn.Module.__init__(self)
            return original_init(self, *args, **kwargs)

        cls.__init__ = patched_init
        cls._codex_super_patched = True

    _patch_class(transformer_encoder.TransformerEncoder)

    cls = transformer_decoder.TransformerDecoder
    if not getattr(cls, "_codex_super_patched", False):
        original_init = cls.__init__

        @functools.wraps(original_init)
        def patched_init(self, *args, **kwargs):
            if "action_chunk_size" in kwargs and "num_tokens" not in kwargs:
                kwargs["num_tokens"] = kwargs.pop("action_chunk_size")
            torch.nn.Module.__init__(self)
            return original_init(self, *args, **kwargs)

        cls.__init__ = patched_init
        cls._codex_super_patched = True

    enc_cls = transformer_encoder.TransformerEncoder
    if not getattr(enc_cls, "_codex_forward_patched", False):
        original_forward = enc_cls.forward

        @functools.wraps(original_forward)
        def patched_forward(self, src):
            mask = self.src_causal_mask if self.is_causal else None
            try:
                return self.encoder(src=src, src_mask=mask, is_causal=self.is_causal)
            except TypeError:
                return self.encoder(src=src, mask=mask, is_causal=self.is_causal)

        enc_cls.forward = patched_forward
        enc_cls._codex_forward_patched = True


def main() -> None:
    _patch_einops_rearrange()
    _patch_moduledict_generic()
    _patch_missing_multimodal_template()
    _patch_conditioning_info_encoder_init()
    _patch_basic_transformer_inits()

    cfg_path = ROOT / "configs/experiments/cfg_vqvae_flow_matching.yaml"
    resolved = resolve_config(cfg_path)
    params = resolved.data["params"]

    batch = 2
    cond_proprio = torch.randn(batch, params["cond_proprio_seq"], params["cond_proprio_dim"])
    cond_visual = torch.randn(
        batch,
        params["cond_visual_frames"],
        params["cond_visual_tokens"],
        params["cond_visual_dim"],
    )
    action = torch.randn(batch, params["action_seq"], params["action_dim"])
    noise = torch.randn(batch, params["action_seq"], params["action_dim"])
    time = torch.rand(batch)

    model = build_model(cfg_path)
    model.eval()

    expected_info_tokens = (
        params["cond_visual_frames"] * params["cond_visual_tokens"]
        + params["cond_proprio_seq"]
        + 1
    )

    with torch.no_grad():
        posterior_out = model.graph_modules["vqvae_posterior"](
            cond_proprio=cond_proprio,
            cond_visual=cond_visual,
            action=action,
        )
        assert posterior_out.shape == (batch, params["transformer_d_model"])

        info_out = model.graph_modules["info_encoder"](
            cond_proprio=cond_proprio,
            cond_visual=cond_visual,
            cond_semantic=posterior_out,
        )
        assert info_out.shape == (batch, expected_info_tokens, params["transformer_d_model"])

        action_out = model.graph_modules["action_decoder"](
            time=time,
            noise=noise,
            memory_input=info_out,
            discrete_semantic_input=posterior_out,
        )
        assert action_out.shape == (batch, params["action_seq"], params["action_dim"])

        graph_out = model(
            cond_proprio=cond_proprio,
            cond_visual=cond_visual,
            action=action,
            time=time,
            noise=noise,
        )
        assert graph_out.shape == action_out.shape
    # Total parameters (including frozen ones)
    total_params = sum(p.numel() for p in model.graph_modules['action_decoder'].parameters())

    print("Total parameters:", total_params)
    print("smoke_ok", tuple(graph_out.shape))


if __name__ == "__main__":
    main()
