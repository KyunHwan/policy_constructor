# `backbones/vision/`

Vision feature extractors used by the experimental policies. Each backbone wraps a pretrained model and exposes a simple `forward(image)` interface that produces latent feature tensors.

Five backbones live here (counting the one VLM backbone in the sibling `vlm/` directory); three are **registered by default** and usable from YAML, two are present in code but **not registered**.

Registration source: [`../../register.py`](../../../register.py) (in `model_constructor/blocks/register.py`).

If you're new to the terms ResNet34, RadioV3, DA3 (Depth-Anything-3), DUNE, Cosmos-Reason2, see [docs/GLOSSARY.md](../../../../../docs/GLOSSARY.md).

---

## Backbone overview

### `ResNet34` ([`resnet34.py`](resnet34.py)) — **registered as `resnet34`**

Standard 34-layer residual CNN ([He et al., 2016](https://arxiv.org/abs/1512.03385)). Loads from `torchvision.models.resnet34(pretrained=True)`, drops the final pooling + classifier (`children()[:-2]`), and exposes the spatial feature map. Normalizes inputs with ImageNet stats, optionally resizes to 240×320.

```python
# Constructor
Resnet34(resize: bool = True)

# Input:  image — (B, 3, H, W) or (3, H, W); values in [0, 1] or [0, 255]
# Output: features — (B, 512, H_feat, W_feat)
```

Pretrained weights download on first use to `~/.cache/torch/hub/checkpoints/`.

### `RadioV3` ([`radiov3.py`](radiov3.py)) — **registered as `radiov3`**

[NVIDIA C-RADIO v3](https://github.com/NVlabs/RADIO) family of vision foundation models. Loads `c-radio_v3-l` from `torch.hub`. Produces a `(features, summary)` pair where `features` is an NCHW spatial map and `summary` is a per-image embedding.

```python
RadioV3(channels: tuple[int, int] = (1024, 3072),
        resize_method: str = 'auto')   # 'auto' = round to RADIO's nearest supported resolution
                                       # other = force 336 x 504

# Input:  image — (B, 3, H, W) or (3, H, W); RADIO requires H, W divisible by its min step
# Output: (features, summary)
#         features: (B, C_feat=1024, H_feat, W_feat) in NCHW format
#         summary:  (B, C_summary=3072)
```

`channels` is a hint about the model's output dims; it's stored in `self._channels` but the actual shapes come from the underlying RADIO model.

### `DepthAnything3Bridge` ([`depth_anything_3.py`](depth_anything_3.py)) — **registered as `da3`** — requires git submodule

[ByteDance Depth-Anything-3](https://github.com/ByteDance-Seed/Depth-Anything-3). A monocular depth/3D vision backbone. Loads `depth-anything/DA3-LARGE-1.1` via `DepthAnything3.from_pretrained(...)` and exposes intermediate transformer features.

```python
DepthAnything3Bridge(resize_method: str = 'forced')   # 'auto' rounds H, W to multiples of 14
                                                       # 'forced' resizes to 336 x 504

# Input:  image — (B, 3, H, W); values in [0, 1] or [0, 255]
# Output: latent_features — (B, sum_feat_dim, H/14, W/14) — concatenated along channel dim
#         of the layers selected by export_feat_layers (default [23], the last layer)
```

**This backbone depends on a git submodule.** See [DA3 submodule setup](#da3-submodule-setup) below before using.

### `DUNE` ([`dune.py`](dune.py)) — **NOT registered**

[Naver Labs DUNE](https://europe.naverlabs.com/research/publications/dune/). A multi-task-trained ViT-base/14 backbone at 448×448. Loaded via `torch.hub.load("naver/dune", "dune_vitbase_14_448_paper")`.

```python
DUNE()
# Input:  image — needs to be (..., 3, 448, 448); the wrapper does NOT auto-resize.
# Output: model's native output (see upstream).
```

**Not currently registered** in [`../../../register.py`](../../../register.py). To use it from YAML, register it yourself from a parent-repo module — see [How to register an unregistered backbone](#how-to-register-an-unregistered-backbone) below.

### `CosmosReason2Encoder` ([`../vlm/cosmos_reason2.py`](../vlm/cosmos_reason2.py)) — **NOT registered**

(Strictly speaking this lives in the sibling `vlm/` directory, not `vision/`; we document it here because it's the only other backbone in the experimental tree.)

NVIDIA [Cosmos-Reason2-2B](https://huggingface.co/nvidia/Cosmos-Reason2-2B) vision-language model. Loads via `transformers.Qwen3VLForConditionalGeneration.from_pretrained(...)`. Produces multimodal token features (combined text + vision sequence) suitable for cross-attention.

```python
CosmosReason2Encoder(model_name: str = "nvidia/Cosmos-Reason2-2B",
                     last_k_layers: int = 4,
                     dtype: torch.dtype = torch.bfloat16,
                     attn_implementation: str = "sdpa",
                     device_map: str = "auto")

# Input:  image (PIL/np/path), question (str), system_prompt (str)
# Output: (mem_tokens [B,S,D], mem_mask [B,S], img_mask [B,S], meta dict)
```

Heavy dependencies: `transformers`, the model itself downloads on first use.

**Not currently registered.**

---

## Registry status — at a glance

| Class | File | Registered? | Registry key |
|---|---|---|---|
| `Resnet34` | [`resnet34.py`](resnet34.py) | Yes | `resnet34` |
| `RadioV3` | [`radiov3.py`](radiov3.py) | Yes | `radiov3` |
| `DepthAnything3Bridge` | [`depth_anything_3.py`](depth_anything_3.py) | Yes (submodule required) | `da3` |
| `DUNE` | [`dune.py`](dune.py) | **No** | — |
| `CosmosReason2Encoder` | [`../vlm/cosmos_reason2.py`](../vlm/cosmos_reason2.py) | **No** | — |

Registration source: [`../../../register.py`](../../../register.py).

---

## DA3 submodule setup

`DepthAnything3Bridge` does `from depth_anything_3.api import DepthAnything3` ([`depth_anything_3.py`](depth_anything_3.py) line 4). The `depth_anything_3` package itself lives in this repo as a git submodule:

```
model_constructor/blocks/experiments/backbones/vision/externals/depth_anything_3/
```

mapped in [`/.gitmodules`](../../../../../.gitmodules):

```
[submodule "model_constructor/blocks/experiments/backbones/vision/externals/depth_anything_3"]
    path = model_constructor/blocks/experiments/backbones/vision/externals/depth_anything_3
    url = https://github.com/ByteDance-Seed/Depth-Anything-3
```

### Initialize the submodule

If you cloned this repo *with* `--recurse-submodules`, the submodule is already populated and you can skip ahead. Otherwise, from the repo root:

```bash
git submodule update --init --recursive
```

Verify the submodule is populated:

```bash
ls model_constructor/blocks/experiments/backbones/vision/externals/depth_anything_3
```

You should see entries such as `LICENSE`, `README.md`, `pyproject.toml`, `src/`, `assets/`, `docs/`. If the directory is empty, the submodule init didn't take — re-run `git submodule update --init --recursive` from the repo root.

### Make `depth_anything_3` importable

The DA3 wrapper does `import depth_anything_3`, so the **`src/` directory** of the submodule (which contains the `depth_anything_3` package) must be on `PYTHONPATH`:

```bash
# (from the repo root)
DA3_SRC="model_constructor/blocks/experiments/backbones/vision/externals/depth_anything_3/src"
export PYTHONPATH="$PWD/$DA3_SRC:$PYTHONPATH"
```

Or follow the upstream README's installation instructions inside the submodule directory (e.g., `pip install -e .` from the submodule root).

### License — read it

The submodule has its own license at [`externals/depth_anything_3/LICENSE`](externals/depth_anything_3/LICENSE) — we do not modify or redistribute its contents from this repo. Treat that directory as read-only and out of scope for changes here.

### Upstream

Bug reports, feature requests, and upstream issues should go to [github.com/ByteDance-Seed/Depth-Anything-3](https://github.com/ByteDance-Seed/Depth-Anything-3), not this repo.

---

## How to register an unregistered backbone

Suppose you want to use `DUNE` from YAML. You do not need to modify any file in `model_constructor/` — the parent-repo `imports:` mechanism works fine.

**Step 1.** Create a small registration module in your parent repo:

```python
# my_project/extra_backbones.py
from model_constructor.blocks.experiments.backbones.vision.dune import DUNE
# (Cosmos-Reason2 lives in vlm/, not vision/:)
# from model_constructor.blocks.experiments.backbones.vlm.cosmos_reason2 import CosmosReason2Encoder

def register(registry):
    registry.register_module("dune", DUNE, signature_policy="best_effort",
                             tags=("experimental", "backbone"))
    # registry.register_module("cosmos_reason2", CosmosReason2Encoder, signature_policy="best_effort",
    #                          tags=("experimental", "vlm"))
```

Note: `DUNE.__init__` takes no kwargs, and `CosmosReason2Encoder.__init__` accepts unusual types (`torch.dtype`) that `inspect` can introspect but you may not always want strict-validated. `best_effort` is a safe default for both.

**Step 2.** In your YAML, allow your prefix and import the module:

```yaml
schema_version: 1
settings:
  allowed_import_prefixes: ["model_constructor.", "my_project."]
imports:
  - my_project.extra_backbones

model:
  graph:
    inputs: [image]
    modules:
      dune: {_type_: dune}
    nodes:
      - {name: feats, call: module:dune, args: [$image]}
    outputs: [$feats]
    return: single
```

**Step 3.** Build and run. Make sure `my_project` is on `PYTHONPATH`.

For the full extension walkthrough, see [docs/QUICKSTART.md Variant 2](../../../../../docs/QUICKSTART.md#variant-2--add-a-custom-block-register-it-use-it-from-yaml).

---

## Where these backbones are used

Inside this repo, the registered backbones are used by:

- **CFG-VQVAE / VFP / naive flow matching policies** — the `*_hand_extractor` modules in each policy directory wrap `ResNet34Encoder` (a thin recreation of the `resnet34` block) for the "hand camera" feature stream. The main visual conditioning is expected to be a pre-encoded tensor, often produced upstream by `radiov3` or `da3` in the parent training repo.
- **DSRL** — `dsrl_img_encoder` uses three independent ResNet34 instances (one per camera: head/left/right) defined in [`../../dsrl/tri_img_embedder.py`](../../dsrl/tri_img_embedder.py).
- **ResFit** — same three-camera ResNet34 pattern, internal to the `resfit_residual_actor` and `resfit_q_function` modules.

Typical wiring in a parent repo: run images through a vision backbone once per batch, pass the resulting features to the policy as `cond_visual` (or equivalent). The policy itself doesn't usually contain the raw-image-to-features step in its graph — see the input shapes in [`configs/experiments/cfg_vqvae_flow_matching.yaml`](../../../../../configs/experiments/cfg_vqvae_flow_matching.yaml), which already takes `cond_visual` as a `(B, frames, tokens, dim)` feature tensor.
