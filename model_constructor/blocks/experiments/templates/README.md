# `templates/` — Abstract base classes for policy components

This directory defines small `abc.ABC` + `torch.nn.Module` mixin classes that pin down the **expected `forward()` signature** for each kind of policy component. They're typed interfaces: a concrete block inheriting from `FlowMatchingBodyTemplate` is announcing "I accept `(time, noise, memory_input, discrete_semantic_input)` and produce a velocity tensor." That convention lets different research configurations swap concrete implementations in YAML without worrying about argument-name drift.

These templates are **not registered** — they're abstract; you can't instantiate `FlowMatchingBodyTemplate` directly. The concrete subclasses are what `register.py` registers.

Naming clarification: the term "template" overloads in this codebase. The other "template" — `_template_` in YAML — is unrelated. That one's about reusable spec snippets in YAML and lives in [`../../../config/templates.py`](../../../config/templates.py). This `templates/` directory is about Python abstract base classes.

---

## The three templates

### `FlowMatchingBodyTemplate` — [`flow_matching.py`](flow_matching.py)

Interface for the **body** of a flow-matching policy: given the current time `t ∈ [0, 1]`, a noise tensor, and conditioning information, produce the predicted velocity field that transforms noise into data.

```python
class FlowMatchingBodyTemplate(ABC, torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

    @abstractmethod
    def forward(
        self,
        time: float,                                  # scalar(s) in [0, 1]
        noise: torch.Tensor,                          # the noisy input
        memory_input: torch.Tensor,                   # cross-attention memory (encoder output)
        discrete_semantic_input: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        raise NotImplementedError
```

**Concrete implementations that inherit from it:**

- [`../cfg_vqvae_flow_matching/action_decoder.py`](../cfg_vqvae_flow_matching/action_decoder.py) `ActionDecoder` (registered as `cfg_vqvae_action_decoder`)
- [`../naive_flow_matching_policy/action_decoder.py`](../naive_flow_matching_policy/action_decoder.py) `ActionDecoder` (registered as `naive_action_decoder`)
- [`../variational_flow_matching_policy/experts.py`](../variational_flow_matching_policy/experts.py) `MoE` (registered as `vfp_moe`) — each expert is a flow-matching body
- [`../vfp_single_expert/action_decoder.py`](../vfp_single_expert/action_decoder.py) `ActionDecoder` (registered as `vfp_single_action_decoder`)

### `MultiModalEncoderTemplate` — [`multimodal_encoder.py`](multimodal_encoder.py)

Interface for modules that **fuse multimodal conditioning** (proprioception + vision + optional semantic + optional action) into a single representation.

```python
class MultiModalEncoderTemplate(ABC, nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

    @abstractmethod
    def forward(
        self,
        cond_proprio: torch.Tensor,                   # (B, seq, features)
        cond_visual: torch.Tensor,                    # multi-shape; see InformationEncoder
        cond_semantic: torch.Tensor | None = None,
        action: torch.Tensor | None = None,
        **kwargs,
    ):
        raise NotImplementedError
```

**Concrete implementations:**

- [`information_encoder.py`](information_encoder.py) `InformationEncoder` (uses the template's signature directly; not separately registered, but used internally by several info-encoder blocks)
- [`../cfg_vqvae_flow_matching/conditioning_info_encoder.py`](../cfg_vqvae_flow_matching/conditioning_info_encoder.py) `ConditioningInfoEncoder` (registered as `cfg_vqvae_info_encoder`)
- The various `*_info_embedder.py` modules across `naive_flow_matching_policy/`, `variational_flow_matching_policy/`, `vfp_single_expert/`

### `InformationEncoderTemplate` — [`information_encoder.py`](information_encoder.py)

A more general "information encoder" interface. The concrete `InformationEncoder` class in the same file is the actual implementation; its signature is the practical contract you'd code against.

Looking at the file: `InformationEncoder.forward(cond_proprio, cond_visual, cond_semantic=None, action=None)` returns `{'cls_token': ..., 'encoder_output': ...}`. Other info-encoder-style blocks across the policies typically expose similar named outputs.

---

## How to author a new flow-matching body

Concrete recipe — say you want to build a new flow-matching body called `MyFlowBody` that uses a Mamba-style state-space layer instead of a transformer. Steps:

**Step 1.** Inherit from `FlowMatchingBodyTemplate`:

```python
# my_project/my_flow_body.py
import torch
import torch.nn as nn
from model_constructor.blocks.experiments.templates.flow_matching import FlowMatchingBodyTemplate
from model_constructor.blocks.experiments.utils.time_embedding import get_time_embedding


class MyFlowBody(FlowMatchingBodyTemplate):
    def __init__(
        self,
        *,
        action_dim: int,
        d_model: int,
        num_layers: int,
        **kwargs,
    ):
        super().__init__(**kwargs)

        # Project (noise + time) into the model's hidden dim
        self.noise_proj = nn.Linear(action_dim, d_model)
        self.time_mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.SiLU(),
            nn.Linear(d_model * 2, d_model),
        )

        # ... your custom body layers here, e.g. a stack of Mamba / SSM blocks
        self.body = nn.Sequential(*[nn.Linear(d_model, d_model) for _ in range(num_layers)])

        self.output_proj = nn.Linear(d_model, action_dim)

    def forward(
        self,
        time: torch.Tensor,                # (B,)
        noise: torch.Tensor,               # (B, seq, action_dim)
        memory_input: torch.Tensor,        # (B, mem_seq, d_model)
        discrete_semantic_input: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        # Embed time into the same space as the body
        t_emb = get_time_embedding(time, embedding_dim=self.noise_proj.out_features)
        t_emb = self.time_mlp(t_emb)             # (B, d_model)

        x = self.noise_proj(noise)               # (B, seq, d_model)
        x = x + t_emb.unsqueeze(1)               # broadcast time along seq

        # (your body here — could attend to memory_input, optionally use discrete_semantic_input)
        x = self.body(x)

        return self.output_proj(x)               # (B, seq, action_dim)
```

Notes on conventions you'll see across the existing implementations:

- The constructor takes **keyword-only arguments** (with the leading `*`), so YAML kwargs map 1:1 to constructor parameters and typos are caught by the `strict` signature policy.
- `**kwargs` in `__init__` forwards to the template — the template's `__init__` just calls `super().__init__()` to wire up the `torch.nn.Module` machinery.
- The `time` argument is typed as `float` in the template signature, but every concrete implementation accepts a `torch.Tensor` of shape `(B,)` in practice (one time value per batch element). The template signature is loose; the concrete implementations are the practical contract.

**Step 2.** Register it from your parent repo:

```python
# my_project/__init__.py  (or a dedicated register module)
def register(registry):
    from .my_flow_body import MyFlowBody
    registry.register_module("my_project.my_flow_body", MyFlowBody,
                             signature_policy="strict",
                             tags=("custom", "flow_matching"))
```

**Step 3.** Use it in YAML in place of any existing `*_action_decoder`:

```yaml
schema_version: 1
settings:
  allowed_import_prefixes: ["model_constructor.", "my_project."]
imports:
  - my_project

model:
  graph:
    inputs: [time, noise, memory_input]
    modules:
      body:
        _type_: my_project.my_flow_body
        action_dim: 24
        d_model: 384
        num_layers: 6
    nodes:
      - {name: out, call: module:body,
         kwargs: {time: $time, noise: $noise, memory_input: $memory_input}}
    outputs: [$out]
    return: single
```

Why this is useful: every concrete `FlowMatchingBodyTemplate` implementation accepts the same `(time, noise, memory_input, discrete_semantic_input)` quartet. You can swap `MyFlowBody` in for `cfg_vqvae_action_decoder` (or any other) without rewiring the surrounding graph — only the `_type_` line and constructor kwargs change.

---

## Why bother with templates?

Two practical benefits:

- **Documentation by signature.** When you read `class MyEncoder(MultiModalEncoderTemplate):`, you know immediately that the class accepts `(cond_proprio, cond_visual, cond_semantic, action)`. You don't have to scroll up to find the `forward` signature.
- **Plug-and-play YAML configs.** Two YAML graphs that wire their info-encoder module the same way (`kwargs: {cond_proprio: $..., cond_visual: $..., ...}`) work with any `MultiModalEncoderTemplate` implementation. Drop-in swaps for A/B-testing different architectures.

The interface is enforced loosely (via `abstractmethod`), not strictly (no parameter type-check at registration). The repo's `signature_policy="strict"` on the concrete classes does the actual kwarg validation.
