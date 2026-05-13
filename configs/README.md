# `configs/`

YAML configurations used to exercise the constructor. This directory has two sub-folders:

- **[`examples/`](examples/)** — small, didactic YAMLs. Each one is meant to be readable end-to-end in under a minute and to demonstrate exactly one feature.
- **[`experiments/`](experiments/)** — full research configs for the experimental policies (CFG-VQVAE, etc.). These reference experimental blocks defined in [`model_constructor/blocks/experiments/`](../model_constructor/blocks/experiments/), and may require additional optional dependencies and non-trivial input shapes.

If you're new to writing these YAMLs:

- **Practical authoring guide** — [`../model_constructor/config/authoring_yaml.md`](../model_constructor/config/authoring_yaml.md). Read this first. It's the cookbook.
- **Normative contract** — [`../model_constructor/config/schema_v1.md`](../model_constructor/config/schema_v1.md). Consult this when you need to know exactly what the parser will accept.

Rule of thumb: write a YAML by copying from `authoring_yaml.md`; when something the parser rejects looks like it should be valid, check `schema_v1.md`.

---

## `examples/` — small didactic configs

These are intended to be **runnable as-is** via `build_model("configs/examples/<name>.yaml")`. Each one isolates one frontend / one feature.

### [`sequential_mlp.yaml`](examples/sequential_mlp.yaml)

A one-layer `model.sequential` config:

```yaml
schema_version: 1
model:
  sequential:
    inputs: [x]
    layers:
      - _type_: mlp
        dims: [null, 64, 16]
        dropout: 0.1
```

Demonstrates: the `sequential` frontend, the `_type_:` key, the `null`-as-first-`dims` lazy-MLP pattern.

**Status:** as of this writing, this YAML references `_type_: mlp`, but the `mlp` key is **not** registered in [`../model_constructor/blocks/register.py`](../model_constructor/blocks/register.py). Building this YAML today raises `ConfigError: Unknown module type 'mlp'`. See [TROUBLESHOOTING.md — Unknown module type 'mlp'](../docs/TROUBLESHOOTING.md#unknown-module-type-mlp--conv_bn_act--residual_block) for two workarounds.

### [`graph_skip_add.yaml`](examples/graph_skip_add.yaml)

A small `model.graph` config with a skip connection via `op:add`:

```yaml
schema_version: 1
model:
  graph:
    inputs: [x]
    modules:
      stem:
        _type_: conv_bn_act
        in_channels: null
        out_channels: 16
        kernel_size: 3
        padding: 1
      main:
        _type_: conv_bn_act
        in_channels: 16
        out_channels: 16
        kernel_size: 3
        padding: 1
        act: {_type_: nn.Identity}     # override the default ReLU activation
    nodes:
      h1: {call: module:stem, args: [$x]}        # stem(x)
      h2: {call: module:main, args: [$h1]}       # main(stem(x))
      h3: {call: op:add, args: [$h1, $h2]}       # stem(x) + main(stem(x))  ← skip
    order: [h1, h2, h3]                          # nodes mapping form requires `order`
    outputs: [$h3]
    return: single
```

Demonstrates: the `graph` frontend, named modules in `modules:`, a skip connection via `op:add`, the mapping form of `nodes:` (requires `order:`), nested specs (the `act:` kwarg of `conv_bn_act` is itself a spec).

**Status:** same as above — references `_type_: conv_bn_act`, which is not currently registered.

For a fully-working DAG example you can run today, use the inline-dict variants in [QUICKSTART.md — Variant 1](../docs/QUICKSTART.md#variant-1--build-and-run-an-existing-model). They use only `nn.*` types (which are always registered).

---

## `experiments/` — full research configs

These are full configs for the experimental policy blocks. They require:

- the corresponding experimental blocks (registered automatically by [`blocks/register.py`](../model_constructor/blocks/register.py));
- often additional optional dependencies (e.g., `einops`, `safetensors`, `jax`, pretrained weight downloads);
- non-trivial input shapes — the smoke-test script under [`../examples/smoke_cfg_vqvae_flow_matching.py`](../examples/smoke_cfg_vqvae_flow_matching.py) documents the exact tensor shapes the model expects.

### [`cfg_vqvae_flow_matching.yaml`](experiments/cfg_vqvae_flow_matching.yaml)

A complete classifier-free-guidance, VQ-VAE-quantized, flow-matching action policy. The architecture is: a **posterior** that produces a discrete latent semantic vector from (proprio, vision, action), an **info encoder** that fuses proprio + vision + the semantic vector into a transformer memory, and an **action decoder** (a flow-matching body) that maps (time, noise) onto a predicted action chunk, conditioned on the memory and the semantic vector.

Inputs and expected shapes (from [`smoke_cfg_vqvae_flow_matching.py`](../examples/smoke_cfg_vqvae_flow_matching.py)):

| Input | Shape | Source of the constants |
|---|---|---|
| `cond_proprio` | `(batch, cond_proprio_seq=40, cond_proprio_dim=62)` | `params.cond_proprio_seq`, `params.cond_proprio_dim` |
| `cond_visual` | `(batch, cond_visual_frames=2, cond_visual_tokens=3, cond_visual_dim=1072)` | `params.cond_visual_frames`, `params.cond_visual_tokens`, `params.cond_visual_dim` |
| `action` | `(batch, action_seq=40, action_dim=24)` | `params.action_seq`, `params.action_dim` |
| `noise` | `(batch, action_seq=40, action_dim=24)` | same as `action` — noise has the same shape as the prediction target |
| `time` | `(batch,)` — scalars in [0, 1] | flow-matching convention |

Output: `(batch, action_seq=40, action_dim=24)` — the predicted action chunk.

#### Line-by-line walkthrough

Top of file:

```yaml
schema_version: 1
```

The required schema version marker.

```yaml
params:
  cond_proprio_dim: 62
  cond_proprio_seq: 40
  cond_visual_dim: 1072
  cond_visual_frames: 2
  cond_visual_tokens: 3
  action_dim: 24
  action_seq: 40
```

The "geometry" of the inputs. `params.*` values are not consumed by the schema — they exist purely to be referenced via `${params.X}` interpolation later. Centralizing them at the top means a single edit propagates everywhere.

```yaml
  transformer_d_model: 384
  transformer_nhead: 16
  transformer_dim_feedforward: 2048
  transformer_dropout: 0.2
  transformer_activation: gelu
  transformer_batch_first: true
  transformer_is_causal: false
  transformer_tgt_is_causal: true
  transformer_num_layers: 12
```

Shared transformer hyperparameters reused across the three modules below. `transformer_is_causal: false` because the encoders operate on the full conditioning sequence at once; `transformer_tgt_is_causal: true` because the action decoder predicts action tokens auto-regressively in the target self-attention.

```yaml
  posterior_num_tokens: 300
  info_num_tokens: 300
  action_chunk_size: 40
```

Sequence-length budgets for each transformer (used to pre-compute causal mask buffers).

```yaml
model:
  graph:
    inputs: [cond_proprio, cond_visual, action, time, noise]
```

Five named inputs. Caller supplies all five (positional or by keyword).

```yaml
    modules:
      vqvae_posterior:
        _type_: cfg_vqvae_posterior
        cond_proprio_dim: ${params.cond_proprio_dim}
        cond_visual_dim: ${params.cond_visual_dim}
        ...
        transformer_num_tokens: ${params.posterior_num_tokens}
        action_dim: ${params.action_dim}
        use_cond_semantic: false               # posterior consumes action; no semantic input
        use_cond_semantic_projection: false
        cond_semantic_dim: null
```

The posterior module: `VQVAE_Posterior` from [`cfg_vqvae_flow_matching/vq_vae_multimodal_posterior.py`](../model_constructor/blocks/experiments/cfg_vqvae_flow_matching/vq_vae_multimodal_posterior.py). All hyperparameters are filled from `${params.*}` interpolations — this is the point of centralizing them. Type-preservation: full-scalar `${params.X}` references preserve the original type (int stays int, bool stays bool).

```yaml
      info_encoder:
        _type_: cfg_vqvae_info_encoder
        cond_proprio_dim: ${params.cond_proprio_dim}
        cond_visual_dim: ${params.cond_visual_dim}
        ...
        use_cond_semantic: true                # info encoder consumes the semantic latent
        cond_semantic_dim: ${params.transformer_d_model}
```

The info encoder: `ConditioningInfoEncoder`. It takes proprio + vision + the semantic vector and produces a transformer memory.

```yaml
      action_decoder:
        _type_: cfg_vqvae_action_decoder
        ...
        transformer_tgt_is_causal: ${params.transformer_tgt_is_causal}
        transformer_action_chunk_size: ${params.action_chunk_size}
        use_cond_semantic: true
        cond_semantic_dim: ${params.transformer_d_model}
```

The action decoder: `ActionDecoder`, a flow-matching body. Cross-attends to the info-encoder memory; takes the semantic vector as additional conditioning.

```yaml
    nodes:
      - name: posterior_semantic
        call: module:vqvae_posterior
        kwargs:
          cond_proprio: $cond_proprio
          cond_visual: $cond_visual
          action: $action
```

Step 1: produce the discrete semantic latent from (proprio, vision, action). Note `args: []` is implicit (omitted) — all arguments are passed as `kwargs:` matching the module's `forward()` signature.

```yaml
      - name: conditioning_info
        call: module:info_encoder
        kwargs:
          cond_proprio: $cond_proprio
          cond_visual: $cond_visual
          cond_semantic: $posterior_semantic         # reuse step 1's output
```

Step 2: produce the transformer memory. The `$posterior_semantic` reference resolves at runtime from the context dict written by step 1.

```yaml
      - name: decoded_action
        call: module:action_decoder
        kwargs:
          time: $time
          noise: $noise
          memory_input: $conditioning_info
          discrete_semantic_input: $posterior_semantic
```

Step 3: predict the flow-matching velocity field at this `time` from `noise`, conditioned on the memory and the semantic latent.

```yaml
    outputs: [$decoded_action]
    return: single
```

Return policy: one output named `decoded_action`, returned as a bare tensor (not a tuple or dict).

#### How to actually run this config

The smoke script [`../examples/smoke_cfg_vqvae_flow_matching.py`](../examples/smoke_cfg_vqvae_flow_matching.py) creates random tensors of the shapes shown in the table above, calls `build_model(cfg_path)`, and runs both the individual sub-modules and the full graph. Read it before writing your own driver.

Note: the smoke script also monkey-patches a handful of small issues at the top of `main()` — these patches don't change the architecture but work around things like a comma-separated `einops` pattern and a missing `super().__init__()` call in one module. Treat those patches as a known maintenance debt rather than an example to imitate.

---

## How to add a new config file

1. Decide which sub-folder fits — `examples/` for didactic, `experiments/` for research.
2. Start with the minimal skeleton in [`authoring_yaml.md`](../model_constructor/config/authoring_yaml.md#minimal-config).
3. Centralize hyperparameters under `params:` and reference them via `${params.X}` so a future user can do experiments by changing only one section.
4. If your config will be a *base* for derived experiments, write the base under `configs/<category>/base.yaml` and the derived experiments under `configs/<category>/exp_*.yaml` with `defaults: [./base.yaml]`. See [`schema_v1.md`](../model_constructor/config/schema_v1.md) for the `defaults` rules.
5. Add a one-line description in this README under the matching section so the next reader can find it.
