# Glossary

Every codebase-specific term and every research term you'll encounter in the experimental blocks. One or two sentences each, plus a "see also" pointer.

Entries are grouped: **codebase terms**, **PyTorch terms used here**, and **research / policy terms**.

If a term has a specific technical meaning in this repo that differs from its general ML meaning, that's flagged.

---

## Codebase terms

### Registry

The lookup table from string keys to Python callables. `Registry` ([`registry/registry.py`](../model_constructor/registry/registry.py) line 23) holds two stores: **modules** (factories that return `torch.nn.Module` instances) and **ops** (pure runtime callables). YAML `_type_: foo` is resolved to `registry.get_module("foo").target`; `call: op:add` resolves to `registry.get_op("add").target`.

See also: [registry/README.md](../model_constructor/registry/README.md), [registry entry](#registry-entry), [signature policy](#signature-policy).

### Registry entry

The bundled metadata for a single registration: name, kind (`module` or `op`), target callable, signature policy, optional tags, optional doc. Defined as `RegistryEntry` ([`registry/registry.py`](../model_constructor/registry/registry.py) line 13).

See also: [registry](#registry).

### Spec

Any YAML dict containing `_type_` (or the gated `_target_`). The instantiator treats it as a request to build a Python object: it resolves the target, validates kwargs, recursively instantiates nested specs in the kwargs, and calls the target.

Example spec:

```yaml
{_type_: nn.Conv2d, in_channels: 3, out_channels: 16, kernel_size: 3}
```

The detection rule is in [`instantiate/instantiate.py`](../model_constructor/instantiate/instantiate.py) `_is_spec()` line 85.

See also: [`_type_`](#_type_), [`_target_`](#_target_), [factory](#factory).

### Factory

A callable that, when called, returns a `torch.nn.Module` instance. Most of the time the "factory" is just a class (`nn.Linear`, `MyBlock`), but it can be any callable — useful for wrappers. Stored in `RegistryEntry.target`.

See also: [registry entry](#registry-entry).

### `_type_`

The recommended way to point a spec at a registered module factory. Value must be a string registry key.

```yaml
_type_: nn.Linear     # registry key
in_features: 64
out_features: 32
```

See also: [`_target_`](#_target_), [spec](#spec).

### `_target_`

Expert-mode alternative to `_type_`: imports a dotted Python path at build time. Disabled by default; requires `settings.allow_target: true` and `settings.allowed_import_prefixes` to include the module prefix. Prefer `_type_`.

```yaml
_target_: my_project.path.MyClass
```

See also: [`_type_`](#_type_), [allowed_import_prefixes](#allowed_import_prefixes).

### `_args_` / `_kwargs_`

Optional spec keys. `_args_` (a list) supplies positional arguments; `_kwargs_` (a dict) supplies extra keyword arguments. Most specs use **inline kwargs** (any key not starting with `_`), and don't need either. Use these only for edge cases where the constructor requires positional args or where a kwarg key would collide with a reserved name.

```yaml
_type_: nn.Linear
_args_: [128, 256]       # rare — positional in_features, out_features
_kwargs_: {bias: false}  # rare — overflow kwargs
```

See also: [spec](#spec), [`instantiate/README.md`](../model_constructor/instantiate/README.md).

### Signature policy

A per-registry-entry rule controlling how strictly the instantiator validates the kwargs you pass against the constructor's signature. One of `strict`, `best_effort`, `runtime_only` (see [`instantiate/signature.py`](../model_constructor/instantiate/signature.py)):

- **strict** — introspect with `inspect.signature`; reject unknown kwargs; if introspection fails, **raise**.
- **best_effort** — introspect if possible, otherwise pass everything through. Default for built-in `nn.*` modules.
- **runtime_only** — never pre-validate; let the underlying call raise. Default for ops.

See also: [Mental Model — Signature policies](MENTAL_MODEL.md#signature-policies--strict-best_effort-runtime_only).

### Construct time vs runtime

- **Construct time** — the period from `build_model()` start to the moment a `GraphModel` is returned. Everything happens once: YAML resolve, IR compile, module instantiation.
- **Runtime** — every subsequent `model.forward(x)` call. This is when `$name` references are resolved against the runtime context dict and ops are invoked.

See also: [`api.py`](../model_constructor/api.py), [runtime reference](#runtime-reference).

### Op (vs module)

In this repo, an **op** is a pure runtime function — no `nn.Parameter`s, not in the module tree, called every forward pass. A **module** is a `torch.nn.Module` that holds parameters and is registered as `self.graph_modules[<name>]`. Ops are called via `call: op:<name>`; modules via `call: module:<name>`. See [Mental Model — Modules vs ops](MENTAL_MODEL.md#modules-vs-ops).

Built-in ops: `add`, `mul`, `cat`, `stack`, `getitem`, `identity` (registered in [`registry/builtins.py`](../model_constructor/registry/builtins.py)).

### `GraphIR`

The compiled, frozen-dataclass intermediate representation of a model. Holds the input names, the module **specs** (not yet instantiated), the ordered list of `StepIR`s, the outputs spec, and the return policy. Defined in [`graph/ir.py`](../model_constructor/graph/ir.py) line 32.

See also: [`graph/README.md`](../model_constructor/graph/README.md), [`GraphModel`](#graphmodel).

### `GraphModel`

The `torch.nn.Module` subclass that **executes** a `GraphIR`. Holds the instantiated modules in a `ModuleDict`, walks the steps in order on each `.forward()` call, resolves `$name` references against a per-call context dict, and packs outputs per the return policy. Defined in [`graph/model.py`](../model_constructor/graph/model.py) line 12.

See also: [`GraphIR`](#graphir), [return policy](#return-policy).

### `ResolvedConfig`

The output of `resolve_config()`: a frozen dataclass with the fully-resolved (defaults-merged, templates-expanded, interpolated, schema-validated) YAML data, a `SourceMap`, the parsed `Settings`, and the root file path (if any). Defined in [`config/resolve.py`](../model_constructor/config/resolve.py) line 18.

See also: [`config/README.md`](../model_constructor/config/README.md), [Source map](#source-map).

### Source map

A mapping from config paths (tuples like `("model", "graph", "modules", "stem")`) to source locations (file, line, col). Used to print "the error is at line 42 of `your.yaml`" instead of "the error is somewhere." Built by [`config/yaml_loader.py`](../model_constructor/config/yaml_loader.py) and threaded through every transform.

See also: [`config/source_map.py`](../model_constructor/config/source_map.py).

### `defaults` composition

A top-level YAML key listing parent YAML files (paths relative to the current file). The resolver loads each parent file, merges them in order, then overlays the current file. Cycles are detected. Only supported when `build_model()` is given a file path (not an in-memory dict).

```yaml
defaults: [../base/backbone.yaml]
schema_version: 1
params: {width: 64}
```

Implemented in [`config/resolve.py`](../model_constructor/config/resolve.py) `_resolve_from_file()` line 50.

See also: [`authoring_yaml.md`](../model_constructor/config/authoring_yaml.md).

### Merge directive (`_merge_`)

YAML lists **replace** by default during composition. To control list merging, wrap the list in a merge container:

```yaml
imports:
  _merge_: append          # or prepend, replace, keyed
  _value_: [my_project.blocks]
```

Modes:

- `replace` (default if you don't use the directive)
- `append` — base list + override list
- `prepend` — override list + base list
- `keyed` — like a dict merge: each list item is identified by a `key:` field; items with the same key are deep-merged

Implemented in [`config/merge.py`](../model_constructor/config/merge.py).

### Template (`_template_`)

A reusable spec defined under top-level `templates:`, referenced via `_template_: <name>` in any dict node. The named template is deep-merged into the referencing node (node wins). Templates are expanded **before** interpolation.

```yaml
templates:
  conv3x3: {_type_: conv_bn_act, kernel_size: 3, padding: 1}

model:
  graph:
    modules:
      stem:
        _template_: conv3x3      # expands the template
        in_channels: null
        out_channels: 16
```

Implemented in [`config/templates.py`](../model_constructor/config/templates.py).

### Interpolation (`${...}`)

Reference values inside the resolved config or environment variables:

- `${params.width}` — config-path reference. **Type-preserving** when the entire scalar value equals the expression; **stringified** when embedded in a larger string.
- `${env:VAR}` — raw env string, no trimming.
- `${env:int:VAR}`, `${env:float:VAR}`, `${env:bool:VAR}`, `${env:json:VAR}` — typed env vars; whitespace-trimmed before casting.
- All of the above support a default: `${env:int:WIDTH,32}` → 32 if `WIDTH` unset.

Cycles in `${...}` references are detected. Implemented in [`config/interpolate.py`](../model_constructor/config/interpolate.py).

### Plugin (`imports:`)

A top-level YAML key listing Python module names. The constructor imports each module and, if the module defines a top-level `register(registry)` function, calls it with the active registry. This is how parent repos add custom blocks without forking. Gated by `settings.allowed_import_prefixes` (default `["model_constructor."]`).

See also: [`util/README.md`](../model_constructor/util/README.md), [Mental Model — extension points](MENTAL_MODEL.md#the-two-extension-points).

### `allowed_import_prefixes`

A `Settings` field (default `("model_constructor.",)`) that gates both `imports:` and `_target_` import paths. Any module name that doesn't start with one of the listed prefixes is rejected. Add your parent-repo prefix here to allow `imports:` from outside `model_constructor`.

### DAG

Directed acyclic graph. `GraphIR` is DAG-only: every `$name` reference must point to a model input or to a value produced by an **earlier** node. Cycles and forward references are caught at compile time in [`graph/compiler.py`](../model_constructor/graph/compiler.py) `_validate_ir()`.

See also: [forward reference](#forward-reference), [Mental Model — DAG-only](MENTAL_MODEL.md#dag-only--why-forward-references-are-forbidden).

### Forward reference

A `$name` reference that points to a value produced by a **later** node. Forbidden — surfaced as `ConfigError: Forward reference(s) not allowed: [name]`. Fix: reorder nodes so producers come first, or move the cyclic part inside a single block module.

### Runtime reference

A YAML value of the form `$name` (note: **not** `${name}`, which is interpolation). It tells the executor at forward time to look up `name` in the per-call context dict and substitute that value. Used inside node `args:` and `kwargs:`:

```yaml
args: [$x, $h1]
kwargs: {memory_input: $info_out}
```

Escape: `$$literal` produces the literal string `$literal`. Implemented in [`graph/compiler.py`](../model_constructor/graph/compiler.py) `_parse_runtime_value()` line 264 (compile time) and [`graph/model.py`](../model_constructor/graph/model.py) `_resolve_runtime_value()` line 91 (runtime).

### Return policy

How `GraphModel.forward()` packs the outputs. One of `single`, `tuple`, `dict`:

- `single` — exactly one output name; returns the bare tensor.
- `tuple` — multiple output names in a list; returns a Python tuple.
- `dict` — outputs as a `{key: name}` mapping; returns a dict.

If `return:` is omitted, it's inferred: mapping → `dict`, single-item list → `single`, otherwise `tuple`. Logic in [`graph/compiler.py`](../model_constructor/graph/compiler.py) `_coerce_return_policy()` line 413.

### Schema v1

The current normative YAML contract: required `schema_version: 1` and `model:` keys; allowed top-level keys `settings`, `defaults`, `imports`, `templates`, `params`; supported merge directives and interpolation forms. Full text in [`config/schema_v1.md`](../model_constructor/config/schema_v1.md).

### Lazy module

A `torch.nn` module whose parameter shapes are deferred to the first forward pass. In YAML, signal it with `null`:

```yaml
{_type_: nn.LazyLinear, out_features: 32}        # in_features inferred
{_type_: nn.LazyConv2d, out_channels: 16, kernel_size: 3, padding: 1}  # in_channels inferred
```

Gotcha: after the first call, the module is no longer lazy — feeding it a different input shape later raises a shape error. See [Mental Model — Lazy modules](MENTAL_MODEL.md#lazy-modules-and-the-null-trick).

### IR

Intermediate representation. In this repo, "IR" always means [`GraphIR`](#graphir) — the compiled, frozen-dataclass form of the model that sits between the resolved YAML and the runnable `GraphModel`.

---

## PyTorch terms used heavily here

### `nn.Module`

The base class for all PyTorch neural-network components. Holds parameters, defines `forward()`, integrates with autograd. The `GraphModel` returned by `build_model()` is one of these — call it like `model(x)`.

### `nn.ModuleDict`

A container mapping string names to `nn.Module`s, registering them as submodules so their parameters show up in `model.parameters()`. `GraphModel` stores its instantiated modules in `self.graph_modules = nn.ModuleDict(modules)`. Reference: [`graph/model.py`](../model_constructor/graph/model.py) line 25.

### `nn.LazyLinear` / `nn.LazyConv2d`

Deferred-shape variants of `nn.Linear` and `nn.Conv2d`. Skip the `in_features` / `in_channels` argument; the layer materializes its weight on the first forward pass. See [lazy module](#lazy-module).

### Parameter

A `torch.Tensor` wrapped in `nn.Parameter`, automatically tracked by `nn.Module` and registered for optimization. Anything in a module's `self.parameters()` iteration. Modules have them; ops don't.

### Buffer

A `torch.Tensor` registered with `self.register_buffer(name, value)`. Tracked by `nn.Module` (moves with `.to(device)`, included in `state_dict`) but **not** trained. Used for things like positional encoding tables or causal masks (see [`basic_blocks/transformer_encoder.py`](../model_constructor/blocks/basic_blocks/transformer_encoder.py) line 119).

### `.forward()`

The method `nn.Module` subclasses implement. Don't call it directly — call `module(x)` instead, which invokes hooks and `forward()` together.

### `.eval()`

Puts a module into evaluation mode: disables dropout, switches batch-norm to use running stats. Call this before inference. Has no effect on plain `nn.Linear`.

### `torch.no_grad()`

A context manager that disables gradient tracking inside the block. Wrap your inference calls in `with torch.no_grad():` to save memory.

---

## Research / policy terms (used by the experimental blocks)

These are the terms you'll see while reading the experimental policy code. Each entry includes a one-paragraph plain-English description; this is not a substitute for the original papers, but enough to follow the code.

### Flow matching

A continuous-time generative-modeling technique that learns a velocity field transforming noise into data along an interpolated trajectory. Compared to diffusion: typically requires fewer denoising steps and uses a simpler training objective (regress velocity at random times). In this repo, "flow matching" specifically refers to action-prediction policies that take `time` ∈ [0, 1] and `noise` ∈ ℝ^action_dim as inputs and predict the velocity field.

See also: [`experiments/templates/flow_matching.py`](../model_constructor/blocks/experiments/templates/flow_matching.py), [naive flow matching policy](#naive-flow-matching-policy), [VFP](#vfp-variational-flow-matching-policy).

### VQ-VAE

Vector-quantized variational autoencoder (van den Oord et al., 2017). An encoder maps inputs to continuous embeddings, which are then snapped to the nearest entry in a learned **codebook** (a discrete vocabulary of D-dimensional vectors). The decoder reconstructs from the discrete code. Used in this repo to discretize the policy's latent semantic representation.

See also: [codebook](#codebook), [`experiments/cfg_vqvae_flow_matching/vq_vae_codebook_manager.py`](../model_constructor/blocks/experiments/cfg_vqvae_flow_matching/vq_vae_codebook_manager.py).

### Codebook

The set of learned discrete embedding vectors used by a VQ-VAE. In the CFG-VQVAE policy here, the codebook lives in `VQCodebookManager` and is shared between the prior and posterior.

### CFG (classifier-free guidance)

A sampling technique from diffusion-model literature (Ho & Salimans, 2022) that improves controllability by interpolating between conditional and unconditional model predictions at inference time, using a guidance scale. In this repo's CFG-VQVAE policy, "CFG" denotes the architectural choice to support a null condition during training so guidance can be applied at sample time. The codebase here builds the **architecture**; the actual guidance arithmetic happens in the parent training/inference repo.

See also: [`experiments/cfg_vqvae_flow_matching/`](../model_constructor/blocks/experiments/cfg_vqvae_flow_matching/).

### VFP (variational flow matching policy)

A flow-matching policy variant that uses **variational inference** (separate prior and posterior networks) to learn a structured latent space, plus a **mixture-of-experts** body for the velocity field. In this repo: [`experiments/variational_flow_matching_policy/`](../model_constructor/blocks/experiments/variational_flow_matching_policy/).

See also: [MoE](#moe-mixture-of-experts), [prior](#prior), [posterior](#posterior).

### Naive flow matching policy

The non-variational, non-VQ flow-matching baseline used as a control in research. Same input signature (`time`, `noise`, conditioning) but with a single-expert action decoder, no codebook, no posterior. In this repo: [`experiments/naive_flow_matching_policy/`](../model_constructor/blocks/experiments/naive_flow_matching_policy/).

### Prior

The "prior" network in a variational architecture. Given only the conditioning inputs (e.g., proprioception + vision features) — not the ground-truth action — it predicts the parameters of the latent distribution. At inference time, the prior is what produces the latent; the posterior is used only during training.

See also: [posterior](#posterior), [VFP](#vfp-variational-flow-matching-policy).

### Posterior

The "posterior" network in a variational architecture. Given the conditioning **and** the ground-truth action, it predicts a tighter latent distribution (used to train against the prior via a KL term). At inference time, the posterior is not used. In the CFG-VQVAE policy here, the posterior also drives discrete latent assignment via the VQ codebook.

### Proprio / proprioception

In robot policy learning, **proprio**ception is the robot's own joint-state input: joint angles, velocities, gripper position, end-effector pose. As a tensor, it's typically `(batch, time_history, proprio_dim)`. The CFG-VQVAE config in [`configs/experiments/cfg_vqvae_flow_matching.yaml`](../configs/experiments/cfg_vqvae_flow_matching.yaml) uses `cond_proprio_dim: 62`, `cond_proprio_seq: 40`.

### Action chunk

A sequence of consecutive actions predicted in a single forward pass, instead of one action at a time. Common in modern robot policies (e.g., RT-2, OpenPI). In this repo, configurable via `action_chunk_size` parameters. The CFG-VQVAE config uses `action_chunk_size: 40`.

### Time embedding

For flow matching, the policy takes a scalar `time` ∈ [0, 1] indicating how far along the noise→data trajectory we are. The time-embedding module ([`experiments/utils/time_embedding.py`](../model_constructor/blocks/experiments/utils/time_embedding.py)) maps that scalar to a high-dimensional vector via sinusoidal features so the transformer body can condition on it.

### MoE (mixture-of-experts)

A neural network architecture where multiple **expert** sub-networks are combined by a **gate network** that decides (per input) which experts to use. Used here by the VFP policy ([`experiments/variational_flow_matching_policy/experts.py`](../model_constructor/blocks/experiments/variational_flow_matching_policy/experts.py)) — different experts can specialize in different sub-skills.

### Gate network

In a mixture-of-experts model, the small network that produces per-expert routing weights from the input. In VFP: [`experiments/variational_flow_matching_policy/gate.py`](../model_constructor/blocks/experiments/variational_flow_matching_policy/gate.py).

### Mutual information estimator

A network that learns to estimate mutual information between two variables (e.g., between actions and visual states), often via a contrastive or variational lower bound. Used as an auxiliary objective in policy training. In this repo: [`experiments/mutual_inf_est/`](../model_constructor/blocks/experiments/mutual_inf_est/) — encoder/decoder pairs for actions and ResNet34-encoded states.

### DSRL

A reinforcement-learning-flavored policy family in this repo, with `dsrl_q_function` (value network), `dsrl_noise_latent_actor` (a flow-matching-style actor over a latent action space), and processor blocks for ingesting image + depth + proprioception. See [`experiments/dsrl/`](../model_constructor/blocks/experiments/dsrl/). The acronym is internal to the project — treat the blocks as defined by their `forward()` signatures rather than relying on the name.

> **Note:** Verify with maintainer: the exact expansion of "DSRL." The repo's commit history mentions it as a research initiative, but no paper or doc inside the repo defines the acronym.

### ResFit

A residual-actor policy family: a learned actor outputs a small **residual** action on top of a base action, and a separate Q-function critic is trained to evaluate state-action pairs. The residual is scaled (multiplied by 0.25 in [`experiments/resfit/residual_actor.py`](../model_constructor/blocks/experiments/resfit/residual_actor.py) line 222) to constrain how far the actor can deviate from the base policy. The two registered blocks are `resfit_residual_actor` and `resfit_q_function`.

> **Note:** Verify with maintainer: the exact expansion of "ResFit" and whether it refers to a published method or an internal initiative.

### OpenPI / PI0

[OpenPI](https://github.com/Physical-Intelligence/openpi) is Physical Intelligence's open-source vision-language-action model project. **PI0** is one of its model variants — a transformer-based policy that takes camera frames + a language instruction and emits action chunks. The directory [`experiments/third_party/openpi/`](../model_constructor/blocks/experiments/third_party/openpi/) is a **vendored** copy of the OpenPI code; the only piece this repo's registry exposes is `openpi_batched` ([`openpi_batched_wrapper.py`](../model_constructor/blocks/experiments/third_party/openpi/openpi_batched_wrapper.py)), a batched-inference wrapper around `PI0Pytorch`.

### RadioV3

NVIDIA's [C-RADIO v3](https://github.com/NVlabs/RADIO) family of vision foundation models. The `radiov3` block ([`experiments/backbones/vision/radiov3.py`](../model_constructor/blocks/experiments/backbones/vision/radiov3.py)) loads `c-radio_v3-l` from `torch.hub` and produces `(features, summary)` from an input image. Used as a frozen feature extractor for downstream policies.

### Depth-Anything-3 (DA3)

A monocular-depth / 3D vision backbone from ByteDance ([repo](https://github.com/ByteDance-Seed/Depth-Anything-3)). The `da3` block ([`experiments/backbones/vision/depth_anything_3.py`](../model_constructor/blocks/experiments/backbones/vision/depth_anything_3.py)) wraps `DepthAnything3` and produces latent feature maps from images. **Requires the git submodule at `model_constructor/blocks/experiments/backbones/vision/externals/depth_anything_3/` to be initialized** — see [vision backbones README](../model_constructor/blocks/experiments/backbones/vision/README.md).

### ResNet34

Standard 34-layer residual CNN ([He et al., 2016](https://arxiv.org/abs/1512.03385)), loaded here from `torchvision.models.resnet34(pretrained=True)`. Used as a vision feature extractor across several experimental blocks (DSRL, ResFit, mutual-information estimator's state encoder). Registered as the `resnet34` block.

### Transformer encoder / decoder

Standard transformer building blocks from [Vaswani et al., 2017](https://arxiv.org/abs/1706.03762). Wrapped here in [`basic_blocks/transformer_encoder.py`](../model_constructor/blocks/basic_blocks/transformer_encoder.py) and [`basic_blocks/transformer_decoder.py`](../model_constructor/blocks/basic_blocks/transformer_decoder.py) with options for causal masking, batch-first vs sequence-first, etc.

### DUNE

[DUNE](https://europe.naverlabs.com/research/publications/dune/) is a Naver Labs vision backbone — a ViT-base model trained on a multi-task objective. The `DUNE` class is defined in [`experiments/backbones/vision/dune.py`](../model_constructor/blocks/experiments/backbones/vision/dune.py) but is **not registered** in the default registry (see [vision backbones README](../model_constructor/blocks/experiments/backbones/vision/README.md)).

### Cosmos-Reason2

NVIDIA's [Cosmos-Reason2](https://huggingface.co/nvidia/Cosmos-Reason2-2B) is a vision-language model. The `CosmosReason2Encoder` class is defined in [`experiments/backbones/vlm/cosmos_reason2.py`](../model_constructor/blocks/experiments/backbones/vlm/cosmos_reason2.py) but is **not registered** in the default registry.
