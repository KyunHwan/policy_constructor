# `model_constructor.instantiate`

The single engine that turns a resolved YAML spec into a Python object — almost always a `torch.nn.Module`.

If you understand the [`registry/`](../registry/README.md), this module is essentially "given a registered factory, build it with the right kwargs while validating against its signature."

If you're new to the terms **spec**, **registry**, **signature policy**, see [docs/GLOSSARY.md](../../docs/GLOSSARY.md). Mental-model context: [docs/MENTAL_MODEL.md](../../docs/MENTAL_MODEL.md).

---

## What's a spec, exactly

Any YAML dict containing `_type_` (or the gated `_target_`) is a spec. The instantiator's job is to build it. Detection is in [`instantiate.py`](instantiate.py) `_is_spec()` line 85:

```python
def _is_spec(obj: Any) -> bool:
    return isinstance(obj, dict) and ("_type_" in obj or "_target_" in obj)
```

Three kinds of YAML data flow through `instantiate_value`:

- **A spec dict** → resolved to a callable + kwargs, instantiated.
- **A plain dict that isn't a spec** → walked recursively (each value processed independently).
- **A list** → walked recursively (each element processed independently).
- **Anything else** (scalar, None) → returned unchanged.

Recursion stops at the leaves (non-dict/list values) and at spec dicts (which are then handled by `_instantiate_spec`, which in turn recurses into its own kwargs).

---

## Reserved spec keys

| Key | Purpose |
|---|---|
| `_type_` | Registry-key lookup. Preferred. |
| `_target_` | Import-by-string. Gated by `settings.allow_target` and `settings.allowed_import_prefixes`. |
| `_args_` | Optional list of positional arguments. Rare. |
| `_kwargs_` | Optional dict of extra keyword arguments. Rare. |
| `_name_` | Reserved; not currently consumed by the instantiator. |

Everything else in the spec dict is a **kwarg** (passed by name to the target). The convention in this repo is that block constructors use **keyword-only arguments** (with `*` in the signature) — see [`blocks/basic_blocks/mlp.py`](../blocks/basic_blocks/mlp.py) for an example. Keyword-only args mean:

- The signature is self-documenting at the call site.
- A typo in YAML (`widht` instead of `width`) becomes a clear `ConfigError` (with a `difflib` suggestion) instead of a silent positional misalignment.

In `strict` mode (default for blocks registered in this repo), any reserved-style key (`_xyz_`) that isn't in the recognized list raises `ConfigError: Unknown reserved keys: [...]`. This catches typos like `_targets_` or `_typ_`.

---

## Worked example — tracing one spec from CFG-VQVAE

Take the first module in [`configs/experiments/cfg_vqvae_flow_matching.yaml`](../../configs/experiments/cfg_vqvae_flow_matching.yaml):

```yaml
modules:
  vqvae_posterior:
    _type_: cfg_vqvae_posterior
    cond_proprio_dim: 62
    cond_visual_dim: 1072
    transformer_d_model: 384
    transformer_nhead: 16
    transformer_dim_feedforward: 2048
    transformer_dropout: 0.2
    transformer_activation: gelu
    transformer_batch_first: true
    transformer_is_causal: false
    transformer_num_layers: 12
    transformer_num_tokens: 300
    action_dim: 24
    use_cond_semantic: false
    use_cond_semantic_projection: false
    cond_semantic_dim: null
```

`build_model()` reaches the instantiation stage with this dict (the `${params.*}` interpolations have already been replaced with concrete values during the resolve stage). Here's what `instantiate_module_spec(spec, ...)` does, step by step:

1. **Spec detection** — sees `_type_` key, recognizes it as a spec. Goes to `_instantiate_spec`.
2. **Args/kwargs separation** — `_args_` is missing → empty list. `_kwargs_` is missing → empty dict. Inline kwargs are every non-reserved key: `cond_proprio_dim: 62`, ..., `cond_semantic_dim: None`. No overlap between inline and `_kwargs_`, no duplicate keys.
3. **Strict-mode reserved-key check** — `strict=true` by default. No keys start with `_` other than `_type_`. Passes.
4. **Recursive kwarg instantiation** — each kwarg value is run through `instantiate_value`. `62`, `True`, `"gelu"`, `None` are scalars → returned unchanged. No nested specs in this particular module's kwargs. (If a kwarg value had been e.g. `act: {_type_: nn.Identity}`, it would have been recursively instantiated into an `nn.Identity()` instance and then passed to the parent constructor.)
5. **Target resolution** — `_type_: cfg_vqvae_posterior` → `registry.get_module("cfg_vqvae_posterior")` → `RegistryEntry` whose `.target` is the `VQVAE_Posterior` class.
6. **Signature validation** — the entry's `signature_policy` is `"strict"`. The validator (in [`signature.py`](signature.py)) calls `inspect.signature(VQVAE_Posterior)`, computes the set of allowed parameter names (positional-or-keyword + keyword-only), and checks that every kwarg in our spec is in that set. If `widht` had been there instead of `width`, this is where the `ConfigError: Unknown kwargs: ['widht'], suggestions: ['width']` would fire.
7. **Construct** — calls `VQVAE_Posterior(**resolved_kwargs)`. If the constructor raises (e.g., an internal `assert`), the error is wrapped:

   ```
   ConfigError: Instantiation failed: <underlying message>  |  path=model.graph.modules.vqvae_posterior  |  loc=cfg_vqvae_flow_matching.yaml:30:7
   ```

8. **Return type check** — `instantiate_module_spec` (the outer wrapper) asserts the result is a `torch.nn.Module`. The `_instantiate_spec` helper is also used internally for nested non-Module values (e.g., if a kwarg's value was itself a spec for a `torchvision.transforms.Normalize` object).

The result is a fully-constructed `VQVAE_Posterior(...)` instance that the `GraphModel` will store in `self.graph_modules["vqvae_posterior"]`.

---

## Nested specs in kwargs — concrete example

`conv_bn_act` accepts an `act` kwarg whose value can be either a default `nn.ReLU` (used when `act=None`) or any `nn.Module` you specify. The YAML way to override is to pass a nested spec:

```yaml
modules:
  main:
    _type_: conv_bn_act
    in_channels: 16
    out_channels: 16
    kernel_size: 3
    padding: 1
    act: {_type_: nn.Identity}        # nested spec — instantiated before being passed to conv_bn_act
```

The instantiator handles this by recursing on every kwarg value: when it encounters `act: {_type_: nn.Identity}`, it calls `instantiate_value(...)` on it, which recognizes the inner dict as a spec and builds an `nn.Identity()` first. The `conv_bn_act` constructor then receives `act=<nn.Identity instance>`.

If you build deeper nests — e.g., a module whose kwarg is a list of submodule specs — the same recursion applies to every list element.

---

## Signature policies — what each means and when to pick which

The three modes are defined in [`signature.py`](signature.py). Decision table:

| Policy | What `validate_kwargs` does | When to pick it |
|---|---|---|
| **`strict`** | Calls `inspect.signature(target)`. If introspection fails, raises (broken registration). If it succeeds, rejects any kwarg not in the recognized parameter set. | Default for blocks in [`../blocks/register.py`](../blocks/register.py). Use for any block whose signature you control and want enforced. Catches YAML typos. |
| **`best_effort`** | Tries to introspect; if it fails (some C extensions, descriptor-based classes), silently allows anything. Otherwise validates like `strict`. | Default for built-in `nn.*` modules in [`../registry/builtins.py`](../registry/builtins.py). Use when introspection might be unreliable for the target. |
| **`runtime_only`** | Never pre-validates. Pass kwargs straight to the call site. | Default for ops. Use when you want errors deferred to the call (rare for modules). |

If your block's constructor accepts `**kwargs`, signature validation always passes regardless of mode — `inspect.signature` reports a `VAR_KEYWORD` parameter, and the validator returns early.

### Example — what `strict` catches

```yaml
- _type_: mlp
  dimms: [null, 256, 10]       # typo: dimms → dims
  dropout: 0.1
```

Result:

```
ConfigError: Unknown kwargs: ['dimms']  |  path=model.sequential.layers[0]  |  suggestions=dims
```

Without `strict`, you'd get a less helpful error at construction time:

```
TypeError: MLP.__init__() got an unexpected keyword argument 'dimms'
```

The `strict` policy turns a runtime construction failure into a YAML-level "did you mean..." diagnostic.

### Example — when to pick `runtime_only`

A module whose signature is dynamic at construction time (e.g., the constructor reflects on a config object to decide what kwargs to accept). Pre-validation can't work because the parameter set isn't statically inspectable. Register with `runtime_only` and the call site will fail with a clear underlying error if something's wrong.

---

## `_args_` and `_kwargs_` — when you actually need them

99% of specs use only inline kwargs. The two exceptions:

### When to use `_args_`

The constructor takes positional-only args (no keyword form). Example: `nn.Linear(in_features, out_features)` accepts these positionally. You *can* write:

```yaml
_type_: nn.Linear
_args_: [128, 256]
```

But the more readable form, which works because `nn.Linear`'s parameters are `POSITIONAL_OR_KEYWORD`, is:

```yaml
_type_: nn.Linear
in_features: 128
out_features: 256
```

Use `_args_` only when the target's parameters are genuinely positional-only and have no readable keyword form.

### When to use `_kwargs_`

You want to pass a kwarg whose name would collide with a reserved word or with a YAML-special character. Example, if for some reason a block's constructor accepted a kwarg literally named `_type_` (which would be a bad idea, but possible):

```yaml
_type_: my_block
_kwargs_:
  _type_: "something"     # forwarded as a kwarg, escaped from the reserved-key check
```

In practice, this is exceedingly rare. The keyword-only convention used by all blocks in this repo (`def __init__(self, *, width: int, dropout: float = 0.0):`) makes inline kwargs the obviously-correct default.

---

## Safety defaults you should know

- **`_type_` resolves through the registry**, which means the set of buildable types is known up front. You can list it (`registry.list_modules()`); you can audit it (it's just `register.py` and `builtins.py`).
- **`_target_` is off by default.** Enabling it (`settings.allow_target: true`) lets a YAML import arbitrary Python attributes by dotted path. The prefix allowlist (`settings.allowed_import_prefixes`) limits what's reachable. This is enforced in [`instantiate.py`](instantiate.py) `_resolve_target()` line 165.
- **The reserved-key namespace (`_xxx_`) is the system's**, not the user's. Don't name your kwargs that way.
- **Settings are frozen.** `Settings` is a `@dataclass(frozen=True)`; you can't mutate it at runtime. Override them only via YAML `settings:`.

---

## Common errors raised from this module

All wrap context (`config_path`, `location`) into the error message. Documented with fixes in [TROUBLESHOOTING.md](../../docs/TROUBLESHOOTING.md):

- `Spec may not contain both _type_ and _target_`
- `Spec did not produce a torch.nn.Module (got X)`
- `_type_ must be a non-empty string` / `_target_ must be a non-empty string`
- `_target_ is disabled by settings.allow_target`
- `_target_ import 'X' is not allowed by settings.allowed_import_prefixes`
- `_args_ must be a list` / `_kwargs_ must be a mapping`
- `Duplicate kwargs keys: [...]`
- `Unknown reserved keys: [...]`
- `Unknown kwargs: [...]`
- `Instantiation failed: <underlying>`
