# Documentation Index

A single navigation hub for the `policy_constructor` repository.
This index is organized by **what you're trying to do**, not by file layout. Pick the row that matches your situation and read in the order shown.

If you just landed here from a search engine and have no context, start with [QUICKSTART.md](QUICKSTART.md).

---

## I just want to run something

1. [QUICKSTART.md](QUICKSTART.md) — clone → build → forward pass in about 30 minutes. Copy-paste script.
2. Root [README.md "Quickstart"](../README.md#quickstart) — three minimal `build_model()` examples (Python dict, graph dict, YAML file).
3. [../examples/build_and_run.py](../examples/build_and_run.py) — the runnable script the Quickstart wraps.

Read [QUICKSTART.md](QUICKSTART.md) **before** anything else if this is your first day.

---

## I want to understand this codebase

1. [MENTAL_MODEL.md](MENTAL_MODEL.md) — the "whiteboard talk." Construct-only philosophy, the two YAML frontends, what is in `GraphIR` vs `GraphModel`, when to use modules vs ops. Read this once and you'll understand the rest.
2. [../model_constructor/README.md](../model_constructor/README.md) — directory map and worked examples for adding custom blocks.
3. [../model_constructor/graph/README.md](../model_constructor/graph/README.md) — how `GraphIR` is compiled and executed.
4. [../model_constructor/instantiate/README.md](../model_constructor/instantiate/README.md) — how YAML `_type_` specs turn into Python objects.

After [MENTAL_MODEL.md](MENTAL_MODEL.md), the component READMEs become much easier to follow because you already know what slot each piece fills.

---

## I want to write YAML

1. [../model_constructor/config/authoring_yaml.md](../model_constructor/config/authoring_yaml.md) — practical guide: `defaults`, `_template_`, `_merge_`, `${...}` interpolation, both frontends.
2. [../model_constructor/config/schema_v1.md](../model_constructor/config/schema_v1.md) — normative contract. This is what the parser actually enforces. If `authoring_yaml.md` is the cookbook, `schema_v1.md` is the law.
3. [../model_constructor/config/README.md](../model_constructor/config/README.md) — high-level resolution pipeline.
4. [../configs/README.md](../configs/README.md) — annotated walkthrough of every example YAML in `configs/`.

Read [authoring_yaml.md](../model_constructor/config/authoring_yaml.md) first; only consult [schema_v1.md](../model_constructor/config/schema_v1.md) when you need to know exactly what's allowed.

---

## I want to add a custom block

1. [../model_constructor/blocks/README.md](../model_constructor/blocks/README.md) — Option A (edit `register.py`) vs Option B (parent-repo `imports:`).
2. [../examples/end_to_end.md](../examples/end_to_end.md) — full two-parent-repo walkthrough.
3. [../model_constructor/registry/README.md](../model_constructor/registry/README.md) — what `signature_policy` means and how `register_module` / `register_op` differ.
4. [QUICKSTART.md](QUICKSTART.md) Variant 2 — minimal custom-block example you can copy.

Option B (parent-repo imports) is what you want for production. Option A is fine for experimenting inside this clone.

---

## I want to use this from my training or inference repo

1. Root [README.md "Integration"](../README.md#integration-training--inference-repositories) — the construct-only contract.
2. [../examples/end_to_end.md](../examples/end_to_end.md) — concrete two-repo layout (training + inference both vendoring this as a submodule, sharing one YAML and one block-registration module).
3. Root [README.md "Installation"](../README.md#installation) — submodule add and `PYTHONPATH` setup.

This repo deliberately does not ship a trainer, dataloader, checkpoint format, or inference server. Your parent repo owns all of those.

---

## Something broke

1. [TROUBLESHOOTING.md](TROUBLESHOOTING.md) — decision-tree by error type. Every error class in [`errors.py`](../model_constructor/errors.py) is covered, with example messages and exact fixes.
2. Root [README.md "Troubleshooting"](../README.md#troubleshooting) — overlaps with the above; the longer guide is in [TROUBLESHOOTING.md](TROUBLESHOOTING.md).
3. [GLOSSARY.md](GLOSSARY.md) — if the error message contains a term you don't recognize.

The flow is: read the error message → find it in [TROUBLESHOOTING.md](TROUBLESHOOTING.md) → if the fix references a term you don't know, jump to [GLOSSARY.md](GLOSSARY.md).

---

## I don't know what X means

[GLOSSARY.md](GLOSSARY.md) defines every codebase-specific term (registry, spec, signature policy, GraphIR, source map, runtime reference, etc.) and every research-y term you'll see in the experimental blocks (flow matching, VQ-VAE, CFG, DSRL, ResFit, OpenPI, etc.). Each entry includes a "see also" pointer.

---

## I want to dig into the experimental blocks (research code)

1. [../model_constructor/blocks/experiments/README.md](../model_constructor/blocks/experiments/README.md) — complete table of registry keys grouped by research area (CFG-VQVAE, VFP, naive flow matching, mutual-information estimator, DSRL, ResFit, OpenPI).
2. [../model_constructor/blocks/experiments/backbones/vision/README.md](../model_constructor/blocks/experiments/backbones/vision/README.md) — RadioV3, Depth-Anything-3 (uses a git submodule), ResNet34, DUNE, Cosmos-Reason2.
3. [../model_constructor/blocks/experiments/templates/README.md](../model_constructor/blocks/experiments/templates/README.md) — abstract base classes for policy components.
4. [../model_constructor/blocks/experiments/utils/README.md](../model_constructor/blocks/experiments/utils/README.md) — positional encoding and time-embedding helpers.
5. [../model_constructor/blocks/experiments/third_party/README.md](../model_constructor/blocks/experiments/third_party/README.md) — vendored OpenPI codebase (you almost never touch this directly).
6. [../configs/experiments/cfg_vqvae_flow_matching.yaml](../configs/experiments/cfg_vqvae_flow_matching.yaml) and the annotated walkthrough in [../configs/README.md](../configs/README.md).
7. [../examples/smoke_cfg_vqvae_flow_matching.py](../examples/smoke_cfg_vqvae_flow_matching.py) — exact tensor shapes the CFG-VQVAE config expects.

If a term is new, look it up in [GLOSSARY.md](GLOSSARY.md) before reading the source.

---

## I want to contribute

1. Root [README.md "Contributing"](../README.md#contributing) — style guide and PR checklist.
2. [../tests/README.md](../tests/README.md) — how to add a test for your new block.
3. [MENTAL_MODEL.md](MENTAL_MODEL.md) — the "what this repo does and does not do" section. Don't add training-loop or dataset code here.

---

## Map of every doc file in the repository

The list below is comprehensive — every Markdown file in the repo.

**Top-level docs (`docs/`)** — what to read first.

- [INDEX.md](INDEX.md) — this file.
- [QUICKSTART.md](QUICKSTART.md) — first-hour walkthrough.
- [MENTAL_MODEL.md](MENTAL_MODEL.md) — how to think about the codebase.
- [GLOSSARY.md](GLOSSARY.md) — terminology.
- [TROUBLESHOOTING.md](TROUBLESHOOTING.md) — decision-tree error guide.

**Root**

- [../README.md](../README.md) — overview, install, build pipeline, integration patterns.

**Component docs (near the code)**

- [../model_constructor/README.md](../model_constructor/README.md) — package directory map and extension tutorials.
- [../model_constructor/config/README.md](../model_constructor/config/README.md) — YAML resolution pipeline.
- [../model_constructor/config/authoring_yaml.md](../model_constructor/config/authoring_yaml.md) — YAML authoring guide.
- [../model_constructor/config/schema_v1.md](../model_constructor/config/schema_v1.md) — normative YAML contract.
- [../model_constructor/registry/README.md](../model_constructor/registry/README.md) — module/op registry.
- [../model_constructor/instantiate/README.md](../model_constructor/instantiate/README.md) — spec-to-object engine.
- [../model_constructor/graph/README.md](../model_constructor/graph/README.md) — `GraphIR` and `GraphModel`.
- [../model_constructor/util/README.md](../model_constructor/util/README.md) — plugin imports mechanism.
- [../model_constructor/blocks/README.md](../model_constructor/blocks/README.md) — custom-block extension points.
- [../model_constructor/blocks/basic_blocks/README.md](../model_constructor/blocks/basic_blocks/README.md) — reusable `nn.Module` primitives.
- [../model_constructor/blocks/experiments/README.md](../model_constructor/blocks/experiments/README.md) — experimental block catalog.
- [../model_constructor/blocks/experiments/backbones/vision/README.md](../model_constructor/blocks/experiments/backbones/vision/README.md) — vision feature extractors.
- [../model_constructor/blocks/experiments/templates/README.md](../model_constructor/blocks/experiments/templates/README.md) — abstract base classes.
- [../model_constructor/blocks/experiments/utils/README.md](../model_constructor/blocks/experiments/utils/README.md) — pos/time encoding helpers.
- [../model_constructor/blocks/experiments/third_party/README.md](../model_constructor/blocks/experiments/third_party/README.md) — vendored OpenPI overview.

**Examples & configs**

- [../examples/README.md](../examples/README.md) — runnable example scripts.
- [../examples/end_to_end.md](../examples/end_to_end.md) — two-repo submodule walkthrough.
- [../configs/README.md](../configs/README.md) — annotated walkthrough of example YAMLs.

**Tests**

- [../tests/README.md](../tests/README.md) — how to run and how to add tests.
