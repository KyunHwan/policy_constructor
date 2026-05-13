# `model_constructor.graph`

The architecture-agnostic model backend. Two pieces:

- **`GraphIR`** — a small, explicit frozen-dataclass representation of a model as **named modules + ordered runtime steps + outputs**.
- **`GraphModel`** — a generic `torch.nn.Module` subclass that executes `GraphIR` on each forward pass.

If you're new to the terms **DAG**, **runtime reference**, **return policy**, see [docs/GLOSSARY.md](../../docs/GLOSSARY.md).

The mental-model context is [docs/MENTAL_MODEL.md](../../docs/MENTAL_MODEL.md) — in particular the [GraphIR vs GraphModel section](../../docs/MENTAL_MODEL.md#graphir-vs-graphmodel--data-vs-executor).

---

## `GraphIR` in one paragraph

Defined in [`ir.py`](ir.py). Fields:

- `inputs: list[str]` — names the model accepts in `forward()`.
- `modules: dict[str, ModuleIR]` — named *specs* (not yet instantiated; that happens in `instantiate/`).
- `steps: list[StepIR]` — the ordered execution plan. Each `StepIR` is either a module call or an op call, with positional + keyword args (which may contain `Ref` runtime references), and a designated output (single name, list of names, or dict of names).
- `outputs: list[str] | dict[str, str]` — what to return from `forward()`.
- `return_policy: "single" | "tuple" | "dict"` — how to pack the outputs.
- `ops: dict[str, Callable]` — the registered ops referenced by any `op:` step, looked up at compile time.

All three YAML frontends (`model.sequential`, `model.graph` with `nodes:` list, `model.graph` with `nodes:` mapping + `order:`, `model.graph` with `steps:`) compile to this same `GraphIR`. The compiler is in [`compiler.py`](compiler.py).

---

## Runtime references and escaping

Step `args:` and `kwargs:` may contain runtime references:

- `$name` — at forward time, look up `name` in the runtime context dict. Producers are: model inputs and earlier step outputs.
- `$$literal` — escape; becomes the literal string `$literal`. Use this when you genuinely need a YAML string that starts with `$`.
- Anything else — passed through unchanged (ints, floats, bools, lists, dicts).

Note: `${...}` (with braces) is **interpolation** handled at config-resolve time and is unrelated to runtime references. By the time `compile_ir()` runs, `${...}` is already resolved to a concrete value.

Compile-time parsing: [`compiler.py`](compiler.py) `_parse_runtime_value()` ~line 264. Runtime resolution: [`model.py`](model.py) `_resolve_runtime_value()` ~line 91.

---

## DAG-only — what's enforced

`GraphIR` is a DAG. The compiler ([`compiler.py`](compiler.py) `_validate_ir()` ~line 292) walks the steps in order and maintains an `available` set of names — initially just the model inputs. For each step:

- If the step calls a `module:<name>`, the module must exist in `ir.modules`. Else `ConfigError: Step references unknown module 'X'`.
- If the step calls an `op:<name>`, the op must exist in the registry (already verified during `_collect_ops`).
- Every `$ref` in `args:`/`kwargs:` must be in `available`. Otherwise: `ConfigError: Forward reference(s) not allowed: [missing_name]`.
- Every output name the step produces must **not** already be in `available`. Otherwise: `ConfigError: Output name collision(s): [name]`.

After all steps: every name in `outputs:` must be in `available`. `return: single` requires exactly one output; `return: dict` requires an outputs mapping; `return: tuple` requires an outputs list.

If you need recurrence (a step's output feeds back into an earlier step), wrap the loop inside a single block module and call that module once from the graph. The graph captures the topology of one forward pass, not the training loop.

---

## Step-by-step trace of `GraphModel.forward()` — a tiny example

Take [`configs/examples/graph_skip_add.yaml`](../../configs/examples/graph_skip_add.yaml) (treating it conceptually — the YAML itself uses unregistered types, but the IR shape is identical to what an inline-dict equivalent would produce):

```yaml
schema_version: 1
model:
  graph:
    inputs: [x]
    modules:
      stem: {_type_: nn.Identity}
      main: {_type_: nn.Identity}
    nodes:
      h1: {call: module:stem, args: [$x]}
      h2: {call: module:main, args: [$h1]}
      h3: {call: op:add, args: [$h1, $h2]}
    order: [h1, h2, h3]
    outputs: [$h3]
    return: single
```

(I substituted `nn.Identity` for `conv_bn_act` so this is runnable today; the topology is the same.)

After compile, the IR has:

```python
GraphIR(
    inputs=["x"],
    modules={"stem": ModuleIR(spec=...), "main": ModuleIR(spec=...)},
    steps=[
        StepIR(call_kind="module", call_name="stem", args=[Ref("x")],  kwargs={}, out="h1", ...),
        StepIR(call_kind="module", call_name="main", args=[Ref("h1")], kwargs={}, out="h2", ...),
        StepIR(call_kind="op",     call_name="add",  args=[Ref("h1"), Ref("h2")], kwargs={}, out="h3", ...),
    ],
    outputs=["h3"],
    return_policy="single",
    ops={"add": operator.add},   # looked up from registry at compile time
)
```

The `_validate_ir` pass walked these in order, growing `available = {"x"} → {"x","h1"} → {"x","h1","h2"} → {"x","h1","h2","h3"}`, with every reference satisfied at each step.

Now you call `model(x_tensor)`. Inside [`model.py`](model.py) `forward()` (~line 31):

1. **Bind inputs.** `_bind_inputs((x_tensor,), {})` walks `self.inputs = ["x"]`, consumes one positional arg → `ctx = {"x": x_tensor}`. Errors if too many or too few positionals/kwargs were given.
2. **Execute steps in order.** For each `StepIR`:
   - **Step 0** (`stem(x)`): resolved args → `[x_tensor]` (the `Ref("x")` is looked up in `ctx`). Call `self.graph_modules["stem"](x_tensor)`. Result is `x_tensor` (since stem is `nn.Identity`). Write `ctx["h1"] = x_tensor` via `_write_outputs`.
   - **Step 1** (`main(h1)`): resolved args → `[ctx["h1"]] = [x_tensor]`. Call `self.graph_modules["main"](x_tensor)` → `x_tensor`. Write `ctx["h2"] = x_tensor`.
   - **Step 2** (`add(h1, h2)`): resolved args → `[x_tensor, x_tensor]`. Call `self._ops["add"](x_tensor, x_tensor)` → `2 * x_tensor`. Write `ctx["h3"] = 2 * x_tensor`.
3. **Pack outputs.** `_pack_outputs(ctx)` — `return_policy="single"`, outputs list `["h3"]` → returns `ctx["h3"]` directly (not wrapped in a tuple).

If any step raises, [`model.py`](model.py) wraps it as `GraphExecutionError: step[i] failed (module:stem): <underlying>` (~line 42). The wrapper preserves the underlying exception via `from exc`, so the original traceback is still available.

---

## Mapping-form `nodes:` vs list-form — they compile to the same IR

The compiler accepts three input shapes for graph nodes; all produce the same `list[StepIR]`. Concrete worked example:

### Form 1 — list of nodes (canonical, self-ordering)

```yaml
nodes:
  - {name: h1, call: module:stem, args: [$x]}
  - {name: h2, call: op:add, args: [$h1, $h1]}
```

The compiler iterates the list and calls `_compile_node()` on each one ([`compiler.py`](compiler.py) ~line 128).

### Form 2 — mapping of nodes + explicit `order:`

```yaml
nodes:
  h1: {call: module:stem, args: [$x]}
  h2: {call: op:add, args: [$h1, $h1]}
order: [h1, h2]
```

The compiler walks `order:`, looks up each entry in the `nodes:` mapping, attaches the name field, and processes it via the same `_compile_node()` path ([`compiler.py`](compiler.py) ~line 137). The `order:` is required because the YAML loader does not guarantee mapping iteration order across implementations.

### Form 3 — explicit `steps:` (power-user, no nodes)

```yaml
steps:
  - {call: module:stem, args: [$x], out: h1}
  - {call: op:add, args: [$h1, $h1], out: h2}
```

The compiler runs `_compile_steps_list()` ([`compiler.py`](compiler.py) ~line 196). The difference from `nodes:` is that `steps:` requires an explicit `out:` (no name field shorthand) — slightly more verbose, less syntactic sugar, identical semantics.

All three produce the same `list[StepIR]`, in the same order, and the rest of the pipeline can't tell them apart.

---

## Outputs and return policies

Compile-time logic is in [`compiler.py`](compiler.py) `_parse_outputs()` / `_coerce_outputs()` / `_coerce_return_policy()` (~lines 365-432).

| YAML `outputs:` form | Implicit `return:` (if omitted) | Forward-pass return type |
|---|---|---|
| `outputs: [$h3]` | `single` | bare tensor |
| `outputs: [$h2, $h3]` | `tuple` | `(tensor, tensor)` |
| `outputs: {logits: $h3, features: $h2}` | `dict` | `{"logits": tensor, "features": tensor}` |

If you supply `return:` explicitly, it must be consistent — a `dict` return needs a mapping `outputs:`, a `single` return needs exactly one output, etc. Mismatch → `ConfigError`.

Runtime: [`model.py`](model.py) `_pack_outputs()` (~line 80).

---

## Multi-output step writes — when `out:` is a list or dict

A single step can produce multiple named runtime values if the underlying module/op returns a tuple, list, or dict. Tell the executor how to unpack via `out:`:

- `out: name` (string) — single output; whatever the module returned is bound to `name`.
- `out: [a, b]` (list) — module/op must return a 2-tuple or 2-list. Bound positionally.
- `out: {key1: name1, key2: name2}` (dict) — module/op must return a dict. Each `key` is plucked from the result and bound to its `name`.

Mismatches (e.g., declaring `out: [a, b]` but the module returned a scalar) raise `GraphExecutionError` at runtime. See [TROUBLESHOOTING.md](../../docs/TROUBLESHOOTING.md#expected-tuplelist-result-for-list-out-mapping).

---

## Weight sharing in graphs

If two steps call the same `module:<name>`, they share the underlying parameters — there's literally one `nn.Module` instance stored in `self.graph_modules[name]` (a [`torch.nn.ModuleDict`](https://pytorch.org/docs/stable/generated/torch.nn.ModuleDict.html)), used both times. Concrete example:

```yaml
modules:
  shared: {_type_: my_block, width: 128}
nodes:
  - {name: h1, call: module:shared, args: [$x]}
  - {name: h2, call: module:shared, args: [$h1]}   # same weights as h1's call
```

In `model.parameters()` you'll see `graph_modules.shared.*` listed once. If you want independent copies, declare them with different names:

```yaml
modules:
  block_a: {_type_: my_block, width: 128}
  block_b: {_type_: my_block, width: 128}
```

`block_a` and `block_b` will each have their own parameter set.

---

## Error classes raised here

All defined in [`../errors.py`](../errors.py):

- `ConfigError` — compile-time issues (forward references, missing modules, etc.).
- `GraphExecutionError` — runtime issues (missing input, step failure, output unpacking mismatch).

Both wrap the underlying exception with context. See [TROUBLESHOOTING.md](../../docs/TROUBLESHOOTING.md) for the complete catalog of error messages and their fixes.
