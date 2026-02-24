# `tests/`

pytest test suite for the `model_constructor` package.

## Running tests

```bash
# From the repository root
pytest

# Verbose output
pytest -v

# Single file
pytest tests/test_build_and_run.py
```

Configuration is in [`pyproject.toml`](../pyproject.toml):

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["."]
```

## Test files

### `test_build_and_run.py`

End-to-end smoke tests for `build_model()`.

- **`test_build_model_sequential_identity_runs`**: Builds a sequential model with `nn.Identity`, verifies output shape matches input.
- **`test_build_model_graph_skip_add_runs`**: Builds a graph model with `nn.Identity` + `op:add`, verifies `y == x + x`.

These tests validate the full pipeline: config resolution → IR compilation → module instantiation → forward pass.

### `test_graph_compiler_contracts.py`

GraphIR compilation contracts using `compile_ir()`.

- **`test_nodes_mapping_requires_order`**: Verifies that `model.graph.nodes` as a mapping without `order` raises `ConfigError`.
- **`test_forward_reference_forbidden`**: Verifies that a node referencing a not-yet-produced value raises `ConfigError`.

### `test_defaults_and_merge.py`

YAML `defaults` composition and merge behavior using `resolve_config()`.

- **`test_defaults_merge_and_cycle`**: Verifies that child configs properly inherit and override parent `params`, and that circular `defaults` includes are detected and rejected with `ConfigError`.

### `test_templates_and_interpolation.py`

Template expansion and interpolation using `resolve_config()`.

- **`test_template_expansion`**: Verifies `_template_` expansion with local overrides (template values merged, local wins).
- **`test_interpolation_type_preservation`**: Verifies `${...}` interpolation preserves types for full-scalar references and stringifies for embedded references. Also tests typed env var forms (`${env:int:...}`).

## Key imports used in tests

| Import | Source |
|--------|--------|
| `build_model` | `model_constructor` ([`model_constructor/api.py`](../model_constructor/api.py)) |
| `compile_ir` | `model_constructor` ([`model_constructor/api.py`](../model_constructor/api.py)) |
| `resolve_config` | `model_constructor.config.resolve` ([`model_constructor/config/resolve.py`](../model_constructor/config/resolve.py)) |
| `ConfigError` | `model_constructor.errors` ([`model_constructor/errors.py`](../model_constructor/errors.py)) |

## Writing new tests

- Tests use inline Python dicts or `tmp_path` YAML files (no dependency on config files in `configs/`).
- For compile-time checks, use `compile_ir()` (skips instantiation).
- For resolution checks, use `resolve_config()` (skips compilation).
- For full end-to-end checks, use `build_model()`.
- Use `pytest.raises(ConfigError, match=...)` for expected errors.
