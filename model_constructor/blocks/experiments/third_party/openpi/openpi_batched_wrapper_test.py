"""Tests for OpenPiBatchedWrapper with dummy data and random weights (no checkpoint).

Verifies the batched wrapper can be instantiated on GPU with a tiny "dummy" model,
load dummy observation variables, and run inference / training forward passes.

Also tests that the wrapper works when built through the model registry
(``build_model`` / ``PolicyConstructorModelFactory``), matching the pattern
used by ``offline_trainer.py``.

Requirements:
  - torch (with CUDA)
  - transformers==4.53.2 (exact version)
  - transformers_replace files copied into installed transformers package
  - pytest

Run:
  pytest .../openpi/openpi_batched_wrapper_test.py -v
  # or
  python .../openpi/openpi_batched_wrapper_test.py
"""

from __future__ import annotations

import contextlib
import dataclasses
import os
import sys
from unittest.mock import patch

import numpy as np
import pytest
import torch

# Ensure openpi is importable and JAX doesn't compete for GPU
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

_THIRD_PARTY = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _THIRD_PARTY not in sys.path:
    sys.path.insert(0, _THIRD_PARTY)

# ---------------------------------------------------------------------------
# Skip the entire module if transformers_replace is not installed correctly
# ---------------------------------------------------------------------------
try:
    from transformers.models.siglip.check import (
        check_whether_transformers_replace_is_installed_correctly,
    )

    _has_transformers_replace = (
        check_whether_transformers_replace_is_installed_correctly()
    )
except ImportError:
    _has_transformers_replace = False

pytestmark = [
    pytest.mark.skipif(
        not _has_transformers_replace,
        reason=(
            "Requires transformers==4.53.2 with transformers_replace files copied into "
            "the installed transformers package."
        ),
    ),
    pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="Requires a CUDA GPU.",
    ),
]

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# Constants matching the pi05_igris + Pi0Config defaults
ACTION_DIM_OUTPUT = 24  # IgrisOutputs clips to 24
ACTION_DIM_MODEL = 32  # Pi0Config default action_dim
ACTION_HORIZON = 50

# ---------------------------------------------------------------------------
# Tiny model helper — reused from pi0_pytorch_test.py
# ---------------------------------------------------------------------------
from openpi.models_pytorch.pi0_pytorch_test import _tiny_paligemma_init


# ---------------------------------------------------------------------------
# Shared patching helpers
# ---------------------------------------------------------------------------


def _get_dummy_config(dtype: str = "float32"):
    """Return (dummy_config, _train_config, PaliGemmaWithExpertModel) for patching."""
    from openpi.models_pytorch.gemma_pytorch import PaliGemmaWithExpertModel
    from openpi.training import config as _train_config

    real_config = _train_config.get_config("pi05_igris")
    dummy_model_config = dataclasses.replace(
        real_config.model,
        paligemma_variant="dummy",
        action_expert_variant="dummy",
        dtype=dtype,
    )
    dummy_config = dataclasses.replace(real_config, model=dummy_model_config)
    return dummy_config, _train_config, PaliGemmaWithExpertModel


@contextlib.contextmanager
def _dummy_patches(dummy_config, _train_config, PaliGemmaWithExpertModel):
    """Context manager that applies the 3 patches needed for tiny model creation."""
    with (
        patch.object(_train_config, "get_config", return_value=dummy_config),
        patch("torch.compile", side_effect=lambda fn, **kwargs: fn),
        patch.object(
            PaliGemmaWithExpertModel, "__init__", _tiny_paligemma_init
        ),
    ):
        yield


# ---------------------------------------------------------------------------
# Wrapper creation (direct)
# ---------------------------------------------------------------------------


def _create_wrapper(dtype: str = "float32"):
    """Create a tiny OpenPiBatchedWrapper with random weights on GPU.

    Patches the train config to use dummy gemma variants and a tiny
    PaliGemma init so the model fits in minimal GPU memory.
    """
    from openpi.openpi_batched_wrapper import OpenPiBatchedWrapper

    dummy_config, _train_config, PaliGemmaWithExpertModel = _get_dummy_config(dtype)

    with _dummy_patches(dummy_config, _train_config, PaliGemmaWithExpertModel):
        wrapper = OpenPiBatchedWrapper(
            train_config_name="pi05_igris",
            ckpt_dir=None,  # random weights, no norm_stats
            action_dim=ACTION_DIM_OUTPUT,
            action_horizon=ACTION_HORIZON,
            num_inference_steps=2,  # minimal steps for speed
        )

    wrapper = wrapper.to(DEVICE)
    return wrapper


# ---------------------------------------------------------------------------
# Wrapper creation via registry / factory
# ---------------------------------------------------------------------------

# Graph config matching openpi_batched.yaml but with ckpt_dir=None and fast settings
_GRAPH_CFG = {
    "schema_version": 1,
    "model": {
        "graph": {
            "inputs": ["observation", "noise"],
            "modules": {
                "openpi_model": {
                    "_type_": "openpi_batched",
                    "train_config_name": "pi05_igris",
                    "ckpt_dir": None,
                    "action_dim": ACTION_DIM_OUTPUT,
                    "action_horizon": ACTION_HORIZON,
                    "num_inference_steps": 2,
                    "gradient_checkpointing": False,
                }
            },
            "nodes": [
                {
                    "name": "predicted_actions",
                    "call": "module:openpi_model",
                    "kwargs": {"obs": "$observation", "noise": "$noise"},
                }
            ],
            "outputs": ["$predicted_actions"],
            "return": "single",
        }
    },
}


def _build_via_registry(dtype: str = "float32"):
    """Build a tiny GraphModel via ``build_model(dict_config)`` with the registry."""
    # Ensure model_constructor is importable (it lives inside policy_constructor/)
    _policy_constructor = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "..")
    )
    if _policy_constructor not in sys.path:
        sys.path.insert(0, _policy_constructor)

    from model_constructor import build_model

    dummy_config, _train_config, PaliGemmaWithExpertModel = _get_dummy_config(dtype)

    with _dummy_patches(dummy_config, _train_config, PaliGemmaWithExpertModel):
        graph_model = build_model(_GRAPH_CFG)

    graph_model = graph_model.to(DEVICE)
    return graph_model


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def wrapper():
    """Module-scoped tiny wrapper on GPU (shared across read-only tests)."""
    return _create_wrapper()


@pytest.fixture(scope="module")
def graph_model():
    """Module-scoped GraphModel built via registry (shared across read-only tests)."""
    return _build_via_registry()


@pytest.fixture()
def dummy_obs():
    """Batched dummy observation dict (B=2) as numpy arrays."""
    B = 2
    return {
        "proprio": np.zeros((B, 24), dtype=np.float32),
        "head": np.random.randint(0, 255, (B, 3, 224, 224), dtype=np.uint8),
        "left": np.random.randint(0, 255, (B, 3, 224, 224), dtype=np.uint8),
        "right": np.random.randint(0, 255, (B, 3, 224, 224), dtype=np.uint8),
        "prompt": "pick up the object",
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_forward_returns_correct_shape(wrapper, dummy_obs):
    """forward() returns (B, action_horizon, action_dim) actions."""
    wrapper.eval()
    with torch.no_grad():
        actions = wrapper.forward(dummy_obs)

    assert actions.shape == (2, ACTION_HORIZON, ACTION_DIM_OUTPUT), (
        f"Expected (2, {ACTION_HORIZON}, {ACTION_DIM_OUTPUT}), got {actions.shape}"
    )
    assert actions.dtype == torch.float32
    assert torch.isfinite(actions).all(), "Actions contain non-finite values"


def test_forward_with_noise(wrapper, dummy_obs):
    """forward() accepts explicit noise tensor (B, action_horizon, model_action_dim)."""
    wrapper.eval()
    B = 2
    noise = np.random.randn(B, ACTION_HORIZON, ACTION_DIM_MODEL).astype(np.float32)

    with torch.no_grad():
        actions = wrapper.forward(dummy_obs, noise=noise)

    assert actions.shape == (B, ACTION_HORIZON, ACTION_DIM_OUTPUT)
    assert torch.isfinite(actions).all(), "Actions contain non-finite values"


def test_forward_with_2d_noise_expansion(wrapper):
    """forward() auto-expands 2D noise (action_horizon, model_action_dim) to batch dim."""
    wrapper.eval()
    obs_1 = {
        "proprio": np.zeros((1, 24), dtype=np.float32),
        "head": np.zeros((1, 3, 224, 224), dtype=np.uint8),
        "left": np.zeros((1, 3, 224, 224), dtype=np.uint8),
        "right": np.zeros((1, 3, 224, 224), dtype=np.uint8),
        "prompt": "pick up the object",
    }
    noise_2d = np.random.randn(ACTION_HORIZON, ACTION_DIM_MODEL).astype(np.float32)

    with torch.no_grad():
        actions = wrapper.forward(obs_1, noise=noise_2d)

    assert actions.shape == (1, ACTION_HORIZON, ACTION_DIM_OUTPUT), (
        f"Expected (1, {ACTION_HORIZON}, {ACTION_DIM_OUTPUT}), got {actions.shape}"
    )


def test_predict_numpy_interface(wrapper, dummy_obs):
    """predict() returns np.float32 array with correct shape."""
    wrapper.eval()
    actions = wrapper.predict(dummy_obs)

    assert isinstance(actions, np.ndarray)
    assert actions.dtype == np.float32
    assert actions.shape == (2, ACTION_HORIZON, ACTION_DIM_OUTPUT), (
        f"Expected (2, {ACTION_HORIZON}, {ACTION_DIM_OUTPUT}), got {actions.shape}"
    )


def test_compute_loss_shape(wrapper, dummy_obs):
    """compute_loss() returns per-element MSE loss (B, action_horizon, model_action_dim)."""
    wrapper.eval()
    B = 2
    # Provide 24-dim actions; PadStatesAndActions will pad to 32
    actions = torch.randn(B, ACTION_HORIZON, ACTION_DIM_OUTPUT, device=DEVICE)

    loss = wrapper.compute_loss(dummy_obs, actions)

    assert loss.shape == (B, ACTION_HORIZON, ACTION_DIM_MODEL), (
        f"Expected ({B}, {ACTION_HORIZON}, {ACTION_DIM_MODEL}), got {loss.shape}"
    )
    assert loss.dtype == torch.float32
    assert torch.isfinite(loss).all(), "Loss contains non-finite values"
    assert (loss >= 0).all(), "MSE loss should be non-negative"


def test_backward_pass_gradients_flow(dummy_obs):
    """Gradients flow through compute_loss → backward."""
    wrapper = _create_wrapper(dtype="float32")
    wrapper.train()

    B = 2
    actions = torch.randn(B, ACTION_HORIZON, ACTION_DIM_OUTPUT, device=DEVICE)

    loss = wrapper.compute_loss(dummy_obs, actions)
    loss.mean().backward()

    params_with_grad = [
        name
        for name, p in wrapper.model.named_parameters()
        if p.grad is not None and p.grad.abs().sum() > 0
    ]
    assert len(params_with_grad) > 0, "No parameters received gradients"


def test_forward_with_per_sample_prompts(wrapper, dummy_obs):
    """forward() handles a list of per-sample prompt strings."""
    wrapper.eval()
    obs_with_prompts = dict(dummy_obs)
    obs_with_prompts["prompt"] = ["pick up the red block", "move to the left"]

    with torch.no_grad():
        actions = wrapper.forward(obs_with_prompts)

    assert actions.shape == (2, ACTION_HORIZON, ACTION_DIM_OUTPUT), (
        f"Expected (2, {ACTION_HORIZON}, {ACTION_DIM_OUTPUT}), got {actions.shape}"
    )
    assert torch.isfinite(actions).all(), "Actions contain non-finite values"


# ---------------------------------------------------------------------------
# Registry / Factory tests
# ---------------------------------------------------------------------------


def test_registry_build_returns_graph_model(graph_model):
    """build_model() returns a GraphModel wrapping OpenPiBatchedWrapper."""
    from model_constructor.graph.model import GraphModel

    assert isinstance(graph_model, GraphModel)
    assert graph_model.inputs == ["observation", "noise"]
    assert "openpi_model" in graph_model.graph_modules
    inner = graph_model.graph_modules["openpi_model"]
    assert type(inner).__name__ == "OpenPiBatchedWrapper"


def test_registry_forward_returns_correct_shape(graph_model, dummy_obs):
    """GraphModel forward returns batched actions on GPU."""
    graph_model.eval()
    with torch.no_grad():
        actions = graph_model(observation=dummy_obs, noise=None)

    assert actions.shape == (2, ACTION_HORIZON, ACTION_DIM_OUTPUT), (
        f"Expected (2, {ACTION_HORIZON}, {ACTION_DIM_OUTPUT}), got {actions.shape}"
    )
    assert actions.dtype == torch.float32
    assert actions.device.type == "cuda"
    assert torch.isfinite(actions).all(), "Actions contain non-finite values"


# ---------------------------------------------------------------------------
# Standalone runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("OpenPiBatchedWrapper Dummy Model Test (GPU)")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("SKIP: No CUDA GPU available.")
        raise SystemExit(1)

    if not _has_transformers_replace:
        print("SKIP: transformers_replace not installed correctly.")
        raise SystemExit(1)

    print(f"Device: {DEVICE} ({torch.cuda.get_device_name()})")
    print()

    print("Creating tiny wrapper...", end=" ", flush=True)
    _wrapper = _create_wrapper()
    print(f"OK  (params: {sum(p.numel() for p in _wrapper.parameters()) / 1e6:.1f}M)")
    print()

    _obs = {
        "proprio": np.zeros((2, 24), dtype=np.float32),
        "head": np.random.randint(0, 255, (2, 3, 224, 224), dtype=np.uint8),
        "left": np.random.randint(0, 255, (2, 3, 224, 224), dtype=np.uint8),
        "right": np.random.randint(0, 255, (2, 3, 224, 224), dtype=np.uint8),
        "prompt": "pick up the object",
    }

    tests = [
        ("test_forward_returns_correct_shape", lambda: test_forward_returns_correct_shape(_wrapper, _obs)),
        ("test_forward_with_noise", lambda: test_forward_with_noise(_wrapper, _obs)),
        ("test_forward_with_2d_noise_expansion", lambda: test_forward_with_2d_noise_expansion(_wrapper)),
        ("test_predict_numpy_interface", lambda: test_predict_numpy_interface(_wrapper, _obs)),
        ("test_compute_loss_shape", lambda: test_compute_loss_shape(_wrapper, _obs)),
        ("test_backward_pass_gradients_flow", lambda: test_backward_pass_gradients_flow(_obs)),
        ("test_forward_with_per_sample_prompts", lambda: test_forward_with_per_sample_prompts(_wrapper, _obs)),
    ]

    for name, fn in tests:
        print(f"Running {name}...", end=" ", flush=True)
        try:
            fn()
            print("PASSED")
        except Exception as e:
            print(f"FAILED: {e}")
            raise

    print()
    print("--- Registry / Factory tests ---")
    print()

    print("Building via registry (build_model)...", end=" ", flush=True)
    _graph_model = _build_via_registry()
    print("OK")
    print()

    registry_tests = [
        ("test_registry_build_returns_graph_model", lambda: test_registry_build_returns_graph_model(_graph_model)),
        ("test_registry_forward_returns_correct_shape", lambda: test_registry_forward_returns_correct_shape(_graph_model, _obs)),
    ]

    for name, fn in registry_tests:
        print(f"Running {name}...", end=" ", flush=True)
        try:
            fn()
            print("PASSED")
        except Exception as e:
            print(f"FAILED: {e}")
            raise

    print()
    print("All tests passed!")
