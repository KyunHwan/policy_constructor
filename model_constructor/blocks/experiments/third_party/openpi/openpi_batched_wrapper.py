"""Batched-inference-compatible nn.Module wrapper for PI0Pytorch.

Accepts the same input fields as Pi05IgrisVlaAdapter but with a batch dimension.
Returns batched action predictions. Compatible with the model-component registration
pattern used by offline_trainer.py (YAML _type_: openpi_batched).
"""

from __future__ import annotations

import logging
import os
import pathlib
import sys
from typing import Any

import numpy as np
import torch
import torch.nn as nn

# Ensure openpi is importable and JAX doesn't compete for GPU
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

_THIRD_PARTY = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _THIRD_PARTY not in sys.path:
    sys.path.insert(0, _THIRD_PARTY)

import safetensors.torch

from openpi.models import model as _model
from openpi.models_pytorch import pi0_pytorch
from openpi.policies import igris_policy
from openpi.training import checkpoints as _checkpoints
from openpi.training import config as _train_config
import openpi.transforms as transforms

logger = logging.getLogger(__name__)

# inference_engine camera key -> openpi camera key
_CAM_MAP = {"head": "cam_head", "left": "cam_left", "right": "cam_right"}


def _get_norm_stats_dim(norm_stats, key: str) -> int | None:
    """Get dimensionality of a norm_stats entry from its mean shape."""
    if not norm_stats:
        return None
    stats = norm_stats.get(key)
    if stats is None:
        return None
    return int(np.asarray(stats.mean, dtype=np.float32).reshape(-1).shape[0])


def _stack_dicts(samples: list[dict]) -> dict:
    """Recursively stack a list of per-sample dicts into a batched dict."""
    result = {}
    for k in samples[0]:
        vals = [s[k] for s in samples]
        if isinstance(vals[0], dict):
            result[k] = _stack_dicts(vals)
        elif isinstance(vals[0], np.ndarray):
            result[k] = np.stack(vals, axis=0)
        elif isinstance(vals[0], (np.bool_, np.generic)):
            result[k] = np.array(vals)
        else:
            # Keep non-array values (e.g. strings) as lists
            result[k] = vals
    return result


def _to_torch(d: dict, device: torch.device) -> dict:
    """Recursively convert numpy arrays in a dict to torch tensors on device."""
    result = {}
    for k, v in d.items():
        if isinstance(v, dict):
            result[k] = _to_torch(v, device)
        elif isinstance(v, np.ndarray):
            result[k] = torch.from_numpy(np.ascontiguousarray(v)).to(device)
        else:
            result[k] = v
    return result


class OpenPiBatchedWrapper(nn.Module):
    """Batched wrapper around PI0Pytorch for use with offline_trainer.py.

    Mirrors the input/output contract of Pi05IgrisVlaAdapter but supports a
    batch dimension.  Registered as ``_type_: openpi_batched`` in YAML configs.

    Args:
        train_config_name: OpenPI config name (e.g. "pi05_igris").
        ckpt_dir: Path to checkpoint containing model.safetensors + assets/.
            If None, the model is initialised with random weights.
        default_prompt: Optional language instruction injected when no prompt
            is present in the observation.
        camera_names: Camera keys expected in the input obs dict.
        action_dim: Action dimensionality (clipped by IgrisOutputs).
        action_horizon: Number of predicted action timesteps.
        num_inference_steps: Denoising steps for ``sample_actions``.
        gradient_checkpointing: Enable gradient checkpointing on the model.
    """

    def __init__(
        self,
        train_config_name: str = "pi05_igris",
        ckpt_dir: str | None = None,
        default_prompt: str | None = None,
        camera_names: list[str] | None = None,
        action_dim: int = 24,
        action_horizon: int = 50,
        num_inference_steps: int = 10,
        gradient_checkpointing: bool = False,
    ):
        super().__init__()

        camera_names = camera_names or ["head", "left", "right"]
        self._camera_names = list(camera_names)
        self._num_inference_steps = num_inference_steps

        # ---- Load OpenPI train config ----
        train_config = _train_config.get_config(train_config_name)

        # ---- Build PI0Pytorch as a sub-module (exposes parameters) ----
        self.model = pi0_pytorch.PI0Pytorch(config=train_config.model)

        # ---- Optionally load pretrained weights ----
        if ckpt_dir is not None:
            ckpt_path = pathlib.Path(ckpt_dir)
            weight_path = str(ckpt_path / "model.safetensors")
            if os.path.exists(weight_path):
                logger.info("Loading PyTorch weights from %s", weight_path)
                safetensors.torch.load_model(self.model, weight_path)
                self.model.paligemma_with_expert.to_bfloat16_for_selected_params(
                    "bfloat16"
                )
            else:
                logger.warning(
                    "model.safetensors not found at %s – using random weights",
                    weight_path,
                )

        # ---- Norm stats discovery (mirrors Pi05IgrisVlaAdapter) ----
        data_config = train_config.data.create(
            train_config.assets_dirs, train_config.model
        )
        norm_stats = None
        if ckpt_dir is not None:
            ckpt_path = pathlib.Path(ckpt_dir)
            asset_id = data_config.asset_id or data_config.repo_id
            if asset_id is None:
                raise ValueError(
                    "Checkpoint asset_id is missing; cannot load norm stats."
                )
            assets_dir = ckpt_path / "assets"
            norm_stats_dir = None

            if assets_dir.exists():
                expected_dir = assets_dir / asset_id
                if (expected_dir / "norm_stats.json").exists():
                    norm_stats_dir = expected_dir
                else:
                    candidates = list(assets_dir.rglob("norm_stats.json"))
                    matching = [
                        p for p in candidates if asset_id in str(p.parent)
                    ]
                    if matching:
                        candidates = matching
                    if len(candidates) == 1:
                        norm_stats_dir = candidates[0].parent
                    elif len(candidates) > 1:
                        candidates.sort(
                            key=lambda p: p.stat().st_mtime, reverse=True
                        )
                        norm_stats_dir = candidates[0].parent
                        logger.warning(
                            "Multiple norm_stats found; using %s",
                            norm_stats_dir,
                        )

            if norm_stats_dir is None:
                raise ValueError(
                    f"norm_stats not found under {assets_dir} "
                    f"(asset_id={asset_id})"
                )
            norm_stats_asset = str(norm_stats_dir.relative_to(assets_dir))
            norm_stats = _checkpoints.load_norm_stats(
                assets_dir, norm_stats_asset
            )
            logger.info("Loaded norm_stats from %s", norm_stats_dir)

        # ---- Extract state / action dimensions ----
        stats_state_key = None
        state_dim = None
        if norm_stats:
            if "state" in norm_stats:
                stats_state_key = "state"
            else:
                cands = [k for k in norm_stats if k != "actions"]
                stats_state_key = cands[0] if cands else None
            state_dim = (
                _get_norm_stats_dim(norm_stats, stats_state_key)
                if stats_state_key
                else None
            )

        # ---- Build transform pipelines (reuses policy_config.py pattern) ----
        # NOTE: data_config.data_transforms.inputs already contains IgrisInputs
        # and data_config.model_transforms.inputs already contains
        # InjectDefaultPrompt, ResizeImages, TokenizePrompt, PadStatesAndActions.
        # We must NOT duplicate IgrisInputs — it is not idempotent (converts
        # data["images"] → data["image"] on first call; KeyError on second).
        # InjectDefaultPrompt IS idempotent (no-op if prompt already set), so
        # the user-provided default_prompt is injected after IgrisInputs.
        self._input_transform = transforms.compose(
            [
                *data_config.data_transforms.inputs,
                transforms.InjectDefaultPrompt(default_prompt),
                transforms.Normalize(
                    norm_stats,
                    use_quantiles=data_config.use_quantile_norm,
                ),
                *data_config.model_transforms.inputs,
            ]
        )

        # Filter norm_stats to only "actions" for the output transform.
        # sample_actions() only returns actions (not state), and Unnormalize
        # uses strict=True which requires all norm_stats keys in the data dict.
        output_norm_stats = (
            {k: v for k, v in norm_stats.items() if k == "actions"}
            if norm_stats is not None
            else None
        )

        self._output_transform = transforms.compose(
            [
                *data_config.model_transforms.outputs,
                transforms.Unnormalize(
                    output_norm_stats,                          # was: norm_stats
                    use_quantiles=data_config.use_quantile_norm,
                ),
                *data_config.data_transforms.outputs,
            ]
        )

        # ---- Expose metadata ----
        self.norm_stats = norm_stats
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_horizon = action_horizon
        self.stats_state_key = stats_state_key

        if gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        logger.info(
            "OpenPiBatchedWrapper: state_dim=%s, action_dim=%d, "
            "action_horizon=%d, num_inference_steps=%d",
            state_dim,
            action_dim,
            action_horizon,
            num_inference_steps,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def forward(
        self,
        obs: dict[str, Any],
        noise: torch.Tensor | np.ndarray | None = None,
    ) -> torch.Tensor:
        """Run batched inference and return predicted actions.

        Args:
            obs: Batched observation dict with the same keys as
                ``Pi05IgrisVlaAdapter.predict()``, each with leading batch dim:
                - ``'proprio'``: ``(B, state_dim)`` np.float32
                - ``'head'``: ``(B, 3, H, W)`` np.uint8
                - ``'left'``: ``(B, 3, H, W)`` np.uint8
                - ``'right'``: ``(B, 3, H, W)`` np.uint8
            noise: Optional noise tensor ``(B, action_horizon, action_dim)``
                or ``(action_horizon, action_dim)``.  If *None*, the model
                samples noise internally.

        Returns:
            ``torch.Tensor`` of shape ``(B, action_horizon, action_dim)``
            with denormalized actions.
        """
        device = next(self.model.parameters()).device
        observation, _ = self._prepare_observation(obs, device=device)

        # Prepare noise
        noise_tensor = None
        if noise is not None:
            if isinstance(noise, np.ndarray):
                noise_tensor = torch.from_numpy(noise).to(device)
            else:
                noise_tensor = noise.to(device)
            if noise_tensor.ndim == 2:
                noise_tensor = noise_tensor[None, ...]

        # Sample actions (batched)
        raw_actions = self.model.sample_actions(
            device,
            observation,
            noise=noise_tensor,
            num_steps=self._num_inference_steps,
        )

        # Apply per-sample output transforms (Unnormalize + IgrisOutputs)
        batch_size = raw_actions.shape[0]
        actions_np = raw_actions.detach().cpu().float().numpy()
        outputs = []
        for i in range(batch_size):
            sample_out = {"actions": actions_np[i]}
            sample_out = self._output_transform(sample_out)
            outputs.append(sample_out["actions"])

        return torch.from_numpy(np.stack(outputs, axis=0)).to(device)

    def compute_loss(
        self,
        obs: dict[str, Any],
        actions: torch.Tensor,
        noise: torch.Tensor | None = None,
        time: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute flow-matching MSE loss for training.

        Args:
            obs: Same batched observation dict as ``forward()``.
            actions: Ground-truth actions ``(B, action_horizon, action_dim)``,
                raw / unnormalized.
            noise: Optional noise ``(B, action_horizon, action_dim)``.
                If *None*, ``PI0Pytorch.forward()`` samples it internally.
            time: Optional diffusion timestep ``(B,)``.
                If *None*, ``PI0Pytorch.forward()`` samples it internally.

        Returns:
            Per-element MSE loss ``(B, action_horizon, action_dim)``.
        """
        device = next(self.model.parameters()).device

        # Convert actions to numpy for normalization via transform pipeline
        if isinstance(actions, torch.Tensor):
            actions_np = actions.detach().cpu().float().numpy()
        else:
            actions_np = np.asarray(actions, dtype=np.float32)

        observation, normalized_actions = self._prepare_observation(
            obs, device=device, actions_np=actions_np
        )

        return self.model.forward(
            observation,
            normalized_actions,
            noise=noise,
            time=time,
        )

    def predict(
        self,
        obs: dict[str, Any],
        noise: np.ndarray | None = None,
    ) -> np.ndarray:
        """Numpy interface matching ``Pi05IgrisVlaAdapter.predict()``.

        Args:
            obs: Batched observation dict (numpy arrays).
            noise: Optional ``(B, action_horizon, action_dim)`` or
                ``(action_horizon, action_dim)`` numpy noise array.

        Returns:
            ``np.float32`` array ``(B, action_horizon, action_dim)`` of
            denormalized actions.
        """
        with torch.inference_mode():
            actions = self.forward(obs, noise=noise)
        return actions.cpu().float().numpy()

    def warmup(self, batch_size: int = 1) -> None:
        """Trigger ``torch.compile`` warmup with a dummy forward pass."""
        device = next(self.model.parameters()).device
        dummy_obs = {
            "proprio": np.zeros(
                (batch_size, self.state_dim or 24), dtype=np.float32
            ),
        }
        for cam in self._camera_names:
            dummy_obs[cam] = np.zeros(
                (batch_size, 3, 224, 224), dtype=np.uint8
            )
        with torch.inference_mode():
            self.forward(dummy_obs)
        logger.info("Warmup complete (batch_size=%d)", batch_size)

    def freeze_all_model_params(self) -> None:
        """Freeze all parameters (model is already in eval after loading)."""
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.model.eval()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _prepare_observation(
        self,
        obs: dict[str, Any],
        *,
        device: torch.device,
        actions_np: np.ndarray | None = None,
    ) -> tuple[_model.Observation, torch.Tensor | None]:
        """Convert a batched Pi05-format obs dict to a batched Observation.

        Returns ``(observation, normalized_actions)`` where
        ``normalized_actions`` is *None* when ``actions_np`` is not provided.
        """
        # Determine batch size
        first_key = "proprio" if "proprio" in obs else self._camera_names[0]
        batch_val = obs[first_key]
        if isinstance(batch_val, torch.Tensor):
            batch_size = batch_val.shape[0]
        else:
            batch_size = np.asarray(batch_val).shape[0]

        transformed_samples = []
        for i in range(batch_size):
            # --- Extract single sample as numpy ---
            # State
            state_i = obs["proprio"][i]
            if isinstance(state_i, torch.Tensor):
                state_i = state_i.cpu().float().numpy()
            state_i = np.asarray(state_i, dtype=np.float32).squeeze()

            # Images: CHW -> HWC, remap camera names
            images = {}
            for cam_name in self._camera_names:
                img = obs[cam_name][i]
                if isinstance(img, torch.Tensor):
                    img = img.cpu().numpy()
                img = np.asarray(img).squeeze()
                # CHW -> HWC
                if img.ndim == 3 and img.shape[0] in (1, 3):
                    img = np.transpose(img, (1, 2, 0))
                openpi_cam = _CAM_MAP.get(cam_name, cam_name)
                images[openpi_cam] = img

            sample: dict[str, Any] = {"images": images, "state": state_i}

            # Prompt (if available)
            if "prompt" in obs:
                prompt = obs["prompt"]
                if isinstance(prompt, list):
                    sample["prompt"] = prompt[i]
                else:
                    sample["prompt"] = prompt

            # Actions (for training – included so Normalize can process them)
            if actions_np is not None:
                sample["actions"] = actions_np[i]

            # Apply per-sample transform pipeline
            transformed = self._input_transform(sample)
            transformed_samples.append(transformed)

        # Stack into batched arrays
        batched = _stack_dicts(transformed_samples)

        # Extract normalized actions before converting to torch
        normalized_actions_tensor = None
        if actions_np is not None and "actions" in batched:
            norm_actions = batched.pop("actions")
            normalized_actions_tensor = torch.from_numpy(
                np.ascontiguousarray(norm_actions)
            ).to(device)

        # Convert observation part to torch tensors
        batched_tensors = _to_torch(batched, device)

        # Build Observation dataclass
        observation = _model.Observation.from_dict(batched_tensors)

        return observation, normalized_actions_tensor
