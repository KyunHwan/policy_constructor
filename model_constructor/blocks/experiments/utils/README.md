# `utils/` — Shared helpers for experimental blocks

Two small functional helpers reused across the flow-matching policies. Neither is an `nn.Module`; neither is registered. Both are plain functions you can drop into any custom block.

## `get_sinusoidal_pos_encoding` — [`pos_encoding.py`](pos_encoding.py)

Builds a **fixed (non-learnable)** sinusoidal positional encoding matrix for transformer inputs.

```python
def get_sinusoidal_pos_encoding(seq_len: int, d_model: int, device) -> torch.Tensor
# returns shape: (1, seq_len, d_model)
```

The construction is the standard sin/cos encoding from "Attention Is All You Need" (Vaswani et al., 2017):

- Position indices `0, 1, ..., seq_len - 1` form a column vector.
- For each even dimension `2i`, the entry is `sin(pos / 10000^(2i / d_model))`.
- For each odd dimension `2i+1`, the entry is `cos(pos / 10000^(2i / d_model))`.

The result has a leading singleton batch dimension so it broadcasts cleanly when added to an input of shape `(B, seq_len, d_model)`. Typical usage:

```python
x = ... # (B, seq_len, d_model)
x = x + get_sinusoidal_pos_encoding(x.shape[1], d_model, x.device)
```

**Where it's used inside this repo:**

- `InformationEncoder` (in [`../templates/information_encoder.py`](../templates/information_encoder.py) ~line 203) adds positional encoding to the concatenated `[visual, proprio, semantic, action]` token sequence before passing it to the transformer.
- Similar usage in the `*_info_embedder.py` modules across the policies.

The function returns a tensor on the requested `device`; it's recomputed every call (cheap — these are small matrices). If you care about avoiding the small overhead, cache the result in a buffer with `self.register_buffer(...)` in your module.

---

## `get_time_embedding` — [`time_embedding.py`](time_embedding.py)

Sinusoidal embedding of **scalar time values** (one per batch element) for flow-matching policies.

```python
def get_time_embedding(timesteps: torch.Tensor, embedding_dim: int, max_positions: int = 10000) -> torch.Tensor
# input shape:  (B,) or (B, 1) of values typically in [0, 1] (flow matching) or step indices (diffusion)
# output shape: (B, embedding_dim)
```

Same family of sinusoidal features as the positional encoding, but adapted for scalar inputs instead of integer positions. The math is essentially:

- `freqs[i] = exp(-log(max_positions) * i / (half_dim - 1))` for `i = 0, ..., embedding_dim/2 - 1`.
- `emb[b, i] = sin(timesteps[b] * freqs[i])` for the first half of the output dim.
- `emb[b, half_dim + i] = cos(timesteps[b] * freqs[i])` for the second half.

`embedding_dim` must be **even** (the implementation asserts this).

**Where it's used:**

- Every flow-matching action decoder takes a scalar `time` input per batch element and embeds it via this function before feeding into the body. Examples:
  - [`../cfg_vqvae_flow_matching/action_decoder.py`](../cfg_vqvae_flow_matching/action_decoder.py) — uses `get_time_embedding(time, transformer_d_model)` followed by `self.time_mlp` to project into a usable conditioning signal.
  - The naive and VFP variants follow the same pattern.

Typical inside-a-module usage:

```python
def forward(self, time: torch.Tensor, noise: torch.Tensor, ...):
    t_emb = get_time_embedding(time, embedding_dim=self.d_model)   # (B, d_model)
    t_emb = self.time_mlp(t_emb)                                    # learnable projection
    # ... add t_emb (broadcast) into the noisy input stream ...
```

The flow-matching convention here: `time ∈ [0, 1]` denotes how far along the noise → data trajectory you are. At `time = 0` the input is pure noise; at `time = 1` it's the target distribution. The embedding turns that scalar into a high-dimensional vector that the body can condition on (mirroring how diffusion models embed denoising-step indices).

---

## Adding your own utilities here

This directory is a fine place to drop helpers shared between experimental blocks. Conventions:

- Keep them **functions, not `nn.Module`s**. If something needs parameters, it belongs in a block file (and probably a registry entry).
- Put a docstring with shapes. The two helpers above are useful precisely because their shape contracts are clear.
- Don't import from `blocks/` here — `utils/` should be a leaf of the dependency tree. Cross-policy reuse is the point.
- If you add a third helper, list it in this README so the next reader can find it.
