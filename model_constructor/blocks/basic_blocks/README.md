# `model_constructor/blocks/basic_blocks/`

Reusable `torch.nn.Module` building blocks used internally by experimental blocks and available for use in custom block implementations.

> **Note**: These classes are **not registered** in the default registry. They cannot be referenced directly via `_type_` in YAML configs without first registering them (see [Registration](#registration) below). They are primarily used as internal components by the experimental blocks in [`../experiments/`](../experiments/).

## Modules

### `MLP` — [`mlp.py`](mlp.py)

Multi-layer perceptron with configurable depth, activations, and dropout.

```python
class MLP(torch.nn.Module):
    def __init__(
        self,
        *,
        dims: list[int | None],         # Layer dimensions; None for first = LazyLinear
        activation: torch.nn.Module | None = None,   # Inter-layer activation (default: ReLU)
        dropout: float = 0.0,           # Dropout rate between layers
        bias: bool = True,              # Linear layer bias
        final_activation: torch.nn.Module | None = None,  # Activation after last layer
    ) -> None
```

**Behavior**:
- `dims` must have length >= 2 (e.g., `[None, 256, 10]`)
- First dimension `None` → uses `torch.nn.LazyLinear` (input size inferred on first forward pass)
- Intermediate dimensions must be integers
- Activation + dropout applied between layers (not after the last)
- `final_activation` optionally applied after the last layer

**Example usage in Python**:
```python
mlp = MLP(dims=[None, 256, 10], dropout=0.1)
x = torch.randn(4, 64)    # (batch=4, features=64)
y = mlp(x)                 # y.shape == torch.Size([4, 10])
```

---

### `ConvBnAct` — [`conv.py`](conv.py)

2D convolution + batch normalization + activation.

```python
class ConvBnAct(torch.nn.Module):
    def __init__(
        self,
        *,
        in_channels: int | None,        # None → LazyConv2d
        out_channels: int,
        kernel_size: int | tuple[int, int] = 3,
        stride: int | tuple[int, int] = 1,
        padding: int | tuple[int, int] = 0,
        dilation: int | tuple[int, int] = 1,
        groups: int = 1,
        bias: bool | None = None,       # None → auto (False when norm present)
        norm: torch.nn.Module | None = None,   # Default: BatchNorm2d
        act: torch.nn.Module | None = None,    # Default: ReLU(inplace=True)
    ) -> None
```

**Behavior**:
- `in_channels=None` → uses `torch.nn.LazyConv2d`
- Default norm is `BatchNorm2d(out_channels)`; pass a custom `nn.Module` or `nn.Identity()` to override
- Default activation is `ReLU(inplace=True)`; pass `nn.Identity()` to disable
- When `norm` is provided, `bias` defaults to `False` (standard practice)

**Example usage in Python**:
```python
conv = ConvBnAct(in_channels=3, out_channels=16, kernel_size=3, padding=1)
x = torch.randn(2, 3, 32, 32)   # (batch, C, H, W)
y = conv(x)                      # y.shape == torch.Size([2, 16, 32, 32])
```

---

### `ResidualBlock` — [`residual.py`](residual.py)

Two-layer residual block with optional projection shortcut. Uses `ConvBnAct` internally.

```python
class ResidualBlock(torch.nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: int | None = None,   # Default: same as in_channels
        stride: int = 1,
        act: torch.nn.Module | None = None,  # Default: ReLU(inplace=True)
    ) -> None
```

**Behavior**:
- Main path: two 3x3 `ConvBnAct` layers (second has `Identity` activation)
- Skip path: `Identity` if `in_channels == out_channels` and `stride == 1`, otherwise 1x1 `ConvBnAct` projection
- Output: `act(main(x) + skip(x))`

**Example usage in Python**:
```python
block = ResidualBlock(in_channels=64, out_channels=128, stride=2)
x = torch.randn(2, 64, 16, 16)   # (batch, C, H, W)
y = block(x)                      # y.shape == torch.Size([2, 128, 8, 8])
```

---

### `NonCausalTransformerEncoder` — [`transformer_encoder.py`](transformer_encoder.py)

Standard transformer encoder (non-causal self-attention).

```python
class NonCausalTransformerEncoder(torch.nn.Module):
    def __init__(
        self,
        d_model: int,           # Feature dimension
        nhead: int,             # Number of attention heads
        dim_feedforward: int,   # FFN hidden dimension
        dropout: float,         # Dropout rate
        activation: str,        # "relu" or "gelu"
        batch_first: bool,      # True → (batch, seq, feature)
        num_layers: int,        # Number of encoder layers
    ) -> None
```

**Behavior**:
- Wraps `torch.nn.TransformerEncoder` with `torch.nn.TransformerEncoderLayer`
- No causal masking (`is_causal=False`)
- Input shape depends on `batch_first`: `(batch, seq, d_model)` if `True`, else `(seq, batch, d_model)`

---

### `CausalTransformerEncoder` — [`transformer_encoder.py`](transformer_encoder.py)

Transformer encoder with optional causal (autoregressive) masking.

```python
class CausalTransformerEncoder(torch.nn.Module):
    def __init__(
        self,
        d_model: int,           # Feature dimension
        nhead: int,             # Number of attention heads
        dim_feedforward: int,   # FFN hidden dimension
        dropout: float,         # Dropout rate
        activation: str,        # "relu" or "gelu"
        batch_first: bool,      # True → (batch, seq, feature)
        is_causal: bool,        # Enable causal masking
        num_layers: int,        # Number of encoder layers
        num_tokens: int,        # Sequence length for causal mask buffer
    ) -> None
```

**Behavior**:
- Pre-computes a causal mask buffer of size `(num_tokens, num_tokens)` via `generate_square_subsequent_mask`
- When `is_causal=True`, applies the causal mask during self-attention

---

### `TransformerDecoder` — [`transformer_decoder.py`](transformer_decoder.py)

Transformer decoder with optional causal masking on the target sequence.

```python
class TransformerDecoder(torch.nn.Module):
    def __init__(
        self,
        d_model: int,           # Feature dimension
        nhead: int,             # Number of attention heads
        dim_feedforward: int,   # FFN hidden dimension
        dropout: float,         # Dropout rate
        activation: str,        # "relu" or "gelu"
        batch_first: bool,      # True → (batch, seq, feature)
        tgt_is_causal: bool,    # Apply causal mask to target self-attention
        num_layers: int,        # Number of decoder layers
        num_tokens: int,        # Target sequence length for causal mask buffer
    ) -> None

    def forward(self, tgt_input: torch.Tensor, memory_input: torch.Tensor) -> torch.Tensor
```

**Behavior**:
- `tgt_input`: target sequence (decoder self-attention)
- `memory_input`: encoder output (cross-attention)
- Pre-computes a causal mask buffer if `tgt_is_causal=True`

## Registration

These blocks are **not** registered by default. To use them in YAML configs, register them in [`../register.py`](../register.py):

```python
from .basic_blocks.mlp import MLP
from .basic_blocks.conv import ConvBnAct
from .basic_blocks.residual import ResidualBlock

registry.register_module("mlp", MLP, signature_policy="strict", tags=("blocks",))
registry.register_module("conv_bn_act", ConvBnAct, signature_policy="strict", tags=("blocks",))
registry.register_module("residual_block", ResidualBlock, signature_policy="strict", tags=("blocks",))
```

After registration, they become available in YAML:

```yaml
schema_version: 1
model:
  sequential:
    layers:
      - _type_: mlp
        dims: [null, 256, 10]
        dropout: 0.1
```

## How these connect to policy construction

The basic blocks are used as internal components by the experimental policy blocks:

- `TransformerEncoder` / `TransformerDecoder` → used by action decoders and info embedders across CFG-VQVAE, VFP, and naive flow matching policies
- `MLP` / `ConvBnAct` / `ResidualBlock` → available as composable primitives for building custom blocks
