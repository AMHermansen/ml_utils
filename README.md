# ml-utils

A PyTorch-based machine learning utilities library providing reusable neural network components, optimizers, learning rate schedulers, and data utilities. The library has a strong focus on **packed-sequence processing** (variable-length sequences without padding), making it well-suited for transformer architectures that work with datasets where sequence lengths vary significantly.

## Features

- **Neural network components** — MLP, embeddings, SwiGLU, attention, transformer encoder/decoder, PairFormer
- **Packed sequence utilities** — convert between padded and packed (flat) representations, manipulate `cu_seqlens`
- **Optimizers** — `AdamWS` (AdamW with selective weight decay) and `Muon` (orthogonal gradient optimizer)
- **LR Schedulers** — linear warmup + cosine decay, warmup to constant, cyclic with warmup
- **Data utilities** — `CumulativeSeqlengthCollator` (sequence packing collate function) and `SequenceBucketingSampler`
- **HDF5 utilities** — helpers for loading packed and fixed-size datasets from HDF5 files
- **PyTorch Lightning utilities** — `LightningConfig`, optimizer configurators, WandB callback
- **Snakemake utilities** — callbacks and CLI helpers for Snakemake-based workflows

## Installation

The library requires **Python 3.11+** and **PyTorch 2.8**.

### Using `uv` (recommended)

```bash
uv add git+https://github.com/AMHermansen/ml_utils
```

### Using `pip`

```bash
pip install git+https://github.com/AMHermansen/ml_utils
```

### Optional extras

| Extra | Description |
|-------|-------------|
| `flash-attn` | Enables the [FlashAttention](https://github.com/Dao-AILab/flash-attention) backend for attention layers |
| `lion` | Enables the [Lion optimizer](https://github.com/lucidrains/lion-pytorch) |

```bash
# Install with FlashAttention support (strongly recommended)
uv add "ml-utils[flash-attn] @ git+https://github.com/AMHermansen/ml_utils"
```

## Modules

### Neural network components (`ml_utils.components`)

#### Multi-layer perceptron (MLP)

```python
from ml_utils.components import MLP, MLPConfig, MLPBlockConfig, MLPContextConfig

# Simple 3-layer MLP: 64 → 128 → 128 → 32
mlp = MLP(
    in_features=64,
    out_features=32,
    config=MLPConfig(
        num_layers=3,
        hidden_features=128,
        block_config=MLPBlockConfig(
            activation="SiLU",
            norm="LayerNorm",
            dropout=0.1,
        ),
    ),
)
out = mlp(x)  # (..., 32)

# MLP with external context conditioning
config = MLPConfig(
    num_layers=3,
    context_config=MLPContextConfig(
        context_features=16,
        apply_on_input=True,
    ),
)
mlp = MLP(in_features=64, out_features=32, config=config)
out = mlp(x, context=ctx)  # ctx shape: (batch, 16)
```

#### Embeddings

```python
from ml_utils.components import FourierEmbedding, CosineEmbedding

# Random Fourier features
fourier = FourierEmbedding(num_frequencies=64)
embedded = fourier(coords)  # (..., 64)

# Cosine embedding with configurable frequency scheme
cosine = CosineEmbedding(
    out_dim=32,
    scheme="exponential",   # "linear", "exponential", or "power"
    min_value=0.0,
    max_value=1.0,
    do_sin=True,            # concatenate sine features as well → output dim = 32
)
embedded = cosine(coords)  # (..., 32)
```

#### SwiGLU activation

```python
from ml_utils.components import SwiGLU, SwiGLUMLP

# Standalone SwiGLU gate
gate = SwiGLU(mode="swish")   # "swish"/"silu", "mp" (magnitude-preserving), "gelu"
out = gate(tensor)  # last dim is halved

# Full SwiGLU MLP block (linear → SwiGLU → linear)
net = SwiGLUMLP(in_features=256, upscale_factor=2.0, mode="swish")
out = net(x)  # same shape as input
```

#### Self-attention for packed sequences

```python
from ml_utils.components import PackedSelfAttention
from ml_utils.components.attention import SelfAttentionConfig

attn = PackedSelfAttention(
    in_features=256,
    config=SelfAttentionConfig(
        nheads=8,
        qkv_bias=True,
        use_flash_attention=False,  # set True if flash-attn is installed
    ),
)

# x: (total_tokens, 256), cu_seqlens: (batch+1,)
out = attn(x, cu_seqlens=cu_seqlens, max_seqlen=max_len)
```

#### Transformer Encoder

```python
from ml_utils.components import (
    TransformerEncoder,
    TransformerEncoderConfig,
    TransformerEncoderBlockConfig,
)

encoder = TransformerEncoder(
    in_features=256,
    config=TransformerEncoderConfig(
        num_layers=6,
        num_registers=4,      # learnable register tokens (removed from output)
        num_class_tokens=1,   # class tokens prepended and kept in output
        transformer_config=TransformerEncoderBlockConfig(),
    ),
    context_dim=64,           # 0 to disable conditioning
)

# x: packed (total_tokens, 256), cu_seqlens: (batch+1,), context: (batch, 64)
x_out, cu_out, max_out = encoder(x, cu_seqlens, max_seqlen, context=context)
```

#### Residual wrappers

```python
from ml_utils.components import Residual, ResidualWithContext, ResidualConfig
# Standard pre-norm residual (wraps any BaseComponent)
wrapped = Residual(
    component,
    config=ResidualConfig(
        norm_name="rms",          # "layer", "rms", or None
        use_layer_scale=True,
        drop_path_rate=0.1,
    ),
)

# Adaptive-norm residual conditioned on a context vector (DiT-style)
wrapped = ResidualWithContext(component, context_dim=64)
out = wrapped(x, context=ctx)
```

#### PairFormer

PairFormer processes both a single (per-token) representation and a pair (token-token) representation jointly, as used in [AlphaFold 3](https://www.nature.com/articles/s41586-024-07487-w).

```python
from ml_utils.components.pairformer import PairFormer
from ml_utils.components.pairformer.utils import PairFormerConfig

pairformer = PairFormer(
    single_features=128,
    pair_features=64,
    config=PairFormerConfig(num_layers=4),
)

# single: (B, L, 128), pair: (B, L, L, 64)
single_out, pair_out = pairformer(single, pair, seq_lens=lengths)
```

---

### Packed sequence utilities (`ml_utils.torch_utils`)

A *packed* tensor stores variable-length sequences concatenated into a single flat tensor along with a `cu_seqlens` tensor (cumulative sequence lengths, shape `(batch+1,)`) that encodes where each sequence starts and ends.

```python
from ml_utils.torch_utils import (
    pack_tensor,
    pack_tensors,
    unpack_tensor,
    unpack_tensors,
    prepend_tokens_to_packed_tensor,
    remove_tokens_from_packed_tensor,
    get_masked_cu_seqlens,
    get_packed_mean_loss,
)

# Pack a padded tensor into a flat packed representation
# mask: (B, N) bool, tensor: (B, N, F)
cu_seqlens, packed = pack_tensor(mask, tensor)   # packed: (total_valid, F)

# Pack multiple tensors at once with the same mask
cu_seqlens, (packed_a, packed_b) = pack_tensors(mask, tensor_a, tensor_b)

# Unpack back to padded form
mask_out, unpacked = unpack_tensor(cu_seqlens, packed)

# Prepend learnable tokens to each sequence in the pack
packed_new, cu_new = prepend_tokens_to_packed_tensor(packed, cu_seqlens, tokens)

# Remove the first n_tokens from each sequence
packed_new, cu_new = remove_tokens_from_packed_tensor(packed, cu_seqlens, n_tokens=2)

# Update cu_seqlens after masking individual tokens
cu_new = get_masked_cu_seqlens(cu_seqlens, token_mask)

# Compute per-sequence mean loss from packed per-token losses
loss = get_packed_mean_loss(packed_losses, cu_seqlens)
```

---

### Optimizers (`ml_utils.torch_utils.optim`)

#### AdamWS

AdamW that only applies weight decay to matrices (2-D+ parameters), following the recipe from [nanoGPT](https://github.com/karpathy/nanoGPT). Parameters wrapped in `ParameterNoWeightDecay` also opt out of decay.

```python
from ml_utils.torch_utils.optim import AdamWS

optimizer = AdamWS(model.parameters(), lr=1e-3, weight_decay=1e-2)
```

#### Muon

[Muon](https://github.com/KellerJordan/Muon) (momentum + orthogonalization via Newton-Schulz) for 2-D parameters, with Adam for everything else. Recommended for the hidden layers of transformers.

```python
from ml_utils.torch_utils.optim import Muon, suitable_for_muon

muon_params = [p for p in model.parameters() if suitable_for_muon(p)]
adam_params  = [p for p in model.parameters() if not suitable_for_muon(p)]

optimizer = Muon([
    {"params": muon_params, "lr": 0.02, "momentum": 0.95, "use_muon": True},
    {"params": adam_params, "lr": 3e-4, "use_muon": False},
])
```

---

### LR schedulers (`ml_utils.torch_utils.optim`)

```python
from ml_utils.torch_utils.optim import (
    LinearWarmupCosineDecay,
    LinearWarmupRootDecay,
    WarmupToConstant,
    CyclicWithWarmup,
)

# Linear warmup → cosine decay
scheduler = LinearWarmupCosineDecay(
    optimizer,
    warmup_steps=1000,
    total_steps=50000,
    final_factor=0.05,
)

# Linear warmup → square root decay (Transformer-style)
scheduler = LinearWarmupRootDecay(optimizer, dim_model=512, warmup_steps=4000)

# Linear warmup → constant LR
scheduler = WarmupToConstant(optimizer, num_steps=500)

# Cyclic schedule with per-cycle warmup (wraps OneCycleLR, resets automatically)
scheduler = CyclicWithWarmup(optimizer, ...)
```

---

### Data utilities (`ml_utils.data_utils`)

#### CumulativeSeqlengthCollator

A collate function that performs *sequence packing* across individual samples, concatenating sequence tensors and building `cu_seqlens`.

```python
from ml_utils.data_utils import CumulativeSeqlengthCollator
from torch.utils.data import DataLoader

collator = CumulativeSeqlengthCollator(
    seqlen_variable_names=["features", "labels"],  # tensors to concatenate
    cu_seqlen_names="cu_seqlen",                   # key holding the sequence length
)

loader = DataLoader(dataset, batch_size=8, collate_fn=collator)
```

Each sample in the dataset must contain the `cu_seqlen` key with the length of its sequence. The collator concatenates the sequence tensors and produces a single `cu_seqlens` tensor for the entire batch.

#### SequenceBucketingSampler

Groups sequences of similar length into buckets to reduce padding waste when used together with a padded collator.

```python
from ml_utils.data_utils import SequenceBucketingSampler

sampler = SequenceBucketingSampler(
    lengths=sequence_lengths,   # 1-D tensor of sequence lengths
    batch_size=32,
    length_splits=[64, 128, 256],  # bucket boundaries
    shuffle=True,
    drop_exceeding=False,
)
loader = DataLoader(dataset, batch_sampler=sampler)
```

---

### HDF5 utilities (`ml_utils.h5_utils`)

```python
from ml_utils.h5_utils import load_packed_datasets, load_fixed_datasets
import h5py

with h5py.File("data.h5", "r") as f:
    # Load a chunk of packed (variable-length) sequences
    data = load_packed_datasets(
        idx=0,
        chunk_size=32,
        h5_file=f,
        target_dataset_name="features",
        cumulative_lengths=cumulative_lengths,
    )

    # Load a chunk of fixed-size sequences
    data = load_fixed_datasets(idx=0, chunk_size=32, h5_file=f, target_dataset_name="labels")
```

---

### PyTorch Lightning utilities (`ml_utils.lightning_utils`)

```python
from ml_utils.lightning_utils import (
    LightningConfig,
    configure_optimizer_standard,
    configure_muon_optimizer,
    WandBSaveConfigCallback,
)
import torch as th
from ml_utils.torch_utils.optim import LinearWarmupCosineDecay

config = LightningConfig(
    optimizer_class=th.optim.AdamW,
    optimizer_kwargs={"lr": 1e-3, "weight_decay": 1e-2},
    scheduler_class=LinearWarmupCosineDecay,
    scheduler_kwargs={"warmup_steps": 1000, "total_steps": 50000},
    scheduler_config={"interval": "step"},
)

class MyModel(LightningModule):
    def configure_optimizers(self):
        return configure_optimizer_standard(self, config)
```

For Muon-based training:

```python
class MyModel(LightningModule):
    def configure_optimizers(self):
        return configure_muon_optimizer(
            muon_parameter_candidates=self.transformer.parameters(),
            remaining_parameters=self.head.parameters(),
            lightning_config=config,
        )
```


## Development

The project uses [`uv`](https://github.com/astral-sh/uv) for dependency management.

```bash
# Install all dependencies including dev tools
uv sync --all-extras --group dev

# Run tests
uv run pytest tests/

# Run linter
uv run ruff check src/ tests/

# Type-check
uv run pyright src/
```

## License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

Portions of the code are adapted from:
- [mltools](https://github.com/mattcleigh/mltools) — Copyright (c) Matthew Leigh (MIT License)
- [Muon](https://github.com/KellerJordan/Muon) — Copyright (c) 2024 Keller Jordan (MIT License)