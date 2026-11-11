# Mamba4Rec

Sequential recommendation with selective state space models
([Mamba4Rec](https://arxiv.org/abs/2403.03900), Liu et al., 2024). The model
swaps SASRec's causal self-attention for a Mamba block: item embedding →
dropout → LayerNorm → Mamba layers → the last-position representation scores
candidate items by dot product.

## Dependencies

The only dependency is `torch` (see `requirements.txt`). The Mamba block is
vendored as a pure-torch selective scan, so training and inference run on
CPU out of the box.

To use the fused CUDA kernel instead, install the optional
[`mamba-ssm`](https://github.com/state-spaces/mamba) package and set
`device="cuda"`; the model then falls back to the vendored scan whenever
`mamba-ssm` is unavailable.

Note that the vendored scan materializes the state for every position, so
its memory grows with `batch_size * max_len * embedding_dim * expand *
d_state`. Reduce `batch_size` or `max_len` for long histories on CPU.

## Attribution

Reimplemented in Cornac from the MIT-licensed reference implementation at
https://github.com/chengkai-liu/Mamba4Rec, adopting Cornac's padding and
session-iterator conventions (the padding index is `total_items`, not
RecBole's `padding_idx=0`).
