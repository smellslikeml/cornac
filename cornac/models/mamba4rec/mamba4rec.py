# Copyright 2026 The Cornac Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Torch backend for Mamba4Rec.

Reimplementation of the Mamba4Rec architecture (Liu et al., 2024) following
the MIT-licensed reference implementation at
https://github.com/chengkai-liu/Mamba4Rec (``mamba4rec.py``), adapted to
Cornac's padding and scoring conventions.

The reference delegates its sequence mixer to the CUDA ``mamba_ssm`` kernel.
To keep the default install CPU-runnable with torch alone, this module
vendors a pure-torch Mamba-1 block whose selective scan is an explicit O(L)
python loop -- numerically equivalent, slower than the fused kernel. The
``mamba_ssm.Mamba`` fast-path is used only when that package imports *and*
the model is placed on a CUDA device.

Item ids are integers in ``[0, item_num)`` with ``item_num`` used as the
padding index (the Cornac convention, not RecBole's ``padding_idx=0``).
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class MambaBlock(nn.Module):
    """Selective state-space block (Mamba-1) with a pure-torch scan.

    Parameters
    ----------
    d_model: int
        Model (item embedding) dimension.
    d_state: int, optional, default: 16
        SSM state expansion factor ``N``.
    d_conv: int, optional, default: 4
        Local causal convolution width.
    expand: int, optional, default: 2
        Block expansion factor; the inner width is ``expand * d_model``.
    device: str, optional, default: 'cpu'
        Device the block will run on. When it starts with ``'cuda'``, the
        ``mamba_ssm.Mamba`` kernel is used if importable, otherwise the
        vendored pure-torch scan is built.
    """

    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, device="cpu"):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = expand * d_model
        self.dt_rank = math.ceil(d_model / 16)

        self.backend = None
        if str(device).startswith("cuda"):
            Mamba = self._import_cuda_mamba()
            if Mamba is not None and torch.cuda.is_available():
                self.backend = Mamba(
                    d_model=d_model,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                )
        if self.backend is not None:
            return

        self.in_proj = nn.Linear(d_model, 2 * self.d_inner, bias=False)
        # causal depthwise convolution: padding d_conv - 1 then truncate to L
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=True,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
        )
        self.x_proj = nn.Linear(
            self.d_inner, self.dt_rank + 2 * self.d_state, bias=False
        )
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)
        self.A_log = nn.Parameter(torch.empty(self.d_inner, d_state))
        self.D = nn.Parameter(torch.empty(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        self.reset_ssm_parameters()

    @staticmethod
    def _import_cuda_mamba():
        try:
            from mamba_ssm import Mamba
        except ImportError:
            return None
        return Mamba

    def reset_ssm_parameters(self):
        """(Re-)initialize the SSM-specific parameters.

        ``A_log``, ``D`` and the ``dt_proj`` bias follow the Mamba-1
        initialisation so that ``softplus(dt_bias)`` falls in
        ``[dt_min, dt_max]``. Called again after a model-wide weight init
        that would otherwise zero the ``dt_proj`` bias.
        """
        if self.backend is not None:
            return
        A = torch.arange(1, self.d_state + 1, dtype=torch.float).repeat(
            self.d_inner, 1
        )
        self.A_log.data.copy_(torch.log(A))
        self.D.data.fill_(1.0)

        dt_min, dt_max, dt_floor = 0.001, 0.1, 1e-4
        dt = torch.exp(
            torch.rand(self.d_inner) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_floor)
        # inverse of softplus
        self.dt_proj.bias.data.copy_(dt + torch.log(-torch.expm1(-dt)))

    def forward(self, input_tensor):
        """Map ``(B, L, d_model)`` to ``(B, L, d_model)``.

        The scan is causal by construction: position ``t`` only reads
        ``dA[:, t]``, ``dBx[:, t]`` and the recurrent state ``h`` carried
        from earlier positions.
        """
        if self.backend is not None:
            return self.backend(input_tensor)

        B, L, _ = input_tensor.shape
        x, z = self.in_proj(input_tensor).chunk(2, dim=-1)  # (B, L, d_inner)

        x = x.transpose(1, 2)  # (B, d_inner, L)
        x = self.conv1d(x)[:, :, :L]
        x = F.silu(x)
        x = x.transpose(1, 2)  # (B, L, d_inner)

        x_dbl = self.x_proj(x)  # (B, L, dt_rank + 2 * d_state)
        dt, Bm, Cm = torch.split(
            x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1
        )
        delta = F.softplus(self.dt_proj(dt))  # (B, L, d_inner)

        # ZOH discretization of A = -exp(A_log)
        A = -torch.exp(self.A_log)
        dA = torch.exp(delta.unsqueeze(-1) * A)  # (B, L, d_inner, N)
        dBx = delta.unsqueeze(-1) * Bm.unsqueeze(2) * x.unsqueeze(-1)

        # h_t = exp(delta_t A) h_{t-1} + delta_t B_t x_t ;  y_t = C_t h_t
        h = x.new_zeros(B, self.d_inner, self.d_state)
        ys = []
        for t in range(L):
            h = dA[:, t] * h + dBx[:, t]
            ys.append((h * Cm[:, t].unsqueeze(1)).sum(-1))
        y = torch.stack(ys, dim=1) + self.D * x  # + D x_t skip

        return self.out_proj(y * F.silu(z))


class FeedForward(nn.Module):
    """Position-wise feed-forward with a post-norm residual."""

    def __init__(self, d_model, inner_size, dropout=0.2):
        super(FeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, inner_size)
        self.activation = nn.GELU()
        self.w_2 = nn.Linear(inner_size, d_model)
        self.dropout = nn.Dropout(dropout)
        self.LayerNorm = nn.LayerNorm(d_model, eps=1e-12)

    def forward(self, input_tensor):
        hidden_states = self.dropout(self.activation(self.w_1(input_tensor)))
        hidden_states = self.dropout(self.w_2(hidden_states))
        return self.LayerNorm(hidden_states + input_tensor)


class MambaLayer(nn.Module):
    """One Mamba block followed by dropout, LayerNorm and a feed-forward.

    Replicates the reference quirk: the residual connection around the Mamba
    branch is added only when the model stacks more than one layer
    (``n_layers > 1``); a single-layer model normalizes the block output
    without a residual.
    """

    def __init__(
        self,
        d_model,
        d_state=32,
        d_conv=4,
        expand=2,
        dropout=0.2,
        n_layers=1,
        device="cpu",
    ):
        super(MambaLayer, self).__init__()
        self.residual = n_layers > 1
        self.mamba = MambaBlock(
            d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            device=device,
        )
        self.dropout = nn.Dropout(dropout)
        self.LayerNorm = nn.LayerNorm(d_model, eps=1e-12)
        self.ffn = FeedForward(d_model, d_model * 4, dropout=dropout)

    def forward(self, input_tensor):
        hidden_states = self.dropout(self.mamba(input_tensor))
        if self.residual:
            hidden_states = hidden_states + input_tensor
        return self.ffn(self.LayerNorm(hidden_states))


class Mamba4RecModel(nn.Module):
    """Mamba4Rec selective-SSM model (Liu et al., 2024).

    Operates on sequences of past item ids, returning the last-position
    representation. Item ids are integers in ``[0, item_num)`` with
    ``item_num`` being used as the padding index.

    The model produces a ``(B, B+N)`` score matrix when called as
    ``forward(_, hist_iids, out_iids, return_hidden=False)`` where
    ``out_iids`` contains the ``B`` in-batch positives followed by ``N``
    shared negatives, matching the contract expected by the loss functions
    in :mod:`cornac.models.seq_utils`.
    """

    def __init__(
        self,
        item_num,
        embedding_dim=64,
        maxlen=200,
        n_layers=1,
        d_state=32,
        d_conv=4,
        expand=2,
        dropout=0.2,
        pad_idx=-1,
        init_std=0.02,
        device="cpu",
    ):
        super(Mamba4RecModel, self).__init__()
        self.item_num = item_num
        self.pad_idx = pad_idx if pad_idx >= 0 else item_num
        self.maxlen = maxlen
        self.dev = device
        self.init_std = init_std

        # +1 row for the padding entry at pad_idx
        self.item_emb = nn.Embedding(
            self.item_num + 1, embedding_dim, padding_idx=self.pad_idx
        )
        self.LayerNorm = nn.LayerNorm(embedding_dim, eps=1e-12)
        self.dropout = nn.Dropout(p=dropout)
        self.mamba_layers = nn.ModuleList(
            [
                MambaLayer(
                    embedding_dim,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                    dropout=dropout,
                    n_layers=n_layers,
                    device=device,
                )
                for _ in range(n_layers)
            ]
        )

        self.apply(self._init_weights)
        for layer in self.mamba_layers:
            layer.mamba.reset_ssm_parameters()
        self.to(device)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.init_std)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
            if isinstance(module, nn.Embedding) and module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def _score_items(self, hidden, cand_items):
        return torch.mm(hidden, cand_items.T)

    def _encode(self, hist_iids):
        # hist_iids: (B, T), left-padded with pad_idx
        seqs = self.item_emb(hist_iids)
        seqs = self.LayerNorm(self.dropout(seqs))
        for layer in self.mamba_layers:
            seqs = layer(seqs)

        pad_mask = hist_iids == self.pad_idx  # (B, T)
        seqs = seqs.masked_fill(pad_mask.unsqueeze(-1), 0.0)
        lengths = (~pad_mask).sum(dim=-1)
        last_idx = lengths.clamp(min=1) - 1
        return seqs[torch.arange(seqs.shape[0], device=seqs.device), last_idx]

    def forward(self, user_ids, hist_iids, out_iids, return_hidden=False):
        hidden = self._encode(hist_iids)
        item_emb = self.item_emb(out_iids)
        if return_hidden:
            return hidden, item_emb
        return self._score_items(hidden, item_emb)

    @torch.no_grad()
    def predict(self, user_ids, log_seqs, item_indices=None):
        """Score all real items for a single padded sequence.

        Returns a 1-D numpy array of size ``item_num`` (padding column is
        already stripped).
        """
        if item_indices is None:
            item_indices = torch.arange(self.item_num, device=self.dev)
        else:
            item_indices = torch.as_tensor(
                item_indices, dtype=torch.long, device=self.dev
            )
        if not isinstance(log_seqs, torch.Tensor):
            log_seqs = torch.as_tensor(log_seqs, dtype=torch.long, device=self.dev)
        hidden = self._encode(log_seqs)
        scores = self._score_items(hidden, self.item_emb(item_indices))
        return scores.squeeze().detach().cpu().numpy()
