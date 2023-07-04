import torch
from torch import nn


class RoPEAttnLayer(nn.Module):
    """
    A version of an Attention-Encoder Layer with Rotary Positional Encodings
    (RoPE) as described in [1].

    [1] https://arxiv.org/pdf/2104.09864.pdf - RoPE Paper
    [2] https://github.com/karpathy/nanoGPT/blob/master/model.py - Guidance for
            loop-free implementation of multi-head architecture.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        max_pos_enc_len: int,
        dropout: float = 0.1,
        bias: bool = False,
    ):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_head = d_model // n_heads
        self.inv_sqrt_d_head = 1.0 / torch.sqrt(torch.tensor(self.d_head))

        self.multi_head_in_projection = nn.Linear(d_model, 3 * d_model, bias=bias)
        self.multi_head_out_projection = nn.Linear(d_model, d_model, bias=bias)

        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.n_heads = n_heads
        self.d_model = d_model
        self._construct_rope_matrices(max_pos_enc_len)

    def _construct_rope_matrices(self, max_pos_enc_len):
        """Constructs rotary embedding matrices for additive version
        [1, p. 7, eq. (34)]. Configured for x beeing of shape
        (batch_size, seqlen, d_model).
        """
        assert self.d_head % 2 == 0
        # [t1, t1, t2, t2, t3, t3, ...]
        thetas = 1000 ** (
            -2.0 * torch.arange(1, self.d_head / 2 + 1) / self.d_head
        ).repeat_interleave(2)
        positions = torch.arange(1, max_pos_enc_len + 1).float()
        # [ [1t1, 1t1, 1t2, 1t2, ...],
        #   [2t1, 2t1, 2t2, 2t2, ...],
        #   [3t1, 3t1, 3t2, 3t2, ...],
        #   ...                       ]
        args = positions.reshape(-1, 1) @ thetas.reshape(1, -1)
        self.register_buffer("rope_sin", torch.sin(args))
        self.register_buffer("rope_cos", torch.cos(args))

    def _reorder_for_rope_sin(self, x):
        """Reorders the inputs according to [1, p. 7, eq. (34)] for the
        multiplication with the sinus-part of the RoPE. Configured for x beeing
        having d_head as last dimension, should be of shape
        (batch_size, n_heads, seqlen, d_head).
        """
        # [x1, x3, x5, ...]
        x_odd = x[..., ::2]
        # [x2, x4, x6, ...]
        x_even = x[..., 1::2]
        # [[-x2, x1], [-x4, x3], [-x6, x5], ...]
        x_stacked = torch.stack([-x_even, x_odd], dim=-1)
        # [-x2, x1, -x4, x3, ...]
        return x_stacked.flatten(start_dim=-2)

    def _apply_rope(self, x):
        """Applies RoPE the inputs according to [1, p. 7, eq. (34)].
        Configured for x being of shape (batch_size, n_heads, seqlen, d_head).
        """
        T = x.shape[2]
        x_sin = self._reorder_for_rope_sin(x)
        x_rope = x * self.rope_cos[:T, :] + x_sin * self.rope_sin[:T, :]
        return x_rope

    def forward(self, x):
        B, T, C = x.size()  # batch_size, seqlen, d_model

        # apply key, query, value projections
        q, k, v = self.multi_head_in_projection(x).split(self.d_model, dim=2)

        # separate heads (batch_size, n_heads, seqlen, d_head)
        k = k.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        q = q.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        # apply RoPE transformation [1, p. 7]
        q_rope = self._apply_rope(q)
        k_rope = self._apply_rope(k)

        # RoPE self attention:
        #   (batch_size, n_heads, seqlen, d_head) x
        #   (batch_size, n_heads, d_head, seqlen)
        #       -> (batch_size, n_heads, seqlen, seqlen)
        #  This is the place, where the rotations get "inserted" into the
        #  attention mechanism as presented in [1, p. 6, eq. 19]. I stick to
        #  the basic `exp` as non-negativities.
        att_numerator = torch.exp(
            (q_rope @ k_rope.transpose(-2, -1)) * self.inv_sqrt_d_head
        )
        att_denominator = torch.exp((q @ k.transpose(-2, -1)) * self.inv_sqrt_d_head)
        att_denominator = torch.sum(att_denominator, dim=-1, keepdim=True)
        att = att_numerator / att_denominator
        # (batch_size, n_heads, seqlen, seqlen) x
        #   (batch_size, n_heads, seqlen, d_head)
        # -> (batch_size, n_heads, seqlen, d_head)
        y = att @ v
        # re-assemble all head outputs side by side
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # output projection
        y = self.multi_head_out_projection(y)

        # skip-connection and regularization
        y = self.layer_norm(y + x)
        y = self.dropout(y)
        return y
