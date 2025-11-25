from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from rl4co.models.nn.attention import MultiHeadAttention
from baseline.routefinder_v2.routefinder.routefinder.models.nn.transformer import Normalization, ParallelGatedMLP, TransformerBlock
from rl4co.models.nn.mlp import MLP
from rl4co.models.nn.moe import MoE
from rl4co.models.nn.attention import scaled_dot_product_attention
from rl4co.utils.pylogger import get_pylogger
from torch import Tensor

from einops import rearrange

from typing import Any

log = get_pylogger(__name__)
   

class GlobalBlock(nn.Module):
    def __init__(
        self,
        embed_dim: int = 128,
        num_heads: int = 8,
        feedforward_hidden: Optional[int] = None,  # if None, use 4 * embed_dim
        normalization: Optional[str] = "instance",
        use_prenorm: bool = False,
        bias: bool = True,
        sdpa_fn: Optional[Callable] = None,
        moe_kwargs: Optional[dict] = None,
        parallel_gated_kwargs: Optional[dict] = None,
    ):
        super(GlobalBlock, self).__init__()
        feedforward_hidden = (
            4 * embed_dim if feedforward_hidden is None else feedforward_hidden
        )

        self.norm_attn_h = (
            Normalization(embed_dim, "rms")
            if normalization is not None
            else lambda x: x
        )
        self.norm_attn_p = (
            Normalization(embed_dim, "rms")
            if normalization is not None
            else lambda x: x
        )
        self.swiglu_h = ParallelGatedMLP(embed_dim, mlp_activation="silu")
        self.swiglu_p = ParallelGatedMLP(embed_dim, mlp_activation="silu")
        self.attention_h = CustomMultiHeadAttention(
            embed_dim, num_heads, bias=bias, sdpa_fn=None
        )
        self.attention_p = CustomMultiHeadAttention(
            embed_dim, num_heads, bias=bias, sdpa_fn=None
        )
        self.norm_ffn_h = (
            Normalization(embed_dim, normalization)
            if normalization is not None
            else lambda x: x
        )
        self.norm_ffn_p = (
            Normalization(embed_dim, normalization)
            if normalization is not None
            else lambda x: x
        )

    def forward(self, x: Tensor, p: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        h_hat = self.norm_attn_h(x + self.attention_h(x, torch.cat([x, p], dim=1), mask))
        h_tilda = self.norm_ffn_h(h_hat + self.swiglu_h(h_hat))

        p_hat = self.norm_attn_p(p + self.attention_p(p, torch.cat([x, p], dim=1), mask))
        p_i = self.norm_ffn_p(p_hat + self.swiglu_p(p_hat))

        return h_tilda, p_i
    
    
class SparseBlock(nn.Module):
    def __init__(
        self,
        embed_dim: int = 128,
        num_heads: int = 8,
        feedforward_hidden: Optional[int] = None,  # if None, use 4 * embed_dim
        normalization: Optional[str] = "instance",
        use_prenorm: bool = False,
        bias: bool = True,
        sdpa_fn: Optional[Callable] = None,
        moe_kwargs: Optional[dict] = None,
        parallel_gated_kwargs: Optional[dict] = None,
    ):
        super(SparseBlock, self).__init__()
        sdpa_fn = scaled_dot_product_attention_sparse
        feedforward_hidden = (
            4 * embed_dim if feedforward_hidden is None else feedforward_hidden
        )

        self.norm_attn_h = (
            Normalization(embed_dim, "rms")
            if normalization is not None
            else lambda x: x
        )
        self.swiglu_h = ParallelGatedMLP(embed_dim, mlp_activation="silu")
        self.attention_h = CustomMultiHeadAttention(
            embed_dim, num_heads, bias=bias, sdpa_fn=sdpa_fn
        )
        self.norm_ffn_h = (
            Normalization(embed_dim, normalization)
            if normalization is not None
            else lambda x: x
        )

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:

        h_hat = self.norm_attn_h(x + self.attention_h(x, x, mask))
        h_tilda = self.norm_ffn_h(h_hat + self.swiglu_h(h_hat))

        return h_tilda
    

class CustomMultiHeadAttention(MultiHeadAttention):
    def __init__(self, 
                 embed_dim: int, 
                 num_heads: int, 
                 bias: bool = True, 
                 attention_dropout: float = 0, 
                 causal: bool = False, 
                 device: str = None, 
                 dtype: torch.dtype = None, 
                 sdpa_fn: Callable[..., Any] | None = None
                 ) -> None:
        super().__init__(
            embed_dim, 
            num_heads, 
            bias, 
            attention_dropout, 
            causal, 
            device, 
            dtype, 
            sdpa_fn
            )
        factory_kwargs = {"device": device, "dtype": dtype}
        self.Wq = nn.Linear(embed_dim, embed_dim, bias=bias, **factory_kwargs)
        self.Wkv = nn.Linear(embed_dim, 2 * embed_dim, bias=bias, **factory_kwargs)

    def forward(self, x, y, attn_mask=None):
        # project q, k, v
        q = rearrange(
            self.Wq(x), "b s (one h d) -> one b h s d", one=1, h=self.num_heads
        )[0]
        k, v = rearrange(
            self.Wkv(y), "b s (two h d) -> two b h s d", two=2, h=self.num_heads
        ).unbind(dim=0)

        if attn_mask is not None:
            attn_mask = (
                attn_mask.unsqueeze(1)
                if attn_mask.ndim == 3
                else attn_mask.unsqueeze(1).unsqueeze(2)
            )

        # Scaled dot product attention
        out = self.sdpa_fn(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=self.attention_dropout,
        )
        return self.out_proj(rearrange(out, "b h s d -> b s (h d)"))
    
def scaled_dot_product_attention_sparse(
    q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False,
):
    """Simple (exact) Scaled Dot-Product Attention in RL4CO without customized kernels (i.e. no Flash Attention)."""

    # Check for causal and attn_mask conflict
    if is_causal and attn_mask is not None:
        raise ValueError("Cannot set both is_causal and attn_mask")

    # Calculate scaled dot product
    scores = torch.matmul(q, k.transpose(-2, -1)) / (k.size(-1) ** 0.5)

    # Apply the provided attention mask
    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            scores.masked_fill_(~attn_mask, float("-inf"))
        else:
            scores += attn_mask

    # Apply causal mask
    if is_causal:
        s, l_ = scores.size(-2), scores.size(-1)
        mask = torch.triu(torch.ones((s, l_), device=scores.device), diagonal=1)
        scores.masked_fill_(mask.bool(), float("-inf"))

    # Softmax to get attention weights
    attn_weights_ = F.softmax(scores, dim=-1)
    
    top_k_mask = get_top_k_mask(attn_weights_)
    masked_weight = attn_weights_.masked_fill(~top_k_mask.bool(), 0.0)
    attn_weights = F.softmax(masked_weight, dim=-1)

    # Apply dropout
    if dropout_p > 0.0:
        attn_weights = F.dropout(attn_weights, p=dropout_p)

    # Compute the weighted sum of values
    return torch.matmul(attn_weights, v)


def get_top_k_mask(scores):
    batch_size, head_size, n1, n2 = scores.size()

    k = int(n1/2)
    
    # top k
    _, top_k_indices = torch.topk(scores, k=k, dim=3)
    
    # empty mask
    mask = torch.zeros_like(scores, dtype=torch.float)
    
    # make indices
    batch_indices = torch.arange(batch_size).view(-1, 1, 1, 1).expand(-1, head_size, n1, k)
    head_indices = torch.arange(head_size).view(1, -1, 1, 1).expand(batch_size, -1, n1, k)
    row_indices = torch.arange(n1).view(1, 1, -1, 1).expand(batch_size, head_size, -1, k)
    
    # assign 1 for top k loc
    mask[batch_indices, head_indices, row_indices, top_k_indices] = 1.0
    
    return mask






class GlobalBlock_n(nn.Module):
    def __init__(
        self,
        embed_dim: int = 128,
        num_heads: int = 8,
        feedforward_hidden: Optional[int] = None,  # if None, use 4 * embed_dim
        normalization: Optional[str] = "instance",
        use_prenorm: bool = False,
        bias: bool = True,
        sdpa_fn: Optional[Callable] = None,
        moe_kwargs: Optional[dict] = None,
        parallel_gated_kwargs: Optional[dict] = None,
    ):
        super(GlobalBlock_n, self).__init__()
        feedforward_hidden = (
            4 * embed_dim if feedforward_hidden is None else feedforward_hidden
        )

        self.norm_attn_h = Normalization(embed_dim, "rms")
        self.norm_attn_p = Normalization(embed_dim, "rms")
        self.swiglu_h = ParallelGatedMLP(embed_dim, mlp_activation="silu")
        self.swiglu_p = ParallelGatedMLP(embed_dim, mlp_activation="silu")
        self.attention_h = CustomMultiHeadAttention(
            embed_dim, num_heads, bias=bias, sdpa_fn=None
        )
        self.attention_p = CustomMultiHeadAttention(
            embed_dim, num_heads, bias=bias, sdpa_fn=None
        )
        self.norm_ffn_h = Normalization(embed_dim, "rms")
        self.norm_ffn_p = Normalization(embed_dim, "rms")

    def forward(self, x: Tensor, p: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        h_hat = self.norm_attn_h(x + self.attention_h(x, torch.cat([x, p], dim=1), mask))
        h_tilda = self.norm_ffn_h(h_hat + self.swiglu_h(h_hat))

        p_hat = self.norm_attn_p(p + self.attention_p(p, torch.cat([x, p], dim=1), mask))
        p_i = self.norm_ffn_p(p_hat + self.swiglu_p(p_hat))

        return h_tilda, p_i
    


class SparseBlock_n(nn.Module):
    def __init__(
        self,
        embed_dim: int = 128,
        num_heads: int = 8,
        feedforward_hidden: Optional[int] = None,  # if None, use 4 * embed_dim
        normalization: Optional[str] = "instance",
        use_prenorm: bool = False,
        bias: bool = True,
        sdpa_fn: Optional[Callable] = None,
        moe_kwargs: Optional[dict] = None,
        parallel_gated_kwargs: Optional[dict] = None,
    ):
        super(SparseBlock_n, self).__init__()
        sdpa_fn = scaled_dot_product_attention_sparse
        feedforward_hidden = (
            4 * embed_dim if feedforward_hidden is None else feedforward_hidden
        )

        self.norm_attn_h = Normalization(embed_dim, "rms")
        self.swiglu_h = ParallelGatedMLP(embed_dim, mlp_activation="silu")
        self.attention_h = CustomMultiHeadAttention(
            embed_dim, num_heads, bias=bias, sdpa_fn=sdpa_fn
        )
        self.norm_ffn_h = Normalization(embed_dim, "rms")

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:

        h_hat = self.norm_attn_h(x + self.attention_h(x, x, mask))
        h_tilda = self.norm_ffn_h(h_hat + self.swiglu_h(h_hat))

        return h_tilda