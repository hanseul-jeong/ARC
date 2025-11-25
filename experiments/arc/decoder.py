from rl4co.models.zoo.am.decoder import AttentionModelDecoder, PrecomputedCache

from typing import Tuple
from tensordict import TensorDict
import torch.nn.functional as F

from rl4co.envs import RL4COEnvBase
from rl4co.utils.ops import batchify, unbatchify

import torch.nn as nn
import torch

class ARCDecoder(AttentionModelDecoder):
    def __init__(
        self, 
        embed_dim: int = 128, 
        num_heads: int = 8, 
        env_name: str = "tsp", 
        context_embedding: nn.Module = None, 
        dynamic_embedding: nn.Module = None, 
        mask_inner: bool = True, 
        out_bias_pointer_attn: bool = False, 
        linear_bias: bool = False, 
        use_graph_context: bool = True, 
        check_nan: bool = True, 
        sdpa_fn: callable = None, 
        pointer: nn.Module = None, 
        moe_kwargs: dict = None
    ):
        super().__init__(
            embed_dim, 
            num_heads, 
            env_name, 
            context_embedding, 
            dynamic_embedding, 
            mask_inner, 
            out_bias_pointer_attn, 
            linear_bias, 
            use_graph_context, 
            check_nan, 
            sdpa_fn, 
            pointer, 
            moe_kwargs
        )
    
    def pre_decoder_hook(
        self, td, env, embeddings, num_starts: int = 0
    ) -> Tuple[TensorDict, RL4COEnvBase, PrecomputedCache]:

        return td, env, self._precompute_cache(embeddings, num_starts=num_starts, prompt=None)
    
    def _precompute_cache(
        self, embeddings: torch.Tensor, num_starts: int = 0, prompt: torch.Tensor = None
    ) -> PrecomputedCache:
        """Compute the cached embeddings for the pointer attention.

        Args:
            embeddings: Precomputed embeddings for the nodes
            num_starts: Number of starts for the multi-start decoding
        """
        # The projection of the node embeddings for the attention is calculated once up front
        (
            glimpse_key_fixed,
            glimpse_val_fixed,
            logit_key_fixed,
        ) = self.project_node_embeddings(embeddings).chunk(3, dim=-1)

        graph_context = 0

        # Organize in a dataclass for easy access
        return PrecomputedCache(
            node_embeddings=embeddings,
            graph_context=graph_context,
            glimpse_key=glimpse_key_fixed,
            glimpse_val=glimpse_val_fixed,
            logit_key=logit_key_fixed,
        )

    