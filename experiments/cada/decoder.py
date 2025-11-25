from rl4co.models.zoo.am.decoder import AttentionModelDecoder, PrecomputedCache
from tensordict import TensorDict
from torch.nn.modules import Module

from torch import Tensor
import torch

class CaDaDecoder(AttentionModelDecoder):
    def __init__(
            self, 
            embed_dim: int = 128, 
            num_heads: int = 8, 
            env_name: str = "tsp", 
            context_embedding: Module = None, 
            dynamic_embedding: Module = None, 
            mask_inner: bool = True, 
            out_bias_pointer_attn: bool = False, 
            linear_bias: bool = False, 
            use_graph_context: bool = True, 
            check_nan: bool = True, 
            sdpa_fn: callable = None, 
            pointer: Module = None, 
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
        
    def _compute_kvl(self, cached: PrecomputedCache, td: TensorDict):
        glimpse_k_stat, glimpse_v_stat, logit_k_stat = (
            cached.glimpse_key,
            cached.glimpse_val,
            cached.logit_key,
        )
        # Compute dynamic embeddings and add to static embeddings
        glimpse_k_dyn, glimpse_v_dyn, logit_k_dyn = self.dynamic_embedding(td)
        glimpse_k = glimpse_k_stat + glimpse_k_dyn
        glimpse_v = glimpse_v_stat + glimpse_v_dyn
        logit_k = logit_k_stat + logit_k_dyn

        # glimpse_k, glimpse_v = self._get_only_feasible(glimpse_k, glimpse_v, td)

        return glimpse_k, glimpse_v, logit_k

    
