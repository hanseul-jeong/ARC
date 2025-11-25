
from typing import Tuple, Union, Callable, Any
from torch import Tensor
from torch.nn.modules import Module
from baseline.routefinder_v2.routefinder.routefinder.models.encoder import RouteFinderEncoder
from baseline.routefinder_v2.routefinder.routefinder.models.nn.transformer import Normalization
from baseline.routefinder_v2.routefinder.routefinder.envs.mtvrp.env import MTVRPEnv
from experiments.cada.nn.transformer import GlobalBlock
import torch.nn as nn
import torch


class ARCEncoderTest(RouteFinderEncoder):
    """
    Encoder for RouteFinder model based on the Transformer Architecture.
    Here we include additional embedding from raw to embedding space, as
    well as more modern architecture options compared to the usual Attention Models
    based on POMO (including multi-task VRP ones).
    """

    def __init__(
        self,
        init_embedding: nn.Module = None,
        num_heads: int = 8,
        embed_dim: int = 128,
        num_layers: int = 6,
        feedforward_hidden: int = 512,
        normalization: str = "instance",
        use_prenorm: bool = False,
        use_post_layers_norm: bool = False,
        parallel_gated_kwargs: dict = None,
        num_glayers: int = 3,
        **transformer_kwargs,
    ):
        super().__init__(
            init_embedding, 
            num_heads, 
            embed_dim, 
            num_layers, 
            feedforward_hidden, 
            normalization, 
            use_prenorm, 
            use_post_layers_norm, 
            parallel_gated_kwargs, 
            **transformer_kwargs
        )
        in_features = 6
        self.layernorm = Normalization(embed_dim, "layer")
        self.wpa = nn.Linear(in_features=in_features, out_features=embed_dim, bias=True)
        self.wpb = nn.Linear(in_features=embed_dim, out_features=embed_dim, bias=True)
        self.init_prompt = self._init_prompt_w_global

        self.prpt_layer = GlobalBranches(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    normalization=normalization,
                    use_prenorm=use_prenorm,
                    feedforward_hidden=feedforward_hidden,
                    parallel_gated_kwargs=parallel_gated_kwargs,
                    num_layers=num_glayers,
                    **transformer_kwargs,
            )

    def _init_prompt_w_global(self, td):
        (
            has_open,
            has_time_window,
            has_duration_limit,
            has_backhaul,
            backhaul_class,
        ) = MTVRPEnv.check_variants(td)
        # (B, 6)
        v = torch.cat([has_open[:,None], has_time_window[:,None], has_duration_limit[:,None], has_backhaul[:,None], td["open_route"].float(), td["distance_limit"]], dim=1).float()
        v = torch.nan_to_num(v, posinf=0.0)
        # (B, 1, H)
        h = self.wpb(self.layernorm(self.wpa(v).unsqueeze(1)))
        return h       

    def forward(
        self, td: Tensor, mask: Union[Tensor, None] = None
    ) -> Tuple[Tensor, Tensor]:

        # Transfer to embedding space
        init_h = self.init_embedding(td)  # [B, N, H]
        p = self.init_prompt(td)

        # Process embedding
        h = init_h
        for layer in self.layers:
            h = layer(h, mask)
        
        h_prpt, _ = self.prpt_layer(h, p, mask)
        h = h + h_prpt

        # https://github.com/meta-llama/llama/blob/8fac8befd776bc03242fe7bc2236cdb41b6c609c/llama/model.py#L493
        if self.post_layers_norm is not None:
            h = self.post_layers_norm(h)

        # Return latent representation
        return h, init_h  # [B, N, H]


class GlobalBranches(nn.Module):
    def __init__(self, 
                 embed_dim: int = 128, 
                 num_heads: int = 8, 
                 feedforward_hidden: int | None = None, 
                 normalization: str | None = "instance", 
                 use_prenorm: bool = False, 
                 bias: bool = True, 
                 sdpa_fn: Callable[..., Any] | None = None, 
                 moe_kwargs: dict | None = None, 
                 parallel_gated_kwargs: dict | None = None,
                 num_layers: int = 2  # Added parameter to control number of layers
        ):
        super(GlobalBranches, self).__init__()
        self.num_layers = num_layers
        
        # Create blocks based on num_layers
        self.blocks = nn.ModuleList()
        for i in range(num_layers):
            self.blocks.append(GlobalBlock(
                            embed_dim=embed_dim,
                            num_heads=num_heads,
                            normalization=normalization,
                            use_prenorm=use_prenorm,
                            feedforward_hidden=feedforward_hidden,
                            parallel_gated_kwargs=parallel_gated_kwargs,
                ))
        
        # Create linear layers for intermediate connections
        if num_layers > 1:
            self.linear_layers = nn.ModuleList()
            for i in range(num_layers - 1):
                self.linear_layers.append(nn.ModuleDict({
                    'wfa': nn.Linear(in_features=embed_dim, out_features=embed_dim, bias=True),
                    'wfb': nn.Linear(in_features=embed_dim, out_features=embed_dim, bias=True)
                }))
        
        self.layernorm2 = Normalization(embed_dim, "layer")

    def forward(self, h, p, mask=None):
        for i in range(self.num_layers):
            # Pass through the current block
            h, p = self.blocks[i](h, p, mask)
            
            # Apply linear layers after each block (except the last one)
            if i < self.num_layers - 1:
                h = h + self.linear_layers[i]['wfa'](h)
                p = p + self.linear_layers[i]['wfb'](p)
        
        return h, p
