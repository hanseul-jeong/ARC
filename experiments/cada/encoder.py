
from typing import Tuple
from torch import Tensor
from torch.nn.modules import Module
from baseline.routefinder_v2.routefinder.routefinder.models.encoder import RouteFinderEncoder
from baseline.routefinder_v2.routefinder.routefinder.models.nn.transformer import Normalization
from baseline.routefinder_v2.routefinder.routefinder.envs.mtvrp.env import MTVRPEnv
from experiments.cada.nn.transformer import GlobalBlock, SparseBlock
import torch.nn as nn
import torch


class CaDaEncoder(RouteFinderEncoder):
    def __init__(
            self, 
            init_embedding: Module = None, 
            num_heads: int = 8, 
            embed_dim: int = 128, 
            num_layers: int = 6, 
            feedforward_hidden: int = 512, 
            normalization: str = "instance", 
            use_prenorm: bool = False, 
            use_post_layers_norm: bool = False, 
            parallel_gated_kwargs: dict = None, 
            **transformer_kwargs
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
        self.layernorm = Normalization(embed_dim, "layer")
        self.wpa = nn.Linear(in_features=5, out_features=embed_dim, bias=True)
        self.wpb = nn.Linear(in_features=embed_dim, out_features=embed_dim, bias=True)

        self.glayers = nn.Sequential(
            *(
                GlobalBlock(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    normalization=normalization,
                    use_prenorm=use_prenorm,
                    feedforward_hidden=feedforward_hidden,
                    parallel_gated_kwargs=parallel_gated_kwargs,
                    **transformer_kwargs,
                )
                for _ in range(num_layers)
            )
        )
        self.slayers = nn.Sequential(
            *(
                SparseBlock(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    normalization=normalization,
                    use_prenorm=use_prenorm,
                    feedforward_hidden=feedforward_hidden,
                    parallel_gated_kwargs=parallel_gated_kwargs,
                    **transformer_kwargs,
                )
                for _ in range(num_layers)
            )
        )

        self.gls = nn.Sequential(
            *(
                nn.Linear(in_features=embed_dim, out_features=embed_dim, bias=True)
                for _ in range(num_layers)
            )
        )
        self.sls = nn.Sequential(
            *(
                nn.Linear(in_features=embed_dim, out_features=embed_dim, bias=True)
                for _ in range(num_layers)
            )
        )

    def init_prompt(self, td):
        (
            has_open,
            has_time_window,
            has_duration_limit,
            has_backhaul,
            backhaul_class,
        ) = MTVRPEnv.check_variants(td)
        has_cap = torch.ones_like(has_open)
        # (B, 5)
        v = torch.cat([has_cap[:, None], has_open[:,None], has_time_window[:,None], has_duration_limit[:,None], has_backhaul[:,None]], dim=1).float()
        # (B, 1, H)
        h = self.wpb(self.layernorm(self.wpa(v).unsqueeze(1)))
        return h
    

    def forward(
            self, 
            td: Tensor, 
            mask: Tensor | None = None
        ) -> Tuple[Tensor, Tensor]:
        
        # Transfer to embedding space
        init_h = self.init_embedding(td)  # [B, N, H]
        prompt_p = self.init_prompt(td)  # [B, 1, H]
        
        h_g, h_s = init_h, init_h
        p = prompt_p
        for glayer, slayer, gl, sl in zip(self.glayers, self.slayers, self.gls, self.sls):
            h_tilda_g, p = glayer(h_g, p, mask)
            h_tilda_s = slayer(h_s, mask)
            h_g = h_tilda_g + sl(h_tilda_s)
            h_s = h_tilda_s + gl(h_tilda_g)

        # final result is out of global branch
        if self.post_layers_norm is not None:
            h = self.post_layers_norm(h_g)

        # Return latent representation
        return h, init_h  # [B, N, H]
