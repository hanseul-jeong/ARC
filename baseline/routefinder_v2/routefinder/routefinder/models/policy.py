from rl4co.models.nn.env_embeddings.dynamic import StaticEmbedding
from rl4co.models.zoo.am import AttentionModelPolicy
from rl4co.utils.pylogger import get_pylogger

from routefinder.models.encoder import RouteFinderEncoder
from routefinder.models.env_embeddings.mtvrp import (
    MTVRPContextEmbeddingRouteFinder,
    MTVRPInitEmbeddingRouteFinder,
)

from typing import Optional, Union
import numpy as np
import torch
import torch.nn.functional as F
from tensordict import TensorDict
from rl4co.envs import RL4COEnvBase

import torch.nn as nn
from routefinder.models.env_embeddings.mtvrp import MTVRPInitEmbeddingRouteFinder, MTVRPContextEmbeddingRouteFinder

from rl4co.utils.decoding import (
    DecodingStrategy,
    get_decoding_strategy,
    get_log_likelihood,
)
from rl4co.utils.ops import batchify, unbatchify

log = get_pylogger(__name__)


class RouteFinderPolicy(AttentionModelPolicy):
    """
    Main RouteFinder policy based on the Transformer Architecture.
    We use the base AttentionModelPolicy for decoding (i.e. masked attention + pointer network)
    and our new RouteFinderEncoder for the encoder.
    """

    def __init__(
        self,
        embed_dim: int = 128,
        num_encoder_layers: int = 6,
        num_heads: int = 8,
        normalization: str = "instance",
        feedforward_hidden: int = 512,
        parallel_gated_kwargs: dict = None,
        encoder_use_post_layers_norm: bool = False,
        encoder_use_prenorm: bool = False,
        env_name: str = "mtvrp",
        use_graph_context: bool = False,
        init_embedding: MTVRPInitEmbeddingRouteFinder = None,
        context_embedding: MTVRPContextEmbeddingRouteFinder = None,
        extra_encoder_kwargs: dict = {},
        **kwargs,
    ):

        encoder = RouteFinderEncoder(
            init_embedding=init_embedding,
            num_heads=num_heads,
            embed_dim=embed_dim,
            num_layers=num_encoder_layers,
            feedforward_hidden=feedforward_hidden,
            normalization=normalization,
            use_prenorm=encoder_use_prenorm,
            use_post_layers_norm=encoder_use_post_layers_norm,
            parallel_gated_kwargs=parallel_gated_kwargs,
            **extra_encoder_kwargs,
        )

        if context_embedding is None:
            context_embedding = MTVRPContextEmbeddingRouteFinder(embed_dim=embed_dim)

        # mtvrp does not use dynamic embedding (i.e. only modifies the query, not key or value)
        dynamic_embedding = StaticEmbedding()

        super(RouteFinderPolicy, self).__init__(
            encoder=encoder,
            embed_dim=embed_dim,
            num_heads=num_heads,
            normalization=normalization,
            feedforward_hidden=feedforward_hidden,
            env_name=env_name,
            use_graph_context=use_graph_context,
            context_embedding=context_embedding,
            dynamic_embedding=dynamic_embedding,
            **kwargs,
        )


    def get_emb_from_decoder(self, td, env, decode_type, decoding_kwargs):
        
        hidden, init_embeds = self.encoder(td)

        # Setup decoding strategy
        # we pop arguments that are not part of the decoding strategy
        decode_strategy: DecodingStrategy = get_decoding_strategy(
            decode_type,
            temperature=decoding_kwargs.pop("temperature", self.temperature),
            tanh_clipping=decoding_kwargs.pop("tanh_clipping", self.tanh_clipping),
            mask_logits=decoding_kwargs.pop("mask_logits", self.mask_logits),
            store_all_logp=decoding_kwargs.pop("store_all_logp", False),
            **decoding_kwargs,
        )

        # Pre-decoding hook: used for the initial step(s) of the decoding strategy
        td, env, num_starts = decode_strategy.pre_decoder_hook(td, env)

        # Additionally call a decoder hook if needed before main decoding
        td, env, hidden = self.decoder.pre_decoder_hook(td, env, hidden, num_starts)

        cached = hidden

        has_dyn_emb_multi_start = self.decoder.is_dynamic_embedding and num_starts > 1

        # Handle efficient multi-start decoding
        if has_dyn_emb_multi_start:
            # if num_starts > 0 and we have some dynamic embeddings, we need to reshape them to [B*S, ...]
            # since keys and values are not shared across starts (i.e. the episodes modify these embeddings at each step)
            cached = cached.batchify(num_starts=num_starts)

        elif num_starts > 1:
            td = unbatchify(td, num_starts)

        glimpse_q = self.decoder._compute_q(cached, td)

        return glimpse_q, cached
