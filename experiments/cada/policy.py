from rl4co.models.nn.env_embeddings.dynamic import StaticEmbedding
from rl4co.models.zoo.am import AttentionModelPolicy
from rl4co.utils.pylogger import get_pylogger

from experiments.cada.encoder import CaDaEncoder
from experiments.cada.decoder import CaDaDecoder

import torch.nn as nn
from baseline.routefinder_v2.routefinder.routefinder.models.env_embeddings.mtvrp import MTVRPInitEmbeddingRouteFinder, MTVRPContextEmbeddingRouteFinder

from rl4co.utils.decoding import (
    DecodingStrategy,
    get_decoding_strategy,
    get_log_likelihood,
)
from rl4co.utils.ops import batchify, unbatchify

log = get_pylogger(__name__)


class CaDaPolicy(AttentionModelPolicy):
    """
    Main CaDA policy based on the Transformer Architecture.
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
        context_full_features_on: bool = False,
        **kwargs,
    ):

        encoder = CaDaEncoder(
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

        if context_embedding is None and (not context_full_features_on):
            context_embedding = MTVRPContextEmbeddingRouteFinder(embed_dim=embed_dim)

        # mtvrp does not use dynamic embedding (i.e. only modifies the query, not key or value)
        dynamic_embedding = StaticEmbedding()

        decoder = kwargs.get('decoder', None)

        if decoder is None:

            sdpa_fn = kwargs.get('sdpa_fn', None)
            sdpa_fn_decoder = kwargs.get('sdpa_fn_decoder', None)
            mask_inner = kwargs.get('mask_inner', True)
            out_bias_pointer_attn = kwargs.get('out_bias_pointer_attn', False)
            linear_bias_decoder = kwargs.get('linear_bias_decoder', False)
            check_nan = kwargs.get('check_nan', True)
            moe_kwargs = kwargs.get('moe_kwargs', {"encoder": None, "decoder": None})

            decoder = CaDaDecoder(
                embed_dim=embed_dim,
                num_heads=num_heads,
                env_name=env_name,
                context_embedding=context_embedding,
                dynamic_embedding=dynamic_embedding,
                sdpa_fn=sdpa_fn if sdpa_fn_decoder is None else sdpa_fn_decoder,
                mask_inner=mask_inner,
                out_bias_pointer_attn=out_bias_pointer_attn,
                linear_bias=linear_bias_decoder,
                use_graph_context=use_graph_context,
                check_nan=check_nan,
                moe_kwargs=moe_kwargs["decoder"],
            )

        super(CaDaPolicy, self).__init__(
            encoder=encoder,
            decoder=decoder,
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