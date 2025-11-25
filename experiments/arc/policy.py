from rl4co.models.zoo.am import AttentionModelPolicy

from experiments.arc.encoder_prpt import ARCEncoderTest

from baseline.routefinder_v2.routefinder.routefinder.models.env_embeddings.mtvrp import MTVRPInitEmbeddingRouteFinder, MTVRPContextEmbeddingRouteFinder

from rl4co.models.nn.env_embeddings.dynamic import StaticEmbedding
from baseline.routefinder_v2.routefinder.routefinder.models.env_embeddings.mtvrp import (
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

from experiments.arc.decoder import ARCDecoder

from rl4co.utils.decoding import (
    DecodingStrategy,
    get_decoding_strategy,
)
from rl4co.utils.ops import batchify, unbatchify



class ARCEmbPolicy(AttentionModelPolicy):
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
        num_glayers: int = 3,
        **kwargs,
    ):

        encoder = ARCEncoderTest(
            init_embedding=init_embedding,
            num_heads=num_heads,
            embed_dim=embed_dim,
            num_layers=num_encoder_layers,
            feedforward_hidden=feedforward_hidden,
            normalization=normalization,
            use_prenorm=encoder_use_prenorm,
            use_post_layers_norm=encoder_use_post_layers_norm,
            parallel_gated_kwargs=parallel_gated_kwargs,
            num_glayers=num_glayers,
            **extra_encoder_kwargs,
        )

        if context_embedding is None:
            context_embedding = MTVRPContextEmbeddingRouteFinder(embed_dim=embed_dim)

        # mtvrp does not use dynamic embedding (i.e. only modifies the query, not key or value)
        dynamic_embedding = StaticEmbedding()

        super(ARCEmbPolicy, self).__init__(
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


class ARCCompPolicy(ARCEmbPolicy):
    """
    The policy of ours based on RouteFinder
    We add task analogy regularization
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
        # ours
        scene: int = 4,
        nce_temp: float = 0.1,
        **kwargs,
    ):
        self.scene = scene  # case 1: attr-masking, case 2: coord. pair
        self.nce_temp = nce_temp
        
        dynamic_embedding = StaticEmbedding()
        decoder = kwargs.get('decoder', None)
        if decoder == 'ARCDecoder':
            sdpa_fn = kwargs.get('sdpa_fn', None)
            sdpa_fn_decoder = kwargs.get('sdpa_fn_decoder', None)
            mask_inner = kwargs.get('mask_inner', True)
            out_bias_pointer_attn = kwargs.get('out_bias_pointer_attn', False)
            linear_bias_decoder = kwargs.get('linear_bias_decoder', False)
            check_nan = kwargs.get('check_nan', True)
            moe_kwargs = kwargs.get('moe_kwargs', {"encoder": None, "decoder": None})

            decoder = ARCDecoder(
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
            kwargs['decoder'] = decoder

        super().__init__(
            embed_dim=embed_dim, 
            num_encoder_layers=num_encoder_layers, 
            num_heads=num_heads, 
            normalization=normalization, 
            feedforward_hidden=feedforward_hidden, 
            parallel_gated_kwargs=parallel_gated_kwargs,
            encoder_use_post_layers_norm=encoder_use_post_layers_norm,
            encoder_use_prenorm=encoder_use_prenorm,
            env_name=env_name, 
            use_graph_context= use_graph_context, 
            init_embedding=init_embedding, 
            context_embedding=context_embedding, 
            extra_encoder_kwargs=extra_encoder_kwargs,
            **kwargs
        )

        self.nn = nn.Linear(128*4,128, True)
        self.nn1 = nn.Linear(128*5,128, True)
        self.nn2 = nn.Linear(128,128, True)
        self.nn3 = nn.Linear(128*4,128, True)
        self.nn4 = nn.Linear(128*1,128, True)
        self.nn5 = nn.Linear(128*2,128, True)
        self.nn6 = nn.Linear(128,128, True)
        
    
    def forward(
        self, 
        td: TensorDict, 
        env: Optional[Union[str, RL4COEnvBase]] = None, 
        phase: str = "train", 
        calc_reward: bool = True, 
        return_actions: bool = True, 
        return_entropy: bool = False, 
        return_hidden: bool = False, 
        return_init_embeds: bool = False, 
        return_sum_log_likelihood: bool = True, 
        actions=None, max_steps=1000000, 
        **decoding_kwargs
    ) -> dict:
        out = super().forward(
            td, 
            env, 
            phase, 
            calc_reward, 
            return_actions, 
            return_entropy, 
            return_hidden, 
            return_init_embeds, 
            return_sum_log_likelihood, 
            actions, 
            max_steps, 
            **decoding_kwargs
        )
        self.decoding_kwargs = decoding_kwargs
        
        if phase != 'train':
            return out

        loss_comp = 0
       
        if self.scene == 0:
            pass
        elif self.scene in [112]:
            td_variants = np.array(env.get_variant_names(td))
            binary_vectors = self.get_keep_mask(env, td)
            om, twm, lm, bm = self.create_single_feature_masks(binary_vectors)
            
            td_o_rm = self.mask_col_features(env, td.clone(), om, 0)
            td_tw_rm = self.mask_col_features(env, td.clone(), twm, 1)
            td_l_rm = self.mask_col_features(env, td.clone(), lm, 2)
            td_b_rm = self.mask_col_features(env, td.clone(), bm, 3)

            # concat all tensors
            combined_td = torch.cat([td, td_o_rm, td_tw_rm, td_l_rm, td_b_rm], dim=0)

            batch_size = td.size(0)

            # calculate all embeddings
            combined_embeds, _ = self.get_embeds(combined_td, env, phase=phase, use_encoder=True)

            # divide embeddings
            embeds_xi_a = combined_embeds[:batch_size]
            embeds_xi_o_rm = combined_embeds[batch_size:batch_size*2]
            embeds_xi_tw_rm = combined_embeds[batch_size*2:batch_size*3]
            embeds_xi_l_rm = combined_embeds[batch_size*3:batch_size*4]
            embeds_xi_b_rm = combined_embeds[batch_size*4:]

            # calculate attribute vectors
            features_o = (embeds_xi_a - embeds_xi_o_rm)[om]
            features_tw = (embeds_xi_a - embeds_xi_tw_rm)[twm]
            features_l = (embeds_xi_a - embeds_xi_l_rm)[lm]
            features_b = (embeds_xi_a - embeds_xi_b_rm)[bm]

            # count activate attributes
            counts = torch.tensor([om.sum(), twm.sum(), lm.sum(), bm.sum()], device=td.device)
            # label
            values = torch.tensor([8, 4, 2, 1], device=td.device)
            # generate label
            labels = torch.repeat_interleave(values, counts)

            # concat features
            features = torch.cat([features_o, features_tw, features_l, features_b], dim=0)

            # calculate compositional loss
            loss_comp = self.MaskBasedInfoNCE(features, labels, temperature=self.nce_temp)
            out["loss_comp"] = loss_comp
            
        else:
            raise NotImplementedError
        return out
    
    def mask_col_features(self, env, td, mask, i):
        if i == 0:
            td = env.generator._default_open(td, mask[:])
        elif i == 1:
            td = env.generator._default_time_window(td, mask[:])
        elif i == 2:
            td = env.generator._default_distance_limit(td, mask[:])
        elif i ==3:
            td = env.generator._default_backhaul(td, mask[:])
        return td
    
    
    def get_embeds(self, td, env=None, phase='', use_encoder=True):
        context=None

        if use_encoder:
            if self.scene in [112, 115]:
                h, h_prpt = get_112_emb(self.encoder, td)
                embeds = h.mean(1)
        return embeds, context
    
    
    def get_positive_pairs2(self, features, labels, temperature=1.0):
        """
       
        Args:
            features: (B, D)-sized feature tensor
            labels: (B,)-sized label tensor
            
        Returns:
            anchor_features: (valid_pairs, D)
            positive_features: (valid_pairs, D)
            valid_mask: (B,) identify positive pair
        """
        unique_labels_ = torch.unique(labels).detach().cpu().numpy()
        device = features.device
        B = features.size(0)
        
        all_features = []
        class_n_samples = {}

        valid_labels = []

        for label in unique_labels_:
            mask = (labels == label)
            if mask.sum() >= 2:  # minimum
                class_features = features[mask]
                all_features.append(class_features)
                class_n_samples[label] = class_features.size(0)
                valid_labels.append(label)  # valid label
            
        unique_labels = np.array(valid_labels)  # new tensor
        
        if not all_features:
            return None
        
        from itertools import chain
        
        min_samples = min(class_n_samples.values())
        new_labels = list(chain(*[[label]*min_samples for label in unique_labels]))
        concatenated_features = torch.cat([v[:min_samples] for v in all_features], dim=0)

        positive_logits, negative_logits = [], []
        new_labels = torch.tensor(new_labels, device=device)
        for i, label in enumerate(unique_labels):
            
            positive_idx =torch.arange(1, min_samples+1) % (class_n_samples[label])
            query = all_features[i][:min_samples]
            positive_key = all_features[i][positive_idx]
            positive_logit = torch.sum(query * positive_key, dim=-1, keepdim=True)
            positive_logits.append(positive_logit)
            
            if len(query.size()) == 2:
                negative_logit = query @ (concatenated_features[new_labels != label]).T
            else:
                # (Q, N+1, H) -> (N+1, Q, H)
                query_ = query.transpose(0,1)
                # (C, N+1, H) -> (N+1, H, C)
                cf_ = concatenated_features[new_labels != label].permute(1, 2, 0)
                # (N+1, Q, C) -> (Q, N+1, C)
                negative_logit = (query_ @ cf_).transpose(0,1)

            negative_logits.append(negative_logit)
        
        positive_logits = torch.cat(positive_logits, dim=0)
        negative_logits = torch.cat(negative_logits, dim=0)
        logits = torch.cat([positive_logits, negative_logits], dim=-1)
        if len(logits.size()) == 3:
            # (Q, N+1, C+1) -> (Q*N+1, C+1)
            logits = logits.reshape(-1, logits.size(-1))
        ce_label = torch.zeros(len(logits), dtype=torch.long, device=logits.device)
        loss = F.cross_entropy(logits/temperature, ce_label, reduction='mean')
        
        return loss
    
    def MaskBasedInfoNCE(self, features, labels, temperature=1.0):
        # L2 regularize
        features = F.normalize(features, dim=-1)
        loss = self.get_positive_pairs2(features, labels, temperature)
        return loss

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
    
    def create_single_feature_masks(self, labels):
        """
        Generate mask for activated attributes.
        
        Args:
            labels (torch.Tensor): Shape (N, 4) boolean tensor
            
        Returns:
            masks_list (list): (N, 4)-sized boolean tensor
        """
        
        batch_size, n_features = labels.shape
        masks_list = []
        
        for feature_idx in range(n_features):
            # activate -> True
            feature_mask = torch.zeros_like(labels, dtype=torch.bool)
            valid_instances = labels[:, feature_idx]
            
            feature_mask[valid_instances, feature_idx] = True
            
            masks_list.append(feature_mask)
        
        return masks_list[0][:,0], masks_list[1][:,1], masks_list[2][:,2], masks_list[3][:,3]

    def get_keep_mask(self, env, td):
        if 'keep_mask' in td.keys():
            binary_vectors = td['keep_mask']
        else:
            ( 
                has_open,
                has_time_window,
                has_duration_limit,
                has_backhaul,
                backhaul_class,
            ) = env.check_variants(td)
            binary_vectors = torch.stack([has_open, has_time_window, has_duration_limit, has_backhaul], dim=1)
        return binary_vectors


def get_112_emb(encoder, td_test):
    # case 2 - intermediate emb
    init_h = encoder.init_embedding(td_test)  # [B, N, H]
    p = encoder.init_prompt(td_test)

    # Process embedding
    h = init_h
    for layer in encoder.layers:
        h = layer(h)
    
    h_prpt, _ = encoder.prpt_layer(h, p)
    
    return h, h_prpt