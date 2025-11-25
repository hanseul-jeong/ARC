

import torch
from baseline.routefinder_v2.routefinder.routefinder.models.env_embeddings.mtvrp.init import MTVRPInitEmbeddingRouteFinderBase
from baseline.routefinder_v2.routefinder.routefinder.envs.mtvrp import MTVRPEnv


class ARCInitEmbedding(MTVRPInitEmbeddingRouteFinderBase):
    
    def __init__(self, embed_dim=128, num_global_feats=5, num_cust_feats=9, bias=False, posinf_val=0):
        super().__init__(num_global_feats, num_cust_feats, embed_dim, bias, posinf_val)

    def _global_feats(self, td):
        return torch.cat(
            [
                td["open_route"].float()[..., None],
                td["locs"][:, :1, :],
                td["distance_limit"][..., None],
                td["time_windows"][:, :1, 1:2],
            ],
            -1,
        )

    def _cust_feats(self, td):
        n_nodes = td['locs'].size(1)-1
        return torch.cat(
            (
                td["locs"][..., 1:, :],
                td["demand_linehaul"][..., 1:, None],
                td["demand_backhaul"][..., 1:, None],
                td["time_windows"][..., 1:, :],
                td["service_time"][..., 1:, None],
                td["open_route"].float()[..., None].repeat(1,n_nodes,1),
                td["distance_limit"][..., None].repeat(1,n_nodes,1),
            ),
            -1,
        )
    

class ARCInitPromptEmbedding(ARCInitEmbedding):
    def __init__(self, embed_dim=128, num_global_feats=9, num_cust_feats=13, bias=False, posinf_val=0):
        super().__init__(embed_dim, num_global_feats, num_cust_feats, bias, posinf_val)

    def _global_feats(self, td):
        # (B, 4)
        p = self._get_prompt(td)
        # (B, 1, 9)
        return torch.cat(
            [
                td["open_route"].float()[..., None],
                td["locs"][:, :1, :],
                td["distance_limit"][..., None],
                td["time_windows"][:, :1, 1:2],
                p[:,None,:]
            ],
            -1,
        )

    def _cust_feats(self, td):
        n_nodes = td['locs'].size(1)-1
        p = self._get_prompt(td)
        # (B, N, 13)
        return torch.cat(
            (
                td["locs"][..., 1:, :],
                td["demand_linehaul"][..., 1:, None],
                td["demand_backhaul"][..., 1:, None],
                td["time_windows"][..., 1:, :],
                td["service_time"][..., 1:, None],
                td["open_route"].float()[..., None].repeat(1,n_nodes,1),
                td["distance_limit"][..., None].repeat(1,n_nodes,1),
                p[:,None,:].repeat(1, n_nodes, 1)
            ),
            -1,
        )
    
    def _get_prompt(self, td):
        (
            has_open,
            has_time_window,
            has_duration_limit,
            has_backhaul,
            backhaul_class,
        ) = MTVRPEnv.check_variants(td)

        return torch.cat([has_open[:,None], has_time_window[:,None], has_duration_limit[:,None], has_backhaul[:,None]], dim=1).float()
    
from rl4co.models.nn.env_embeddings.context import MTVRPContext, EnvContext


class MTVRPContextEmbeddingM(MTVRPContext):
    """Context embedding MTVRP with mixed backhaul.
    This is for the zero-shot or few-short on backhaul_class 2 instances.
    - current time
    - used capacity
    - remaining distance (set to default_remain_dist if positive infinity)
    """

    def __init__(self, embed_dim=128, default_remain_dist=10):
        EnvContext.__init__(self, embed_dim=embed_dim, step_context_dim=embed_dim + 6)
        self.default_remain_dist = default_remain_dist

    def _state_embedding(self, embeddings, td):
        context_feats = super(MTVRPContextEmbeddingM, self)._state_embedding(
            embeddings, td
        )
        # this will be 0 and tell the model we are *not* doing VRPMPD if backhaul class is not 2
        available_load_vrpmpd = (
            td["vehicle_capacity"] - td["used_capacity_backhaul"]
        ) * (td["backhaul_class"] == 2)

        # Note: now we need the projection to have embed_dim + 4 features!
        return torch.cat(
            (
                context_feats,
                available_load_vrpmpd,
            ),
            -1,
        )