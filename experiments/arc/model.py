import random

from typing import Any, Union

import torch
import torch.nn as nn

from rl4co.envs.common.base import RL4COEnvBase
from rl4co.utils.pylogger import get_pylogger

from routefinder.models.model import RouteFinderBase



log = get_pylogger(__name__)


class ARCModel(RouteFinderBase):
    """
    Main model
    """
    
    def __init__(
        self, 
        env: RL4COEnvBase, 
        policy: nn.Module, 
        nce_lambda: float = 0.1,
        **kwargs
    ):
        self.nce_lambda = nce_lambda
        
        super().__init__(
            env, 
            policy, 
            **kwargs
        )
        
    def log_metrics(
        self, metric_dict: dict, phase: str, dataloader_idx: Union[int, None] = None
    ):
        metric_dict.update({"loss_comp" : metric_dict.get("loss_comp", 0)})
        loss, loss_comp = metric_dict.get('loss', 0), metric_dict['loss_comp']
        metric_dict.update({
            "loss_total" : 
            loss 
            + self.nce_lambda * loss_comp
            })

        metrics = super().log_metrics(
            metric_dict, phase, dataloader_idx
        )
        gap = None if not isinstance(metric_dict.get('gap_to_bks', None), torch.Tensor) else metric_dict['gap_to_bks']
        metrics.update({"gap_to_bks": gap})
        return metrics
    
    def shared_step(
        self, batch: Any, batch_idx: int, phase: str, dataloader_idx: int = None
    ):
        out = super().shared_step(
            batch, batch_idx, phase, dataloader_idx
        )
        loss = out.get("loss", None)
        
        if loss is not None:
            loss_comp = out.get(f"{phase}/loss_comp", 0)
            out["loss"] = loss + self.nce_lambda * loss_comp
            
        return out
    
    