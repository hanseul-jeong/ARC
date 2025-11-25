from typing import Any

import torch
import torch.nn as nn

from typing import Union
from tensordict.tensordict import TensorDict
import numpy as np
import torch

from baseline.routefinder_v2.routefinder.routefinder.envs.mtvrp import MTVRPEnv
from baseline.routefinder_v2.routefinder.routefinder.models.model import RouteFinderSingleVariantSampling
from .generator import MTVRPGenerator
from experiments.ours.generator import MTVRPGeneratorLeaveOut
from rl4co.envs.common.base import RL4COEnvBase
from rl4co.utils.ops import gather_by_index, get_distance

from rl4co.utils.pylogger import get_pylogger


log = get_pylogger(__name__)

class OurEnv(MTVRPEnv):
    def __init__(
        self, 
        generator: MTVRPGenerator = None, 
        generator_params: dict = ..., 
        select_start_nodes_fn: Union[str, callable] = "all", 
        check_solution: bool = False, 
        load_solutions: bool = True, 
        solution_fname: str = "_sol_pyvrp.npz", 
        use_penalty: bool = False,
        PENALTY: float = 10.0,
        use_prob_penalty: bool = False,
        penalty_prob: float = 0.5,
        change_var_check: bool = False,
        **kwargs
    ):
        if generator is None:
            generator = MTVRPGeneratorLeaveOut(**generator_params)
        super().__init__(generator, generator_params, select_start_nodes_fn, check_solution, load_solutions, solution_fname, **kwargs)


class RouteFinderSingleVariantSamplingLeaveOut(RouteFinderSingleVariantSampling):
    """This is the default sampling method for MVMoE and MTPOMO
    (without Mixed-Batch Training) as first proposed in MTPOMO (https://arxiv.org/abs/2402.16891)

    The environment generates by default all the features,
    and we subsample them at each batch to train the model (i.e. we select a subset of the features).

    For example: we always sample OVRPBLTW (with all features) and we simply take a subset of them at each batch.

    Note we removed the support for single_feat_otw (original MVMoE more restricted setting) since it is not used
    in the experiments in Foundation Model settings, however it can be added back if needed
    """

    def __init__(
        self,
        env: RL4COEnvBase,
        policy: nn.Module,
        preset: str = "all",
        leaveout: list = [],
        **kwargs,
    ):
        assert preset in [
            "all",
        ], "preset must be all"
        self.preset = preset
        self.leaveout = leaveout
        self.problem_presets = {
        "cvrp": [0, 0, 0, 0],
        "ovrp": [1, 0, 0, 0],
        "vrpb": [0, 0, 0, 1],
        "vrpl": [0, 0, 1, 0],
        "vrptw": [0, 1, 0, 0],
        "ovrptw": [1, 1, 0, 0],
        "ovrpb": [1, 0, 0, 1],
        "ovrpl": [1, 0, 1, 0],
        "vrpbl": [0, 0, 1, 1],
        "vrpbtw": [0, 1, 0, 1],
        "vrpltw": [0, 1, 1, 0],
        "ovrpbl": [1, 0, 1, 1],
        "ovrpbtw": [1, 1, 0, 1],
        "ovrpltw": [1, 1, 1, 0],
        "vrpbltw": [0, 1, 1, 1],
        "ovrpbltw": [1, 1, 1, 1],
        }
        self._update_problem_list()
        assert (
            not env.generator.subsample
        ), "The env generator must not subsample the features, this is done in the `shared_step` method"

        super(RouteFinderSingleVariantSampling, self).__init__(
            env,
            policy,
            **kwargs,
        )

    def shared_step(
        self, batch: Any, batch_idx: int, phase: str, dataloader_idx: int = None
    ):
        td = batch

        # variant subsampling: given a batch with *all* features, we subsample a part of them
        if phase == "train":

            # Sample single variant (i.e which features to *remove* with 50% probability)
            # indices = torch.bernoulli(torch.tensor([0.5] * 4))
            indices = self.subsample_problems()

            # Process the indices
            if indices[0] == 1:  # Remove open
                td["open_route"] &= False
            if indices[1] == 1:  # Remove time window
                td["time_windows"][..., 0] *= 0
                td["time_windows"][..., 1] += float("inf")
                td["service_time"] *= 0
            if indices[2] == 1:  # Remove distance limit
                td["distance_limit"] += float("inf")
            if indices[3] == 1:  # Remove backhaul
                td.set("demand_linehaul", td["demand_linehaul"] + td["demand_backhaul"])
                td.set("demand_backhaul", torch.zeros_like(td["demand_backhaul"]))

        return super(RouteFinderSingleVariantSampling, self).shared_step(
            td, batch_idx, phase, dataloader_idx
        )

    def _update_problem_list(self):
        for problem in self.leaveout:
            self.problem_presets.pop(problem, None)
        self.problem_presets = ~torch.tensor(list(self.problem_presets.values()), dtype=torch.bool)
        
    def subsample_problems(self):
        idx = torch.randint((len(self.problem_presets)), ())
        return self.problem_presets.int()[idx]





    