import torch
from typing import Callable, Tuple, Union

# from baseline.routefinder_v2.routefinder.envs.mtvrp.generator import MTVRPGenerator
from baseline.routefinder_v2.routefinder.routefinder.envs.mtvrp.generator import MTVRPGenerator
from tensordict.tensordict import TensorDict
from collections import Counter
import numpy as np
from rl4co.utils.ops import get_distance

problems_dict = {
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
        # 'leaveout':[-1, -1, -1, -1],
    }
     
class MTVRPGeneratorLeaveOut(MTVRPGenerator):
    problem_presets = {
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
        # 'leaveout':[-1, -1, -1, -1],
    }
    
    def __init__(self, all_mb=False, **kwargs) -> None:
        super().__init__(**kwargs)
        assert self.use_combinations and self.leaveout is not None, f"Use default Generator class {self.leaveout} / {self.use_combinations}"
        self._update_problem_list()
        self.c = kwargs.get('c', 0.2)
        if all_mb:
            self.generate_backhaul_class = self._generate_all_backhaul_class
            
    def _generate_all_backhaul_class(self, shape: Tuple[int, int], sample: bool = False):
        """Generate backhaul class (B) for each node. If sample is True, we sample the backhaul class
        otherwise, we return the same class for all nodes.
        - Backhaul class 1: classic backhaul (VRPB), linehauls must be served before backhauls in a route (every customer is either, not both)
        - Backhaul class 2: mixed backhaul (VRPMPD or VRPMB), linehauls and backhauls can be served in any order (every customer is either, not both)
        """
        if sample:
            return torch.ones(shape, dtype=torch.float32)*2
        else:
            raise NotImplementedError
        
    def _update_problem_list(self):
        for problem in self.leaveout:
            self.problem_presets.pop(problem, None)
        self.problem_presets = torch.tensor(list(self.problem_presets.values()), dtype=torch.bool)
        
    def subsample_problems(self, td):
        batch_size = td.batch_size[0]
        indices = torch.randint(len(self.problem_presets), (batch_size,))
        keep_mask = self.problem_presets[indices]
        
        # reset to default problem setting for index of False
        td = self._default_open(td, ~keep_mask[:, 0])
        td = self._default_time_window(td, ~keep_mask[:, 1])
        td = self._default_distance_limit(td, ~keep_mask[:, 2])
        td = self._default_backhaul(td, ~keep_mask[:, 3])

        td = self._sanitycheck_backhaul(td, keep_mask[:,3])
        
        return td
    
    def _sanitycheck_backhaul(self, td, keep):
        def find_zero_values(td_values, keep):
            zero_mask = (td_values == 0) & keep
            return torch.where(zero_mask)[0]
        for _ in range(3):  # max three trials
            invalid = find_zero_values(td['demand_backhaul'].sum(1), keep)
            if not invalid.size(0):
                return td
            print('attempt to regenerate invalid samples')
            demand_backhaul = self._regenerate_backhaul(invalid.size(0))
            td['demand_backhaul'][invalid] = demand_backhaul
        
        raise NotImplementedError("three trials are not enough")
            
    def _regenerate_backhaul(self, n_regen):

        demand_linehaul, demand_backhaul = self.generate_demands(
            batch_size=[n_regen], num_loc=self.num_loc
        )
        vehicle_capacity = torch.full(
            (n_regen, 1), self.capacity, dtype=torch.float32
        )
        if self.scale_demand:
            demand_backhaul /= vehicle_capacity
        
        return demand_backhaul
    
    def _init_task_weights_variables(self, ):
        self.g = torch.zeros(self.problem_presets.size(0))
        self.task_names = self.get_problem_names()

    def get_problem_names(self, ):
        preset_lists = [[int(x) for x in preset] for preset in self.problem_presets.tolist()]
        
        problem_names = []
        for preset in preset_lists:
            for name, pattern in problems_dict.items():
                if preset == pattern:
                    problem_names.append(name)
                    break
                
        return problem_names
    
    def update_task_weights(self, td_tasks, gap_to_bks):
        indices = self.get_equal_val(td_tasks)
        task_weights = self.cal_task_weight(indices, gap_to_bks)

    def get_equal_val(self, td_tasks):
        np_task = np.array(td_tasks)
        if len(Counter(td_tasks)) == 1:
            return np.empty([])
        n_sample = Counter(td_tasks).most_common()[-1][-1]
        indices = {}
        for task in self.task_names:
            mask = np.zeros_like(np_task, dtype=bool)
            task_indices = np.where(np_task == task.upper())[0]
            selected_indices = np.random.choice(task_indices, n_sample, replace=False)
            mask[selected_indices] = True
            indices[task.upper()] = mask
        
        return indices

    def generate_time_windows(
        self,
        locs: torch.Tensor,
        speed: torch.Tensor,
    ) -> torch.Tensor:
        """Generate time windows (TW) and service times for each location including depot.
        We refer to the generation process in "Multi-Task Learning for Routing Problem with Cross-Problem Zero-Shot Generalization"
        (Liu et al., 2024). Note that another way to generate is from "Learning to Delegate for Large-scale Vehicle Routing" (Li et al, 2021) which
        is used in "MVMoE: Multi-Task Vehicle Routing Solver with Mixture-of-Experts" (Zhou et al, 2024). Note that however, in that case
        the distance limit would have no influence when time windows are present, since the tw for depot is the same as distance with speed=1.
        This function can be overridden for that implementation.
        See also https://github.com/RoyalSkye/Routing-MVMoE

        Args:
            locs: [B, N+1, 2] (depot, locs)
            speed: [B]

        Returns:
            time_windows: [B, N+1, 2]
            service_time: [B, N+1]
        """

        batch_size, n_loc = locs.shape[0], locs.shape[1] - 1  # no depot

        a, b, c = 0.15, 0.18, 0.2
        # print('c = ',c)
        service_time = a + (b - a) * torch.rand(batch_size, n_loc)
        tw_length = b + (c - b) * torch.rand(batch_size, n_loc)
        d_0i = get_distance(locs[:, 0:1], locs[:, 1:])
        h_max = (self.max_time - service_time - tw_length) / d_0i * speed - 1
        tw_start = (1 + (h_max - 1) * torch.rand(batch_size, n_loc)) * d_0i / speed
        tw_end = tw_start + tw_length

        # Depot tw is 0, max_time
        time_windows = torch.stack(
            (
                torch.cat((torch.zeros(batch_size, 1), tw_start), -1),  # start
                torch.cat((torch.full((batch_size, 1), self.max_time), tw_end), -1),
            ),  # en
            dim=-1,
        )
        # depot service time is 0
        service_time = torch.cat((torch.zeros(batch_size, 1), service_time), dim=-1)
        return time_windows, service_time  # [B, N+1, 2], [B, N+1]





        
        