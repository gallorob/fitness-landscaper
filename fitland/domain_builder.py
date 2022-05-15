import numpy as np
from typing import List, Union

from domain import Domain2D
from func_components import *
from random_generators import *


# _feasible_components = [Gaussian2D, Circle2D, Circle2D, Poly2D, Plane2D, Perlin2D]
_feasible_components = [Gaussian2D, Circle2D, Poly2D, Plane2D, Perlin2D, Peak2D]
_infeasible_components = [Circle2D, Poly2D]

component_generator = {
    Gaussian2D: get_random_gaussian2d,
    Circle2D: get_random_circle2d,
    Poly2D: get_random_poly2d,
    Plane2D: get_random_plane2d,
    Perlin2D: get_random_perlin2d,
    Peak2D: get_random_peak2d
}


class DomainBuilder:
    def __init__(self) -> None:
        pass
    
    def build_domain(self,
                     name: str,
                     components: Union[List[FunctionComponent], Tuple[int, int]],
                     **kwargs) -> Domain2D:
        if type(components[0]) is int:
            cs = []
            for _ in range(components[0]):  # feasible domain
                cs.append(component_generator[np.random.choice(_feasible_components)](xy_bounds=kwargs['xy_bounds'])) 
            for _ in range(components[1]):  # infeasible domain
                cs.append(component_generator[np.random.choice(_infeasible_components)](xy_bounds=kwargs['xy_bounds'], feasible=False)) 
            components = cs
        assert type(components) is list, f'Unexpected type for `components` argument: {type(components)}, expected list.'
        domain = Domain2D(name=name,
                          xy_bounds=kwargs['xy_bounds'],
                          z_bounds=kwargs.get('z_bounds', []),
                          func_components=components,
                          step_res=kwargs.get('step_res', 1e-2),
                          noisy_factor=kwargs.get('noisy_factor', 0.),
                          use_max=kwargs.get('use_max', False),
                          offset_n=kwargs.get('offset_n', None),
                          scale_n=kwargs.get('scale_n', None),
                          offset=kwargs.get('offset', None),
                          scale=kwargs.get('scale', None))
        return domain
    