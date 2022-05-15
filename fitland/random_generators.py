import numpy as np
from scipy.spatial import ConvexHull, Delaunay
from func_components import *


def get_random_gaussian2d(xy_bounds: Tuple[Tuple[float, float]]) -> Gaussian2D:
    min_x, max_x = xy_bounds[0][0], xy_bounds[0][1]
    min_y, max_y = xy_bounds[1][0], xy_bounds[1][1]
    return Gaussian2D(means=(min_x + np.random.random() * (max_x - min_x), min_y + np.random.random() * (max_y - min_y)),
                      stds=(1e-2 + np.random.random() * 2, 1e-2 + np.random.random() * 2),
                      theta=np.random.random() * np.pi / 2,
                      w=np.random.random())


def get_random_circle2d(xy_bounds: Tuple[Tuple[float, float]],
                        feasible: bool = True) -> Circle2D:
    min_x, max_x = xy_bounds[0][0], xy_bounds[0][1]
    min_y, max_y = xy_bounds[1][0], xy_bounds[1][1]
    return Circle2D(center=(min_x + np.random.random() * (max_x - min_x), min_y + np.random.random() * (max_y - min_y)),
                    radius=1e-2 + np.random.random() * min(np.sqrt(max_x ** 2 + min_x ** 2), np.sqrt(max_y ** 2 + min_y ** 2)) * .25,
                    w=np.random.random() if feasible else np.nan)


def get_random_plane2d(xy_bounds: Tuple[Tuple[float, float]]) -> Plane2D:
    params = [
        np.random.random() * 2 - 1,
        np.random.random() * 2 - 1,
        np.random.random() * 2 - 1,
        np.random.random() * 2 - 1,        
    ]
    return Plane2D(params=params,
                   w=np.random.random())
    

def get_random_perlin2d(xy_bounds: Tuple[Tuple[float, float]]) -> Perlin2D:
    return Perlin2D(w=np.random.random())

def get_random_peak2d(xy_bounds: Tuple[Tuple[float, float]]) -> Peak2D:
    min_x, max_x = xy_bounds[0][0], xy_bounds[0][1]
    min_y, max_y = xy_bounds[1][0], xy_bounds[1][1]
    return Peak2D(center=(min_x + np.random.random() * (max_x - min_x), min_y + np.random.random() * (max_y - min_y)),
                  r=np.random.random(),
                  w=np.random.random())

def get_random_poly2d(xy_bounds: Tuple[Tuple[float, float]],
                      feasible: bool = True) -> Poly2D:
    min_x, max_x = xy_bounds[0][0], xy_bounds[0][1]
    min_y, max_y = xy_bounds[1][0], xy_bounds[1][1]
    center = min_x + (np.random.random() * (max_x - min_x)), min_y + (np.random.random() * (max_y - min_y))
    points = np.transpose(np.vstack([np.random.normal(loc=center[0], scale=1, size=16), np.random.normal(loc=center[1], scale=1, size=16)]))
    hull = ConvexHull(points)
    return Poly2D(points=points[hull.vertices],
                  w=np.random.random() if feasible else np.nan)