from abc import ABC, abstractmethod
from typing import Callable, List, Optional, Tuple

import matplotlib.path as plt_path
import numpy as np


class FunctionComponent(ABC):
    def __init__(self,
                 w: float,
                 smooth: bool = False):
        """Generate a function component.

        Args:
            w (float): The weight of the component.
            smooth (bool, optional): Whether to smooth the edges of the function. Defaults to False.
        """
        assert type(
            w) is not None, f'A weight must always be specified for a component.'
        if not np.isnan(w):
            assert 0 <= abs(w) <= 1, f'Weight must be `0 <= |w| <= 1`; you passed {w}.'
        self.w = w
        self.smooth = smooth

    @abstractmethod
    def _eval(self,
              x: np.array,
              y: np.array) -> np.array:
        """Internal evaluation of the component.

        Args:
            x (np.array): The input X vector.
            y (np.array): The input Y vector.

        Returns:
            np.array: The output Z vector.
        """
        pass

    def eval(self,
             x: np.array,
             y: np.array) -> np.array:
        """Evaluate the function component.

        Args:
            x (np.array): The input X vector.
            y (np.array): The input Y vector.

        Returns:
            np.array: The output Z vector.
        """
        s = self._eval(x, y)
        if np.isnan(self.w):
            s[s > 0] = 0
            s[s < 0] = self.w
        else:
            s = self.w * s
        if self.smooth:
            from scipy.ndimage.filters import gaussian_filter
            s = gaussian_filter(s, sigma=10)
        return s

    @abstractmethod
    def __str__(self) -> str:
        pass

class CustomFunc2D(FunctionComponent):
    def __init__(self,
                 name: str,
                 f: Callable[[np.array, np.array], np.array],
                 w: float = 1):
        """A custom function component.

        Args:
            f (Callable[[np.array, np.array], np.array]): The custom 2D function.
            w (float): The weight of the component. Defaults to 1.
        """
        super().__init__(w)
        self.name = name
        self.f = f

    def _eval(self,
              x: np.array,
              y: np.array) -> np.array:
        """Internal evaluation of the component.

        Args:
            x (np.array): The input X vector.
            y (np.array): The input Y vector.

        Returns:
            np.array: The output Z vector.
        """
        return self.f(x, y)
    
    def __str__(self) -> str:
        return f'{self.name} component: weight={self.w}'

class Gaussian2D(FunctionComponent):
    def __init__(self,
                 means: Tuple[float, float],
                 stds: Tuple[float, float],
                 theta: float,
                 w: float = 1):
        """Generate a 2D Gaussian PDF component.

        Args:
            means (Tuple[float, float]): The means of the 2D Gaussian.
            stds (Tuple[float, float]): The standard deviation of the 2D Gaussian.
            theta (float): The rotation (in radians).
            w (float): The weight of the component. Defaults to 1.
        """
        super().__init__(w)
        self.x0, self.y0 = means
        self.sigma_x, self.sigma_y = stds
        self.theta = theta

    def _eval(self,
              x: np.array,
              y: np.array) -> np.array:
        """Internal evaluation of the component.

        Args:
            x (np.array): The input X vector.
            y (np.array): The input Y vector.

        Returns:
            np.array: The output Z vector.
        """
        a = (np.cos(self.theta) ** 2) / (2 * self.sigma_x ** 2) + (np.sin(self.theta) ** 2) / (2 * self.sigma_y ** 2)
        b = -(np.sin(2 * self.theta)) / (4 * self.sigma_x ** 2) + (np.sin(2 * self.theta)) / (4 * self.sigma_y ** 2)
        c = (np.sin(self.theta) ** 2) / (2 * self.sigma_x ** 2) + (np.cos(self.theta) ** 2) / (2 * self.sigma_y ** 2)
        return np.exp(-(a * ((x - self.x0) ** 2) + 2 * b * (x - self.x0) * (y - self.y0) + c * ((y - self.y0) ** 2)))

    def __str__(self) -> str:
        return f'Gaussian2D component: N({self.x0}, {self.sigma_x}), N({self.y0}, {self.sigma_y}), theta={self.theta}, weight={self.w}'

class Circle2D(FunctionComponent):
    def __init__(self,
                 center: Tuple[float, float],
                 radius: float,
                 w: float = 1):
        """Generate a 2D circle component.

        Args:
            center (Tuple[float, float]): The center of the circle.
            radius (float): The radius of the circle.
            w (_type_): The weight of the component. Defaults to 1.
        """
        super().__init__(w, not np.isnan(w))
        self.x, self.y = center
        self.r = radius

    def _eval(self,
              x: np.array,
              y: np.array) -> np.array:
        """Internal evaluation of the component.

        Args:
            x (np.array): The input X vector.
            y (np.array): The input Y vector.

        Returns:
            np.array: The output Z vector.
        """
        v = (x - self.x) ** 2 + (y - self.y) ** 2 - self.r ** 2
        v[v > 0] = 0
        v[v < 0] = -1 if np.isnan(self.w) else 1
        return v
    
    def __str__(self) -> str:
        return f'Circle2D component: center=({self.x}, {self.y}), radius={self.r}, weight={self.w}'

class Poly2D(FunctionComponent):
    def __init__(self,
                 points: List[Tuple[float, float]],
                 w: float = 1):
        """Generate a 2D polygon component.

        Args:
            points (List[Tuple[float, float]]): The vertices of the polygon, described as sequence of 2D coordinates.
            w (float): The weight of the component. Defaults to 1.
        """
        super().__init__(w, not np.isnan(w))
        self.poly = plt_path.Path(points)

    def _eval(self,
              x: np.array,
              y: np.array) -> np.array:
        """Internal evaluation of the component.

        Args:
            x (np.array): The input X vector.
            y (np.array): The input Y vector.

        Returns:
            np.array: The output Z vector.
        """
        s = self.poly.contains_points(
            np.vstack([x, y]).transpose()).astype(float)
        s[s == 0] = 0
        s[s == 1] = -1 if np.isnan(self.w) else 1
        return s
    
    def __str__(self) -> str:
        return f'Poly2D component: points: {self.poly.vertices.tolist()}, weight={self.w}'

class Plane2D(FunctionComponent):
    def __init__(self,
                 points: List[Tuple[float, float]] = [],
                 params: Optional[List[float]] = [],
                 w: float = 1):
        """Generates a 2D plane component.

        Args:
            points (List[Tuple[float, float]], optional): A sequence of points on which the plane lies on. Defaults to [].
            params (Optional[List[float]], optional): The parameters of the plane equation. Defaults to [].
            w (float, optional): The weight of the component. Defaults to 1.

        Raises:
            ValueError: Raised if neither `points` nor `params` are passed.
        """
        super().__init__(w)
        if points:
            # https://stackoverflow.com/questions/53701626/plane-fitting-through-points-in-3d-using-python
            x1, y1, z1 = points[0]
            x2, y2, z2 = points[1]
            x3, y3, z3 = points[2]
            a1 = x2 - x1
            b1 = y2 - y1
            c1 = z2 - z1
            a2 = x3 - x1
            b2 = y3 - y1
            c2 = z3 - z1
            self.a = b1 * c2 - b2 * c1
            self.b = a2 * c1 - a1 * c2
            self.c = a1 * b2 - b1 * a2
            self.d = (- self.a * x1 - self.b * y1 - self.c * z1)
        elif params:
            self.a, self.b, self.c, self.d = params[0], params[1], params[2], params[3]
        else:
            raise ValueError(
                'You must pass either a list of 3D points or the plane equation\'s parameters.')
    
    def _eval(self,
              x: np.array,
              y: np.array) -> np.array:
        """Internal evaluation of the component.

        Args:
            x (np.array): The input X vector.
            y (np.array): The input Y vector.

        Returns:
            np.array: The output Z vector.
        """
        return -((self.a * x) + (self.b * y) + self.d)

    def __str__(self) -> str:
        return f'Plane2D component: {self.a}x + {self.b}y + {self.c}z + {self.d} = 0, weight={self.w}'

class Perlin2D(FunctionComponent):
    # adapted from https://stackoverflow.com/questions/42147776/producing-2d-perlin-noise-with-numpy
    def __init__(self,
                 w: float = 1):
        """Generate a 2D Perlin noise component.

        Args:
            as_noise (bool, optional): Whether to use the component solely as noise. Defaults to False.
            w (float, optional): The weight of the component. Defaults to 1.
        """
        super().__init__(w)
        self._initialize()

    def _initialize(self) -> None:
        """Create the permutation table `p`."""
        p = np.arange(0, 256, dtype=int)
        np.random.shuffle(p)
        self.p = np.stack([p, p]).flatten()

    def _lerp(self,
              a: np.array,
              b: np.array,
              x: np.array) -> np.array:
        """Apply linear interpolation.

        Args:
            a (np.array): The lower value.
            b (np.array): The upper value.
            x (np.array): The array to interpolate.

        Returns:
            np.array: The interpolated array.
        """
        return a + x * (b - a)

    def _fade(self,
              t: np.array):
        """Fade out array of floats following `6t^5 - 15t^4 + 10t^3`.

        Args:
            t (np.array): The array of floats.

        Returns:
            np.array: The faded-out array.
        """
        return 6 * t**5 - 15 * t**4 + 10 * t**3

    def _gradient(self,
                  h: int,
                  x: np.array,
                  y: np.array) -> np.array:
        """Compute the gradient and return dot product with `x` and `y`.

        Args:
            h (int): The patch number.
            x (np.array): The input X vector.
            y (np.array): The input Y vector.

        Returns:
            np.array: The gradient in the patch direction.
        """
        vectors = np.array([[0, 1], [0, -1], [1, 0], [-1, 0]])
        g = vectors[h % 4]
        return g[:, 0] * x + g[:, 1] * y

    def _perlin(self,
                x: np.array,
                y: np.array) -> np.array:
        """Compute the Perlin noise.

        Args:
            x (np.array): The input X vector.
            y (np.array): The input Y vector.

        Returns:
            np.array: The Perlin noise vector.
        """
        x = np.abs(np.min(x)) + x
        y = np.abs(np.min(y)) + y
        xi, yi = x.astype(int), y.astype(int)
        xf, yf = x - xi, y - yi
        u, v = self._fade(xf), self._fade(yf)
        n00 = self._gradient(self.p[self.p[xi] + yi], xf, yf)
        n01 = self._gradient(self.p[self.p[xi] + yi + 1], xf, yf - 1)
        n11 = self._gradient(self.p[self.p[xi + 1] + yi + 1], xf - 1, yf - 1)
        n10 = self._gradient(self.p[self.p[xi + 1] + yi], xf - 1, yf)
        x1 = self._lerp(n00, n10, u)
        x2 = self._lerp(n01, n11, u)
        return self._lerp(x1, x2, v)

    def _eval(self,
              x: np.array,
              y: np.array) -> np.array:
        """Internal evaluation of the component.

        Args:
            x (np.array): The input X vector.
            y (np.array): The input Y vector.

        Returns:
            np.array: The output Z vector.
        """
        return self._perlin(x, y)
    
    def __str__(self) -> str:
        return f'Perlin2D component: weight={self.w}'

class Peak2D(FunctionComponent):
    def __init__(self,
                 center: Tuple[float, float],
                 r: float,
                 w: float = 1):
        """Generates a 2D peak.

        Args:
            center (Tuple[float, float]): The center of the peak.
            r (float): The slope of the peak.
            w (float, optional): The weight of the component. Defaults to 1.
        """
        super().__init__(w)
        self.x, self.y = center[0], center[1]
        self.r = r
    
    def _eval(self,
              x: np.array,
              y: np.array) -> np.array:
        """Internal evaluation of the component.

        Args:
            x (np.array): The input X vector.
            y (np.array): The input Y vector.

        Returns:
            np.array: The output Z vector.
        """
        return 1 - self.r * np.sqrt((x - self.x) ** 2 + (y - self.y) ** 2)

    def __str__(self) -> str:
        return f'Peak2D component: center=({self.x}, {self.y}), slope={self.r}, weight={self.w}'