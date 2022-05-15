from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm

from func_components import FunctionComponent


class Domain2D:
    def __init__(self,
                 name: str,
                 xy_bounds: Tuple[Tuple[float, float]],
                 z_bounds: Optional[Tuple[float]],
                 func_components: List[FunctionComponent],
                 step_res: float = 1e-2,
                 noisy_factor: float = 0.,
                 use_max: bool = False,
                 offset_n: Optional[float] = None,
                 scale_n: Optional[float] = None,
                 offset: Optional[float] = None,
                 scale: Optional[float] = None):
        """Generate a 2D domain.

        Args:
            name (str): The name of the domain.
            xy_bounds (Tuple[Tuple[float, float]]): The bounds of the XY domain.
            z_bounds (Optional[Tuple[float]]): The bounds of the Z axis.
            func_components (List[FunctionComponent]): The list of components.
            step_res (float, optional): Resolution of the domain preview. Defaults to 1e-2.
            noisy_factor (float): Percentage of additional noise to add in the final evaluation. Note that this may bring values outside of the [0, 1] range. Defaults to 0.0.
            use_max (bool): Whether to combine components by max or by summing them. Defaults to False.
            offset_n (Optional[float], optional): Offset (normalization) for the Z values. Defaults to None.
            scale_n (Optional[float], optional): Scale (normalization) for the Z values. Defaults to None.
            offset (Optional[float], optional): Offset for the Z values. Defaults to None.
            scale (Optional[float], optional): Scale for the Z values. Defaults to None.
        """
        self.name = name
        self.x_min, self.x_max = xy_bounds[0][0], xy_bounds[0][1]
        assert self.x_min < self.x_max, f'Invalid Z bounds: {self.x_min} must be < than {self.x_max}'
        self.y_min, self.y_max = xy_bounds[1][0], xy_bounds[1][1]
        assert self.y_min < self.y_max, f'Invalid Z bounds: {self.y_min} must be < than {self.y_max}'
        self.func_components = func_components
        self.step_res = step_res
        self.noisy_factor = noisy_factor
        self.use_max = use_max
        if z_bounds:
            self.z_min = z_bounds[0]
            self.z_max = z_bounds[1]
            assert self.z_min < self.z_max, f'Invalid Z bounds: {self.z_min} must be < than {self.z_max}'
        else:
            self.z_min = 0
            self.z_max = 1
        if offset_n is not None and scale_n is not None and offset is not None and scale is not None:
            self.offset_n = offset_n
            self.scale_n = scale_n
            self.offset = offset
            self.scale = scale
        else:
            self.offset_n = 0
            self.scale_n = 1
            self.offset = 0
            self.scale = 1
            self._initialize()

    def _precompute(self) -> Tuple[np.array, np.array, np.array]:
        """Compute the domain.

        Returns:
            Tuple[np.array, np.array, np.array]: The X, Y, and Z values of the domain.
        """
        xs = np.arange(self.x_min, self.x_max, self.step_res)
        ys = np.arange(self.y_min, self.y_max, self.step_res)
        X, Y = np.meshgrid(xs, ys)
        Z = self.eval(np.ravel(X), np.ravel(Y)).reshape(X.shape)

        return X, Y, Z

    def _initialize(self):
        """Compute the offset and scale values of the domain."""
        _, _, z = self._precompute()
        zmin = np.min(z[~np.isnan(z)])
        zmax = np.max(z[~np.isnan(z)])
        self.offset_n = zmin
        self.scale_n = zmax - zmin
        self.offset = 0 - self.z_min
        self.scale = self.z_max - self.z_min

    def eval(self,
             x: np.array,
             y: np.array) -> np.array:
        """Evaluates the given point in the domain.

        Args:
            x (np.array): The array of X coordinates.
            y (np.array): The array of Y coordinates.

        Returns:
            np.array: The array of Z values for the given (x,y) coordinates. `NaN` values are used for infeasible areas.
        """
        op = np.max if self.use_max else np.sum
        f = op([f.eval(x, y) for f in self.func_components], axis=0)
        f = self.scale * ((f - self.offset_n) / self.scale_n) - self.offset
        f += self.noisy_factor * np.random.normal(loc=0., scale=1., size=f.shape)

    def add_component(self,
                      component: FunctionComponent,
                      needs_reinit: bool = True) -> None:
        """Add a component to the domain.

        Args:
            component (FunctionComponent): The component to add.
            needs_reinit (bool, optional): Whether to re-initialize the domain. Defaults to True.
        """
        self.func_components.append(component)
        if needs_reinit:
            self.offset_n = 0
            self.scale_n = 1
            self.offset = 0
            self.scale = 1
            self._initialize()
    
    def show(self) -> None:
        """Plot the domain (in 3D and 2D)"""
        X, Y, Z = self._precompute()

        fig = plt.figure(figsize=plt.figaspect(0.5))
        ax = fig.add_subplot(121,
                             projection='3d')
        surf = ax.plot_surface(X, Y, Z,
                               cmap=cm.coolwarm,
                               antialiased=True,
                               vmin=self.z_min,
                               vmax=self.z_max)
        cbar = fig.colorbar(surf)
        cbar.set_label('Fitness')
        ax.set_xlim(self.x_min, self.x_max)
        ax.set_ylim(self.x_min, self.x_max)
        ax.set_zlim(self.z_min, self.z_max)
        ax.set_xlabel(r'$\vec{X}$')
        ax.set_ylabel(r'$\vec{Y}$')
        ax.set_zlabel(r'$\vec{Z}$')
        plt.title('3D landscape')

        plt.subplot(122)
        surf = plt.imshow(Z,
                          cmap=cm.coolwarm,
                          aspect='equal',
                          origin='lower',
                          vmin=self.z_min,
                          vmax=self.z_max)
        cbar = plt.colorbar(surf)
        cbar.set_label('Fitness')
        plt.xticks([])
        plt.yticks([])
        plt.xlabel(r'$\vec{X}$')
        plt.ylabel(r'$\vec{Y}$')
        plt.title('2D landscape')

        plt.suptitle(f'{self.name} preview')
        plt.show()

    def __str__(self) -> str:
        s = f'Properties of \"{self.name}\":\n'
        s += f'Bounds are:\n\tX: [{self.x_min}, {self.x_max}]\n\tY: [{self.y_min}, {self.y_max}]\n\tZ: [{self.z_min}, {self.z_max}]\n'
        s += f'Noise factor: {self.noisy_factor}\n'
        s += f'Z-values manipulations are:\n\tOffset (normalize): {np.round(self.offset_n, 4)}\tScale (normalize): {np.round(self.scale_n, 4)}\n\tOffset: {np.round(self.offset, 4)}\tScale: {np.round(self.scale, 4)}\n'
        s += f'Z-values are combined with {"max" if self.use_max else "sum"}.\n'
        s += f'This domain has {len(self.func_components)} active components:\n\t'
        s += '\n\t'.join([str(c) for c in self.func_components])
        return s
