'''.optim.helper.py'''

import dolfin
import logging
import numpy as np

from . import filter
from . import config

logger = config.logger


def make_defect_like_phasefield(V, xc, r):

    if not isinstance(xc, np.ndarray) \
       or xc.ndim != 1 or len(xc) != 2:
        raise TypeError('Parameter `xc`')

    p = dolfin.Function(V)
    p_arr = p.vector().get_local()

    x = V.tabulate_dof_coordinates()
    s = ((x-xc)**2).sum(axis=1)

    p_arr[s < r**2] = 1.0
    p.vector()[:] = p_arr

    return p


def make_defect_like_phasefield_array(V, xs, r, kappa=0):
    '''
    Parameters
    ----------
    kappa : float or dolfin.Constant
        Phasefield diffusivity parameter. It is used in a diffusion filter that
        is applied to each defect-like phasfield to smooth the edge sharpness.

    '''

    if not isinstance(xs, np.ndarray) \
       or xs.ndim != 2 or xs.shape[1] != 2:
        raise TypeError('Parameter `xs`')

    if not isinstance(V, dolfin.FunctionSpace):
        raise TypeError('Parameter `V`')

    ps = []

    for x_i in xs:
        ps.append(make_defect_like_phasefield(V, x_i, r))

    if kappa is not None and float(kappa) != 0:

        if not isinstance(kappa, dolfin.Constant):
            kappa = dolfin.Constant(kappa)

        diffusion_filter = filter.DiffusionFilter(V, kappa)

        for p_i in ps:
            diffusion_filter.apply(p_i)
            p_i.vector()[:] /= p_i.vector().max()

    return ps


def meshgrid_uniform(xlim, ylim, nx, ny):

    x0, x1 = xlim
    y0, y1 = ylim

    x = np.linspace(x0, x1, nx)
    y = np.linspace(y0, y1, ny)

    x, y = np.meshgrid(x, y)

    x = x.reshape((-1,))
    y = y.reshape((-1,))

    return np.stack((x, y), axis=1)


def meshgrid_uniform_with_margin(xlim, ylim, nx, ny):

    margin_x = (xlim[1] - xlim[0]) / nx / 2
    margin_y = (ylim[1] - ylim[0]) / ny / 2

    xlim = [xlim[0] + margin_x, xlim[1] - margin_x]
    ylim = [ylim[0] + margin_y, ylim[1] - margin_y]

    return meshgrid_uniform(xlim, ylim, nx, ny)


# def meshgrid_checker_symmetric(xlim, ylim, nx, ny):

#     grid_A = meshgrid_uniform(xlim, ylim, nx, ny)

#     if nx == 1 and ny == 1:
#         return grid_A

#     x0, x1 = xlim
#     y0, y1 = ylim
#     dx = / (nx - 1)
#     dy =
#     xlim, ylim =

#     grid_B = meshgrid_uniform(xlim, ylim, nx-1, ny-1)

#     return np.concatenate((grid_A, grid_B), axis=0)


def meshgrid_checker_asymmetric(xlim, ylim, nx, ny):

    x0, x1 = xlim
    y0, y1 = ylim

    nx_B = int(nx/2)
    nx_A = nx - nx_B

    ny_B = int(ny/2)
    ny_A = ny - ny_B

    assert nx_A >= nx_B
    assert ny_A >= ny_B

    dx = (x1 - x0) / (nx - 1)
    dy = (y1 - y0) / (ny - 1)

    if nx_A == nx_B:
        xlim_A = (x0, x1-dx)
        xlim_B = (x0+dx, x1)
    else:
        xlim_A = (x0, x1)
        xlim_B = (x0+dx, x1-dx)

    if ny_A == ny_B:
        ylim_A = (y0, y1-dy)
        ylim_B = (y0+dy, y1)
    else:
        ylim_A = (y0, y1)
        ylim_B = (y0+dy, y1-dy)

    grid_A = meshgrid_uniform(xlim_A, ylim_A, nx_A, ny_A)
    grid_B = meshgrid_uniform(xlim_B, ylim_B, nx_B, ny_B)

    return np.concatenate((grid_A, grid_B), axis=0)
