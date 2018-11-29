
import math
import dolfin
import numpy as np

from . import config
from . import filter
from . import utility

NP_EXP = np.exp
MATH_E = math.e
INV_MATH_E = 1.0/MATH_E


def bump(x):
    # y = (2*x-1)**2 * (1.0-(2*x-1)**2) # 0 .. 1
    # y = 0.5 - 0.5*np.cos((PI)*x)
    # y = 1/(1+x)
    y = np.array((x > 0.25) * (x < 0.75))

    return y

def ramp(x):

    y0 = 0.5 * (0.5 - 0.5*np.cos((2*PI)*x))
    y1 = 0.5 + 0.5 * (0.5 - 0.5*np.cos((2*PI)*(x-0.5)))

    y = np.where(x < 0.5, y0, y1)

    return y

def weight(x, bump_alpha=10):
    # return 1.0 + bump(x) * bump_alpha
    return 1.0


def apply_diffusion_filter(fn, kappa):
    '''One-time application of the diffusion filter.'''

    diffusion_filter = filter.make_diffusion_filter(fn, kappa)

    utility.update_parameters(
        diffusion_filter.parameters,
        config.parameters_linear_solver)

    # Apply filter on `fn`
    diffusion_filter.apply()


def diffusion_filter_weakform(fn, kappa):
    '''Return the variational form of the diffusion filter problem.

    Parameters
    ==========
    fn (dolfin.Function): Function to be smoothed (in-place)
    kappa (dolfin.Constant):Filter diffusivity constant.

    '''

    if not isinstance(kappa, dolfin.Constant):
        print('WARNING: Parameter `kappa` should be of type '
              '`dolfin.Constant` (doing type conversion)')
        kappa = dolfin.Constant(kappa)

    if float(kappa.values()) < 0:
        raise ValueError('Require kappa > 0')

    dx = dolfin.dx
    dot = dolfin.dot
    grad = dolfin.grad

    V = fn.function_space()
    v = dolfin.TestFunction(V)
    f = dolfin.TrialFunction(V)

    F = ( (f-fn)*v + kappa*dot(grad(f),grad(v)) ) * dx

    return F


def make_diffusion_filter(fn, kappa):
    '''As a minimization problem.

    Parameters
    ==========
    fn (dolfin.Function): Function to be smoothed (in-place)
    kappa (dolfin.Constant):Filter diffusivity constant.

    Important
    =========
    Aim to do as little smoothing as possible -- just enough to eliminate
    element-level oscillations in the total cost dissipation vector.

    `fn` is a `dolfin.Function` that will be filtered (smoothed) by calling
    the `solve()` method of the returned `dolfin.VariationalLinearSolver` object.

    For a given rough solution f0, find the smooth solution p that minimizes:
        J := 1/2 * ((f-f0)**2 + kappa*grad(f)**2) * dx

    The minimization problem is solved by solving the associated stationary
    (variational) problem:
        F := dJdp = 0

    No boundary conditions are required for the solution.

    Returns
    =======
    dolfin.LinearVariationalSolver : Call method `solve()` to do filtering

    '''

    F = diffusion_filter_weakform(fn, kappa)

    a = dolfin.lhs(F) # bilinear form, a(f,v)
    L = dolfin.rhs(F) # linear form, L(v)

    variational_problem = dolfin.LinearVariationalProblem(a, L, fn, bcs=[])
    variational_solver = dolfin.LinearVariationalSolver(variational_problem)

    class DiffusionFilter:
        '''Apply diffusion filter by calling method `apply()`.'''
        def __init__(self):
            self.parameters = variational_solver.parameters
            self.apply = variational_solver.solve

    return DiffusionFilter()
