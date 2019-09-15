
import dolfin
import logging
import numpy as np
import scipy.linalg as linalg

from . import optim
from . import filter
from . import config

logger = config.logger

EPS = 1e-12


class PeriodicExpression(dolfin.UserExpression):

    def __new__(cls, func, *args, **kwargs):

        if not isinstance(func, dolfin.Function):
            raise TypeError('Parameter `func` must be a `dolfin.Function`')

        if 'degree' not in kwargs and 'element' not in kwargs:
            raise TypeError('Require `degree` or `element` as keyword argument')

        self = super().__new__(cls)
        self._ufl_shape = func.ufl_shape

        return self

    def __init__(self, func, xlim, ylim, nx, ny, mirror_x=False,
                 mirror_y=False, overhang_fraction=0.0, **kwargs):

        # Must initialize base class
        super().__init__(**kwargs)

        self._func = func

        self.mirror_x = mirror_x
        self.mirror_y = mirror_y

        self._period_x = (xlim[1] - xlim[0]) / (nx-2*overhang_fraction)
        self._period_y = (ylim[1] - ylim[0]) / (ny-2*overhang_fraction)

        self._x0 = xlim[0] - self._period_x*overhang_fraction
        self._y0 = ylim[0] - self._period_y*overhang_fraction

        coord = func.function_space().mesh().coordinates()

        x0_loc, y0_loc = coord.min(axis=0)
        x1_loc, y1_loc = coord.max(axis=0)

        self._x0_loc = x0_loc
        self._y0_loc = y0_loc

        self._scale_x = (x1_loc - x0_loc) / (self._period_x*(1.0+EPS))
        self._scale_y = (y1_loc - y0_loc) / (self._period_y*(1.0+EPS))

    def __repr__(self):
        return f'<{self.__class__.__name__} at {hex(id(self))}>'

    def eval(self, value, x):

        y = x[1]
        x = x[0]

        num_periods_x = (x - self._x0) // self._period_x
        num_periods_y = (y - self._y0) // self._period_y

        if self.mirror_x and num_periods_x % 2:
            # Mirror about y-axis
            dx = ((self._x0 + (num_periods_x+1.0)*self._period_x) - x)
        else:
            dx = (x - (self._x0 + num_periods_x*self._period_x))

        if self.mirror_y and num_periods_y % 2:
            # Mirror about x-axis
            dy = ((self._y0 + (num_periods_y+1.0)*self._period_y) - y)
        else:
            dy = (y - (self._y0 + num_periods_y*self._period_y))

        x_loc = self._x0_loc + dx * self._scale_x
        y_loc = self._y0_loc + dy * self._scale_y

        value[:] = self._func(x_loc, y_loc)

    def value_shape(self):
        return self._ufl_shape


def project_function_periodically(func, nx, ny, V,
    mirror_x=False, mirror_y=False, overhang_fraction=0.0):
    '''

    Parameters
    ----------
    func : dolfin.Function
    V : dolfin.FunctionSpace

    Returns
    -------
    dolfin.Function

    '''

    if not isinstance(func, dolfin.Function):
        raise TypeError('Parameter `func` must be a `dolfin.Function`')

    if not isinstance(V, dolfin.FunctionSpace):
        raise TypeError('Parameter `V` must be a `dolfin.FunctionSpace`')

    V_func = func.function_space()
    coord = V_func.mesh().coordinates()

    x0, y0 = coord.min(axis=0)
    x1, y1 = coord.max(axis=0)

    expr = PeriodicExpression(func, [x0, x1], [y0, y1], nx, ny, mirror_x,
        mirror_y, overhang_fraction, degree=V_func.ufl_element().degree())

    return dolfin.project(expr, V)


def compute_maximal_compressive_stress_field(stress_field):
    '''
    Returns
    -------
    maximal_compressive_stress_field : dolfin.Function
        Scalar-valued function for the maximal compressive stress.

    '''

    s = stress_field

    if not isinstance(s, dolfin.Function) \
       or len(s.ufl_shape) != 2 or s.ufl_shape[0] != s.ufl_shape[1]:
        raise TypeError('Parameter `stress_field` must be a tensor `dolfin.Function`.')

    shape = s.ufl_shape
    size = np.prod(shape)

    V = s.function_space()
    S = dolfin.FunctionSpace(
        V.mesh(),
        V.ufl_element().family(),
        V.ufl_element().degree())

    dofs = -np.array(
        [linalg.eigvalsh(v.reshape(shape))[0] for v
        in s.vector().get_local().reshape((-1, size))])

    dofs[dofs<0.0] = 0.0 # Zero-out tensile values

    maximal_compressive_stress_field = dolfin.Function(S)
    maximal_compressive_stress_field.vector()[:] = dofs

    return maximal_compressive_stress_field


def compute_fraction_compressive_stress_field(stress_field):
    '''
    Returns
    -------
    fraction_compressive_stress_field : dolfin.Function
        Scalar-valued function whose value at a point is computed as
        `normL2(negative_eigenvalues) / normL2(all_eigenvalues)`.

    '''

    s = stress_field

    if not isinstance(s, dolfin.Function) \
       or len(s.ufl_shape) != 2 or s.ufl_shape[0] != s.ufl_shape[1]:
        raise TypeError('Parameter `stress_field` must be a tensor `dolfin.Function`.')

    shape = s.ufl_shape
    size = np.prod(shape)

    eigs = [linalg.eigvalsh(v.reshape(shape)) for v
        in s.vector().get_local().reshape((-1, size))]

    eignorms = np.sqrt([(v**2).sum() for v in eigs])
    eignorm_min = eignorms.max() * EPS
    eignorms[eignorms < eignorm_min] = eignorm_min

    V = s.function_space()
    S = dolfin.FunctionSpace(
        V.mesh(),
        V.ufl_element().family(),
        V.ufl_element().degree())

    dofs = np.sqrt([(v[v<0.0]**2).sum() for v in eigs]) / eignorms

    fraction_compressive_stress_field = dolfin.Function(S)
    fraction_compressive_stress_field.vector()[:] = dofs

    return fraction_compressive_stress_field


def solve_compliance_maximization_problem(
    W, R, u, p, bcs_u,
    defect_nucleation_centers,
    defect_nucleation_diameter,
    phasefield_collision_distance,
    phasefield_iteration_stepsize=1e-2,
    phasefield_fraction_increment=1e-2,
    phasefield_regularization_weight=0.450,
    phasefield_convergence_tolerance=None,
    phasefield_maximum_domain_fraction=1.0,
    iteration_state_recording_function=None):
    '''
    Parameters
    ----------
    W : dolfin.Form
        Potential energy of a hyperelastic solid, W(u(p), p).
    R : dolfin.Form
        Phasefield regularization, R(p).
    u : dolfin.Function
        Displacement function or a mixed function.
    p : dolfin.Function
        Phasefield function (scalar-valued).
    bcs_u : (list of) dolfin.DirichletBC(s), or None, or empty list
        Sequence of Dirichlet boundary conditions for `u`.

    '''

    INITIAL_PHASEFIELD_DIFFUSION = 1e-4
    MINIMUM_STOP_REQUESTS_FOR_TRIGGERING_STOP = 3

    MINIMUM_ITERATIONS_FOR_REQUESTING_STOP_WHEN_CONVERGED = \
        config.parameters_topology_solver['maximum_convergences'] + 3

    MINIMUM_ITERATIONS_FOR_REQUESTING_STOP_WHEN_DIVERGED = \
        config.parameters_topology_solver['maximum_divergences'] + 3

    energy_vs_iterations = []
    energy_vs_phasefield = []
    phasefield_fractions = []

    # Variational form of equilibrium
    F = dolfin.derivative(W, u)

    # Phasefield target fraction
    p_mean_target = dolfin.Constant(0.0)

    # Phasefield fraction constraint
    C = (p - p_mean_target) * dolfin.dx

    V_p = p.function_space()
    mesh = V_p.mesh()

    p_locals = make_defect_like_phasefield_array(
        V_p, defect_nucleation_centers, r=0.5*defect_nucleation_diameter)

    apply_diffusive_smoothing(p_locals, kappa=INITIAL_PHASEFIELD_DIFFUSION)
    apply_interval_bounds(p_locals, lower=0.0, upper=1.0)

    optimizer = optim.TopologyOptimizer(W, R, F, C, u, p, p_locals, bcs_u,
        function_to_call_during_iterations=iteration_state_recording_function)

    phasefield_fraction_i = dolfin.assemble(sum(p_locals)*dolfin.dx(mesh)) \
                          / dolfin.assemble(1*dolfin.dx(mesh))

    iterations_failed = False
    count_stop_requests = 0

    while phasefield_fraction_i < phasefield_maximum_domain_fraction:

        try:

            logger.info('Solving for phasefield domain fraction '
                        f'{phasefield_fraction_i:.3f}')

            p_mean_target.assign(phasefield_fraction_i)

            iterations_i, converged_i, energy_vs_iterations_i = optimizer.optimize(
                phasefield_iteration_stepsize, phasefield_regularization_weight,
                phasefield_collision_distance, phasefield_convergence_tolerance)

        except RuntimeError as exc:

            logger.error(str(exc))
            logger.error('Solver failed for domain fraction '
                         f'{phasefield_fraction_i:.3f}')

            iterations_failed = True

            break

        phasefield_fractions.append(phasefield_fraction_i)
        energy_vs_iterations.extend(energy_vs_iterations_i)
        energy_vs_phasefield.append(energy_vs_iterations_i[-1])

        if energy_vs_iterations_i[0] < energy_vs_iterations_i[-1]:
            logger.info('Energy did not decrease')
            count_stop_requests += 1

        elif converged_i and \
          iterations_i <= MINIMUM_ITERATIONS_FOR_REQUESTING_STOP_WHEN_CONVERGED:
            logger.info('Converged within a threshold number of iterations')
            count_stop_requests += 1

        elif not converged_i and \
          iterations_i <= MINIMUM_ITERATIONS_FOR_REQUESTING_STOP_WHEN_DIVERGED:
            logger.info('Diverged within a threshold number of iterations')
            count_stop_requests += 1

        if count_stop_requests == MINIMUM_STOP_REQUESTS_FOR_TRIGGERING_STOP:
            logger.info('Reached threshold number of consecutive stop requests')
            break

        phasefield_fraction_i += phasefield_fraction_increment

    else:
        logger.warning('Reached upper limit of phasefield domain fraction')

    return iterations_failed, energy_vs_iterations, \
           energy_vs_phasefield, phasefield_fractions, \
           optimizer, p_locals, p_mean_target


def apply_diffusive_smoothing(ps, kappa=1e-4):

    if isinstance(ps, (list, tuple)):
        return_as_sequence = True
    else:
        return_as_sequence = False
        ps = (ps,)

    if not all(isinstance(p_i, dolfin.Function) for p_i in ps):
        raise TypeError('Parameter `ps` must either be a `dolfin.Function` '
                        'or a sequence (list, tuple) of `dolfin.Function`s.')

    diffusion_filter = filter.DiffusionFilter(
        ps[0].function_space(), kappa)

    for p_i in ps:
        diffusion_filter.apply(p_i)

    return ps if return_as_sequence else ps[0]


def apply_interval_bounds(ps, lower=0.0, upper=1.0):

    if isinstance(ps, (list, tuple)):
        return_as_sequence = True
    else:
        return_as_sequence = False
        ps = (ps,)

    if not all(isinstance(p_i, dolfin.Function) for p_i in ps):
        raise TypeError('Parameter `ps` must either be a `dolfin.Function` '
                        'or a sequence (list, tuple) of `dolfin.Function`s.')

    for p_i in ps:

        x = p_i.vector().get_local()

        x[x < lower] = lower
        x[x > upper] = upper

        p_i.vector().set_local(x)

    return ps if return_as_sequence else ps[0]


def make_defect_like_phasefield(V, xc, r):

    if not isinstance(xc, np.ndarray):
        xc = np.array(xc, ndmin=1)

    if xc.ndim != 1 or len(xc) != 2:
        raise TypeError('Parameter `xc`')

    if not isinstance(V, dolfin.FunctionSpace):
        raise TypeError('Parameter `V`')

    p = dolfin.Function(V)
    p_arr = p.vector().get_local()

    x = V.tabulate_dof_coordinates()
    s = ((x-xc)**2).sum(axis=1)

    p_arr[s < r**2] = 1.0
    p.vector()[:] = p_arr

    return p


def make_defect_like_phasefield_array(V, xs, r):

    if not isinstance(xs, np.ndarray):
        xs = np.array(xs, ndmin=2)

    if xs.ndim != 2 or xs.shape[1] != 2:
        raise TypeError('Parameter `xs`')

    if not isinstance(V, dolfin.FunctionSpace):
        raise TypeError('Parameter `V`')

    ps = []

    for x_i in xs:
        ps.append(make_defect_like_phasefield(V, x_i, r))

    return ps


def meshgrid_uniform(xlim, ylim, nrow, ncol):

    x0, x1 = xlim
    y0, y1 = ylim

    x = np.linspace(x0, x1, ncol)
    y = np.linspace(y0, y1, nrow)

    x, y = np.meshgrid(x, y)

    x = x.reshape((-1,))
    y = y.reshape((-1,))

    return np.stack((x, y), axis=1)


def meshgrid_uniform_with_margin(xlim, ylim, nrow, ncol):

    margin_x = (xlim[1] - xlim[0]) / ncol / 2
    margin_y = (ylim[1] - ylim[0]) / nrow / 2

    xlim = [xlim[0] + margin_x, xlim[1] - margin_x]
    ylim = [ylim[0] + margin_y, ylim[1] - margin_y]

    return meshgrid_uniform(xlim, ylim, nrow, ncol)


def meshgrid_checker_symmetric(xlim, ylim, nrow, ncol):

    if not nrow % 2:
        raise ValueError('Require `nrow` to be odd.')

    if not ncol % 2:
        raise ValueError('Require `ncol` to be odd.')

    x0, x1 = xlim
    y0, y1 = ylim

    dx = (x1 - x0) / (ncol - 1)
    dy = (y1 - y0) / (nrow - 1)

    xlim_a = xlim
    ylim_a = ylim

    xlim_b = [xlim[0]+dx, xlim[1]-dx]
    ylim_b = [ylim[0]+dy, ylim[1]-dy]

    nrow_b = (nrow-1)//2
    ncol_b = (ncol-1)//2

    nrow_a = nrow_b + 1
    ncol_a = ncol_b + 1

    xs_a = meshgrid_uniform(xlim_a, ylim_a, nrow_a, ncol_a)
    xs_b = meshgrid_uniform(xlim_b, ylim_b, nrow_b, ncol_b)

    return np.concatenate((xs_a, xs_b), axis=0)


def pertub_gridrows(xs, nrow, ncol, dx, rowstep=2):

    if len(xs) != nrow*ncol:
        raise ValueError('Expected `len(xs) == nrow*ncol`')

    xs = xs.reshape((nrow,ncol,2)).copy()

    xs[0::rowstep,:] += [dx/2, 0.0]
    xs[1::rowstep,:] -= [dx/2, 0.0]

    return xs.reshape((-1,2))


def pertub_gridcols(xs, nrow, ncol, dy, colstep=2):

    if len(xs) != nrow*ncol:
        raise ValueError('Expected `len(xs) == nrow*ncol`')

    xs = xs.reshape((nrow,ncol,2)).transpose((1,0,2)).copy()

    xs[0::colstep,:] += [0.0, dy/2]
    xs[1::colstep,:] -= [0.0, dy/2]

    return xs.reshape((-1,2))


def points_inside_rectangle(xs, xlim, ylim):

    if not isinstance(xs, np.ndarray) or xs.ndim != 2:
        raise TypeError('Parameter `xs` must be a 2D numpy array.')

    if not hasattr(xlim, '__len__') or len(xlim) != 2:
        raise TypeError('Parameter `xlim` must have length 2.')

    if not hasattr(ylim, '__len__') or len(ylim) != 2:
        raise TypeError('Parameter `ylim` must have length 2.')

    mask = (xs[:,0] > xlim[0]) & \
           (xs[:,0] < xlim[1]) & \
           (xs[:,1] > ylim[0]) & \
           (xs[:,1] < ylim[1])

    return np.array(xs[mask])


def make_parameter_combinations(*parameters):
    '''Return all combinations of `parameters`.'''

    if len(parameters) == 0:
        return []
    elif len(parameters) == 1:
        y = parameters[0]
        if not isinstance(y, list):
            y = [y]
        return y

    y = parameters[-1]

    if isinstance(y, list):
        y = [(yi,) for yi in y]
    else:
        y = [(y,)]

    def outer(x, y):

        assert isinstance(y, list)
        assert all(isinstance(yi, tuple) for yi in y)

        if not isinstance(x, list):
            x = [x]

        nx = len(x)
        ny = len(y)

        X = []
        Y = y * nx

        for xi in x:
            X += [xi]*ny

        return [(Xi,)+Yi for Xi, Yi in zip(X, Y)]

    for x in parameters[-2::-1]:
        y = outer(x, y)

    return y
