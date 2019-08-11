
import dolfin
import logging
import numpy as np
import scipy.linalg as linalg

from . import optim
from . import filter
from . import config

logger = config.logger

EPS = 1e-12


def project_material_fraction(expr, V, kappa=1e-5):
    '''Project material fraction expression. Then, apply some smoothing,
    and finally enforce the lower bound (0.0) and upper bound (1.0) values.

    Parameters
    ----------
    expr : dolfin.Expression-like
        Scalar-valued expression that can be projected on `V`.
    V : dolfin.FunctionSpace
        Scalar-valued function space.

    Returns
    -------
    func : dolfin.Function
        Projected expression for the material fraction.

    '''

    if not hasattr(expr, 'ufl_shape') or expr.ufl_shape != ():
        raise TypeError('Parameter `expr` must be a scalar-valued expression.')

    if not isinstance(V, dolfin.FunctionSpace):
        raise TypeError('Parameter `V` must be a `dolfin.FunctionSpace`.')

    func = dolfin.project(expr, V)
    apply_diffusive_smoothing(func, kappa)
    apply_interval_bounds(func, 0.0, 1.0)

    return func


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

    MINIMUM_STOP_REQUESTS_FOR_TRIGGERING_STOP = 3

    MINIMUM_ITERATIONS_FOR_REQUESTING_STOP_WHEN_CONVERGED = \
        config.parameters_topology_solver['minimum_convergences'] + 2

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

    apply_diffusive_smoothing(p_locals, kappa=1e-4)
    apply_interval_bounds(p_locals, lower=0.0, upper=1.0)

    optimizer = optim.TopologyOptimizer(W, R, F, C, u, p, p_locals, bcs_u,
        function_to_call_during_iterations=iteration_state_recording_function)

    phasefield_fraction_i = dolfin.assemble(sum(p_locals)*dolfin.dx(mesh)) \
                          / dolfin.assemble(1*dolfin.dx(mesh))

    iterations_failed = False
    count_stop_requests = 0

    while True:

        try:

            logger.info('Solving for phasefield domain fraction '
                        f'{phasefield_fraction_i:.3f}')

            p_mean_target.assign(phasefield_fraction_i)

            iterations_i, converged_i, energy_vs_iterations_i = optimizer.optimize(
                phasefield_iteration_stepsize, phasefield_regularization_weight,
                phasefield_collision_distance, phasefield_convergence_tolerance)

        except RuntimeError:

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
            logger.info('Convergence within a threshold number of iterations')
            count_stop_requests += 1

        elif not converged_i and \
          iterations_i <= MINIMUM_ITERATIONS_FOR_REQUESTING_STOP_WHEN_DIVERGED:
            logger.info('Divergence within a threshold number of iterations')
            count_stop_requests += 1

        else:
            count_stop_requests = 0

        if count_stop_requests == MINIMUM_STOP_REQUESTS_FOR_TRIGGERING_STOP:
            logger.info('Reached threshold number of consecutive stop requests')
            break

        phasefield_fraction_i += phasefield_fraction_increment

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
