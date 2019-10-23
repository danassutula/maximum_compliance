
import os
import dolfin
import numpy as np
import scipy.linalg as linalg
import matplotlib.pyplot as plt

from dolfin import Constant
from dolfin import Function
from dolfin import assemble
from dolfin import dx

from optim import optim
from optim import filter
from optim import config

logger = config.logger

EPS = 1e-12


def solve_compliance_maximization_problem(
    W, P, p, equilibrium_solve,
    defect_nucleation_centers,
    defect_nucleation_diameter,
    phasefield_penalty_weight,
    phasefield_collision_distance,
    phasefield_iteration_stepsize,
    phasefield_fraction_increment,
    minimum_phasefield_fraction,
    maximum_phasefield_fraction,
    minimum_energy_threshold,
    function_to_call_at_each_phasefield_iteration=None,
    function_to_call_at_each_phasefield_fraction=None):
    '''
    Parameters
    ----------
    W : dolfin.Form
        Energy (`W(u(p), p)`) to be minimized.
    P : dolfin.Form
        Phasefield penalty (`P(p)`) to be minimized.
    p : dolfin.Function
        Phasefield function (scalar-valued).

    '''

    if function_to_call_at_each_phasefield_fraction is None:
        function_to_call_at_each_phasefield_fraction = lambda : None
    elif not callable(function_to_call_at_each_phasefield_fraction):
        raise TypeError('Parameter `function_to_call_at_each_phasefield_fraction` '
                        'must be callable without any arguments')

    energy_vs_iterations = []
    energy_vs_phasefield = []
    phasefield_fractions = []

    # Phasefield target fraction
    p_mean_target = Constant(0.0)

    # Phasefield fraction constraint
    C = (p - p_mean_target) * dx

    V_p = p.function_space()

    p_locals = make_defect_like_phasefield_array(V_p,
        defect_nucleation_centers, r=0.5*defect_nucleation_diameter)

    p.assign(sum(p_locals))

    optimizer = optim.TopologyOptimizer(W, P, C, p, p_locals, equilibrium_solve,
        equilibrium_write=function_to_call_at_each_phasefield_iteration)

    iterations_failed = False

    try:

        phasefield_fraction_i = max(minimum_phasefield_fraction,
            assemble(p*dx) / assemble(1*dx(V_p.mesh())))

        while phasefield_fraction_i < maximum_phasefield_fraction:

            try:

                logger.info('Solve for phasefield domain fraction '
                            f'{phasefield_fraction_i:.3f}')

                p_mean_target.assign(phasefield_fraction_i)

                _, energy_vs_iterations_i = optimizer.optimize(
                    phasefield_iteration_stepsize, phasefield_penalty_weight,
                    phasefield_collision_distance)

            except RuntimeError as exc:

                logger.error(str(exc))
                logger.error('Solver failed for domain fraction '
                             f'{phasefield_fraction_i:.3f}')

                iterations_failed = True

                break

            phasefield_fractions.append(phasefield_fraction_i)
            energy_vs_iterations.extend(energy_vs_iterations_i)
            energy_vs_phasefield.append(energy_vs_iterations_i[-1])

            function_to_call_at_each_phasefield_fraction()

            if energy_vs_phasefield[-1] < minimum_energy_threshold:
                logger.info('Energy converged within threshold')
                break

            phasefield_fraction_i += phasefield_fraction_increment

        else:
            logger.info('Reached phasefield domain fraction limit')

    except KeyboardInterrupt:
        logger.info('Received a keyboard interrupt')

    return iterations_failed, energy_vs_iterations, \
           energy_vs_phasefield, phasefield_fractions, \
           optimizer, p_locals, p_mean_target


def equilibrium_solver(F, u, bcs, bcs_set_values, bcs_values):
    '''Nonlinear solver for the hyper-elastic equilibrium problem.

    F : dolfin.Form
<<<<<<< HEAD
        Variational form of equilibrium.
=======
        Variational form of equilibrium, i.e. F(u;v)==0 forall v. Usually `F`
        is obtained by taking the derivative of the potential energy `W`, e.g.
        `F = dolfin.derivative(W, u)`
>>>>>>> dev
    u : dolfin.Function
        Displacement function (or a mixed field function).
    bcs : (sequence of) dolfin.DirichletBC's
        Dirichlet boundary conditions.
    bcs_set_values : function
        Called with elements of `bcs_values`.
    bcs_values : numpy.ndarray (2D)
        Sequence of displacement values.

    '''

    if not isinstance(bcs_values, np.ndarray) or bcs_values.ndim != 2:
        raise TypeError('Parameter `bcs_values` must be a 2D `numpy.ndarray`')

    try:
        bcs_set_values(bcs_values[-1])
    except:
        logger.error('Unable to set Dirichlet boundary conditions')
        raise

    dFdu = dolfin.derivative(F, u)

    nonlinear_solver = dolfin.NonlinearVariationalSolver(
        dolfin.NonlinearVariationalProblem(F, u, bcs, dFdu))

<<<<<<< HEAD
    u_arr_backup = np.zeros((u.vector().size(),), float)

    def equilibrium_solve():
        try:
            nonlinear_solver.solve()
        except RuntimeError:
            logger.error('Could not solve equilibrium problem; '
                         'Trying to re-load incrementally.')

            nonlocal bcs_values
            u.vector()[:] = 0.0
=======
    update_parameters(nonlinear_solver.parameters,
                      config.parameters_nonlinear_solver)

    def equilibrium_solve(incremental=False):

        if incremental:

            nonlocal bcs_values
            u.vector()[:] = 0.0
            u_arr_backup = None
>>>>>>> dev

            try:
                for i, values_i in enumerate(bcs_values):
                    logger.info(f'Solving for load {values_i}')

                    bcs_set_values(values_i)
                    nonlinear_solver.solve()

<<<<<<< HEAD
                    u_arr_backup[:] = u.vector().get_local()

            except:
                logger.error('Could not solve equilibrium problem for load '
                             f'{values_i}; assuming previous load value.')

                u.vector()[:] = u_arr_backup
                bcs_values = bcs_values[:i]

    return equilibrium_solve


def update_parameters(cls, target, source):
=======
                    u_arr_backup = u.vector().get_local()

            except RuntimeError:
                logger.error('Could not solve equilibrium problem for load '
                             f'{values_i}; assuming previous load value.')

                if u_arr_backup is None:
                    raise RuntimeError('Previous load value is not available')

                u.vector()[:] = u_arr_backup
                bcs_values = bcs_values[:i]

        else:

            try:
                nonlinear_solver.solve()
            except RuntimeError:
                logger.error('Could not solve equilibrium problem; '
                             'Trying to re-load incrementally.')

                equilibrium_solve(incremental=True)

    return equilibrium_solve


def update_parameters(target, source):
>>>>>>> dev
    '''Update dict-like `target` with dict-like `source`.'''

    for k in source.keys():

        if k not in target.keys():
            raise KeyError(k)

        if hasattr(target[k], 'keys'):

            if not hasattr(source[k], 'keys'):
                raise TypeError(f'`source[{k}]` must be dict-like')
            else:
<<<<<<< HEAD
                cls._update_parameters(target[k], source[k])
=======
                update_parameters(target[k], source[k])
>>>>>>> dev

        elif hasattr(source[k], 'keys'):
            raise TypeError(f'`source[{k}]` can not be dict-like')

        else:
            target[k] = source[k]


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


### File I/O

class FunctionWriter:

    def __init__(self, outdir, func, name, writing_period=1,
                 write_pvd=True, write_npy=True):

        if not isinstance(func, Function):
            raise TypeError('Parameter `func` must be a `dolfin.Function`')

        if not isinstance(name, str):
            raise TypeError('Parameter `name` must be a `str`')

        if not os.path.exists(outdir):
            os.makedirs(outdir)

        self.func = func
        self.writing_period = writing_period

        self._calls_count = 0
        self._index_write = 0

        self._outfile_pvd = dolfin.File(
            os.path.join(outdir, f'{name}.pvd'))

        self._outfile_npy_format = os.path.join(
            outdir, f'{name}'+'{:06d}'+'.npy').format

        if write_pvd and write_npy:
            def write():

                self._outfile_pvd << self.func

                np.save(self._outfile_npy_format(self._index_write),
                        self.func.vector().get_local())

                self._calls_count += 1
                self._index_write += 1

        elif write_pvd:
            def write():

                self._outfile_pvd << self.func

                self._calls_count += 1
                self._index_write += 1

        elif write_npy:
            def write():

                np.save(self._outfile_npy_format(self._index_write),
                        self.func.vector().get_local())

                self._calls_count += 1
                self._index_write += 1

        else:
            raise ValueError('Parameters `write_pvd` and `write_npy`; '
                             'require at lease one to be `True`.')

        self.write = write

    def periodic_write(self):

        if self._calls_count % self.writing_period:
            self._calls_count += 1; return

        self.write()

    # def __del__(self):
    #
    #     if hasattr(self._outfile_pvd, 'close'):
    #         self._outfile_pvd.close()


def remove_outfiles(subdir, file_extensions):

    if not isinstance(file_extensions, (list, tuple)):
        file_extensions = (file_extensions,)

    if not all(isinstance(ext, str) for ext in file_extensions):
        raise ValueError('Parameter `file_extensions` must be '
                         'a (`list` or `tuple` of) `str`(s).')

    file_extensions = tuple(ext if ext.startswith('.')
        else ('.' + ext) for ext in file_extensions)

    for item in os.listdir(subdir):
        item = os.path.join(subdir, item)

        if os.path.isfile(item):
            _, ext = os.path.splitext(item)

            if ext in file_extensions:
                os.remove(item)


def extract_substring(filename, str_beg, str_end):

    pos_beg = filename.find(str_beg)

    if pos_beg < 0:
        raise IndexError(f'Could not find str_beg="{str_beg}"')

    pos_beg += len(str_beg)
    pos_end = filename.find(str_end, pos_beg)

    if pos_end < 0:
        raise IndexError(f'Could not find str_end="{str_end}"')

    return filename[pos_beg:pos_end]


### Mesh

def rectangle_mesh(p0, p1, nx, ny, diagonal="left/right"):
    '''
    Parameters
    ----------
    nx: int
        Number of cells along x-axis.
    ny (optional): int
        Number of cells along y-axis.
    diagonal: str
        Possible options: "left/right", "crossed", "left", or "right".

    '''

    return dolfin.RectangleMesh(dolfin.Point(p0), dolfin.Point(p1), nx, ny, diagonal)


def boundaries_of_rectangle_mesh(mesh):

    x0, y0 = mesh.coordinates().min(0)
    x1, y1 = mesh.coordinates().max(0)

    atol_x = (x1 - x0) * EPS
    atol_y = (y1 - y0) * EPS

    bot = dolfin.CompiledSubDomain('x[1] < y0 && on_boundary', y0=y0+atol_y)
    rhs = dolfin.CompiledSubDomain('x[0] > x1 && on_boundary', x1=x1-atol_x)
    top = dolfin.CompiledSubDomain('x[1] > y1 && on_boundary', y1=y1-atol_y)
    lhs = dolfin.CompiledSubDomain('x[0] < x0 && on_boundary', x0=x0+atol_x)

    return bot, rhs, top, lhs


### BC's

def uniform_extension_bcs(V):
    '''

    Parameters
    ----------
    V: dolfin.FunctionSpace
        Vector function space for the displacement field.

    '''

    mesh = V.mesh()

    boundary_bot, boundary_rhs, boundary_top, boundary_lhs = \
        boundaries_of_rectangle_mesh(mesh)

    uy_bot = Constant(0.0)
    ux_rhs = Constant(0.0)
    uy_top = Constant(0.0)
    ux_lhs = Constant(0.0)

    bcs = [
        dolfin.DirichletBC(V.sub(1), uy_bot, boundary_bot),
        dolfin.DirichletBC(V.sub(0), ux_rhs, boundary_rhs),
        dolfin.DirichletBC(V.sub(1), uy_top, boundary_top),
        dolfin.DirichletBC(V.sub(0), ux_lhs, boundary_lhs),
        ]

    def bcs_set_values(ux, uy=None):
        if uy is None: ux, uy = ux
        uy_bot.assign(-uy)
        ux_rhs.assign( ux)
        uy_top.assign( uy)
        ux_lhs.assign(-ux)

    return bcs, bcs_set_values


### Defect Initialization

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


### Post-processing

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

        '''
        Parameters
        ----------
        func : dolfin.Function
            The function to be tiled along x/y-axes.
        xlim : sequence
            The x-axis limits of the expression.
        ylim : sequence
            The y-axis limits of the expression.
        nx : int
            Number of function tiles along x-axis.
        ny : int
            Number of function tiles along y-axis.

        '''

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

    coord = V.mesh().coordinates()

    x0, y0 = coord.min(axis=0)
    x1, y1 = coord.max(axis=0)

    expr = PeriodicExpression(func, [x0, x1], [y0, y1],
        nx, ny, mirror_x, mirror_y, overhang_fraction,
        degree=func.function_space().ufl_element().degree())

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

    V = s.function_space()
    S = dolfin.FunctionSpace(
        V.mesh(),
        V.ufl_element().family(),
        V.ufl_element().degree())

    fraction_compressive_stress_field = dolfin.Function(S)

    if eignorm_min > 0.0:
        eignorms[eignorms < eignorm_min] = eignorm_min
        dofs = np.sqrt([(v[v<0.0]**2).sum() for v in eigs]) / eignorms
        fraction_compressive_stress_field.vector()[:] = dofs

    return fraction_compressive_stress_field


### Plotting

def plot_energy_vs_iterations(energy_vs_iteration,
                              figname="energy_vs_iterations",
                              ylabel='Energy', semilogy=False):

    fh = plt.figure(figname)
    fh.clear(); ah=fh.subplots()

    if semilogy:
        ah.semilogy(energy_vs_iteration, '-')
    else:
        ah.plot(energy_vs_iteration, '-')

    ah.set_ylabel(ylabel)
    ah.set_xlabel('Iteration number')

    fh.tight_layout()
    fh.show()

    return fh, figname


def plot_energy_vs_phasefields(energy_vs_phasefield,
                               phasefield_fractions,
                               figname="energy_vs_phasefield",
                               ylabel='Energy', semilogy=False):

    fh = plt.figure(figname)
    fh.clear(); ah=fh.subplots()

    if semilogy:
        ah.semilogy(phasefield_fractions, energy_vs_phasefield, '-')
    else:
        ah.plot(phasefield_fractions, energy_vs_phasefield, '-')

    ah.set_ylabel(ylabel)
    ah.set_xlabel('Phasefield fraction')

    fh.tight_layout()
    fh.show()

    return fh, figname


def plot_phasefiled(p, figname="phasefield"):

    fh = plt.figure(figname)
    fh.clear(); ah=fh.subplots()

    dolfin.plot(p)

    plt.title('Phasefield, $p$\n('
              + r'$p_\mathrm{min}$ = '
              + f'{p.vector().get_local().min():.3f}, '
              + r'$p_\mathrm{max}$ = '
              + f'{p.vector().get_local().max():.3f})')

    fh.tight_layout()
    fh.show()

    return fh, figname


def plot_material_fraction(m, figname="material_fraction"):

    fh = plt.figure(figname)
    fh.clear(); ah=fh.subplots()

    dolfin.plot(m)

    plt.title('Material fraction, $m$\n('
              + r'$m_\mathrm{min}$ = '
              + f'{m.vector().get_local().min():.3f}, '
              + r'$m_\mathrm{max}$ = '
              + f'{m.vector().get_local().max():.3f})')

    fh.tight_layout()
    fh.show()

    return fh, figname
