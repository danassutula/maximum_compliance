
import math
import dolfin
import numpy as np

from dolfin import Constant
from dolfin import Function
from dolfin import interpolate
from dolfin import grad, dot, dx

from . import config
logger = config.logger

EPS = 1e-12


def variational_distance_solver(phasefield_functions):

    if isinstance(phasefield_functions, tuple):
        return_distances_as_tuple = True
    elif isinstance(phasefield_functions, list):
        phasefield_functions = tuple(phasefield_functions)
        return_distances_as_tuple = True
    else:
        phasefield_functions = (phasefield_functions,)
        return_distances_as_tuple = False

    if not all(isinstance(func_i, Function) for func_i in phasefield_functions):
        raise TypeError('Parameter `phasefield_functions` must contain `dolfin.Function`s')

    threshold = config.parameters_distance_solver['variational_solver']['threshold']
    viscosity = config.parameters_distance_solver['variational_solver']['viscosity']
    penalty   = config.parameters_distance_solver['variational_solver']['penalty']

    if not (isinstance(threshold, (float, int)) and 0.0 < threshold < 1.0):
        raise ValueError('`config.parameters_distance_solver'
                         '[\'variational_solver\'][\'threshold\']`')

    if not (isinstance(viscosity, (float, int)) and viscosity > 0.0):
        raise ValueError('`config.parameters_distance_solver'
                         '[\'variational_solver\'][\'viscosity\']`')

    if not (isinstance(penalty, (float, int)) and penalty > 0.0):
        raise ValueError('`config.parameters_distance_solver'
                         '[\'variational_solver\'][\'penalty\']`')

    V = phasefield_functions[0].function_space()

    _distance_functions = tuple(Function(V)
        for _ in range(len(phasefield_functions)))

    _distance_functions_arr = np.zeros(
        (len(_distance_functions), V.dim()))

    solver = VariationalDistanceSolver(V, viscosity, penalty)
    solution_vector = solver.get_distance_function().vector()
    compute_distance = solver.compute_distance_function

    def solve_distances():
        for p_i, d_i, d_arr_i, in zip(phasefield_functions,
                _distance_functions, _distance_functions_arr):

            solution_vector[:] = d_i.vector()       # Reusing previous solution
            compute_distance(p_i, threshold, False) # Without re-initialization

            d_i.vector()[:] = solution_vector
            d_arr_i[:] = solution_vector.get_local()

    def solve_distances_init():
        for p_i, d_i, d_arr_i, in zip(phasefield_functions,
                _distance_functions, _distance_functions_arr):

            solver._mark_zero_distance_subdomain(p_i, threshold)
            solver._solve_initdist_problem() # Estimate solution

            d_i.vector()[:] = solution_vector
            d_arr_i[:] = solution_vector.get_local()

    solve_distances_init() # Okey to initialize here

    if not return_distances_as_tuple:
        distance_functions = _distance_functions[0]
        distance_functions_arr = _distance_functions_arr[0]
        assert distance_functions_arr.flags['OWNDATA'] == False
    else:
        distance_functions = _distance_functions
        distance_functions_arr = _distance_functions_arr

    return solve_distances, distance_functions, distance_functions_arr


def algebraic_distance_solver(phasefield_functions):

    if isinstance(phasefield_functions, tuple):
        return_distances_as_tuple = True
    elif isinstance(phasefield_functions, list):
        phasefield_functions = tuple(phasefield_functions)
        return_distances_as_tuple = True
    else:
        phasefield_functions = (phasefield_functions,)
        return_distances_as_tuple = False

    if not all(isinstance(func_i, Function) for func_i in phasefield_functions):
        raise TypeError('Parameter `phasefield_functions` must contain `dolfin.Function`s')

    threshold = config.parameters_distance_solver['algebraic_solver']['threshold']

    if not (isinstance(threshold, (float, int)) and 0.0 < threshold < 1.0):
        raise ValueError('`config.parameters_distance_solver'
                         '[\'variational_solver\'][\'threshold\']`')

    V = phasefield_functions[0].function_space()

    mesh = V.mesh()
    he = mesh.hmin()
    xs = mesh.coordinates()
    p0 = xs.min(axis=0)
    p1 = xs.max(axis=0)
    if len(p1) == 2:
        L, H = p1 - p0
        hx = hy = he
        nx = int(round(L/hx))
        ny = int(round(H/hy))
        nz = None
    elif len(p1) == 3:
        L, H, W = p1 - p0
        hx = hy = hz = he
        nx = int(round(L/hx))
        ny = int(round(H/hy))
        nz = int(round(W/hz))
    else:
        raise RuntimeError

    _distance_functions = tuple(Function(V)
        for _ in range(len(phasefield_functions)))

    _distance_functions_arr = np.zeros(
        (len(_distance_functions), V.dim()))

    solver = AlgebraicDistanceSolver(p0, p1, nx, ny, nz)
    compute_distance = solver.compute_distance_function

    def solve_distances():
        for p_i, d_i, d_arr_i, in zip(phasefield_functions,
                _distance_functions, _distance_functions_arr):

            compute_distance(p_i, threshold, d_i)
            d_arr_i[:] = d_i.vector().get_local()

    if not return_distances_as_tuple:
        distance_functions = _distance_functions[0]
        distance_functions_arr = _distance_functions_arr[0]
        assert distance_functions_arr.flags['OWNDATA'] == False
    else:
        distance_functions = _distance_functions
        distance_functions_arr = _distance_functions_arr

    return solve_distances, distance_functions, distance_functions_arr


def fast_marching_method(phasefield_functions):

    if isinstance(phasefield_functions, tuple):
        return_distances_as_tuple = True
    elif isinstance(phasefield_functions, list):
        phasefield_functions = tuple(phasefield_functions)
        return_distances_as_tuple = True
    else:
        phasefield_functions = (phasefield_functions,)
        return_distances_as_tuple = False

    if not all(isinstance(func_i, Function) for func_i in phasefield_functions):
        raise TypeError('Parameter `phasefield_functions` must contain `dolfin.Function`s')

    threshold = config.parameters_distance_solver['algebraic_solver']['threshold']

    if not (isinstance(threshold, (float, int)) and 0.0 < threshold < 1.0):
        raise ValueError('`config.parameters_distance_solver'
                         '[\'variational_solver\'][\'threshold\']`')

    V = phasefield_functions[0].function_space()

    mesh = V.mesh()
    xs = mesh.coordinates()
    p0 = xs.min(axis=0)
    p1 = xs.max(axis=0)

    _distance_functions = tuple(Function(V)
        for _ in range(len(phasefield_functions)))

    _distance_functions_arr = np.zeros(
        (len(_distance_functions), V.dim()))

    solver = FastMarchingMethod(p0, p1, stepsize=mesh.hmin())
    compute_distance = solver.compute_distance_function

    def solve_distances():
        for p_i, d_i, d_arr_i, in zip(phasefield_functions,
                _distance_functions, _distance_functions_arr):

            compute_distance(p_i, threshold, d_i)
            d_arr_i[:] = d_i.vector().get_local()

    if not return_distances_as_tuple:
        distance_functions = _distance_functions[0]
        distance_functions_arr = _distance_functions_arr[0]
        assert distance_functions_arr.flags['OWNDATA'] == False
    else:
        distance_functions = _distance_functions
        distance_functions_arr = _distance_functions_arr

    return solve_distances, distance_functions, distance_functions_arr


class VariationalDistanceSolver:

    def __init__(self, V, viscosity=1e-2, penalty=1e5):
        '''
        Parameters
        ----------
        V : dolfin.FunctionSpace
            Function space for the distance function.
        viscosity: float or dolfin.Constant
            Stabilization for the unique solution of the distance problem.
        penalty: float or dolfin.Constant
            Penalty for weakly enforcing the zero-distance boundary conditions.

        '''

        if not isinstance(V, dolfin.FunctionSpace):
            raise TypeError('Parameter `V` must be a `dolfin.FunctionSpace`')

        if not isinstance(viscosity, Constant):
            if not isinstance(viscosity, (float, int)):
                raise TypeError('`Parameter `viscosity`')
            viscosity = Constant(viscosity)

        if not isinstance(penalty, Constant):
            if not isinstance(penalty, (float, int)):
                raise TypeError('Parameter `penalty`')
            penalty = Constant(penalty)

        self._viscosity = viscosity
        self._penalty = penalty

        mesh = V.mesh()
        xs = mesh.coordinates()
        l0 = (xs.max(0)-xs.min(0)).min()

        self._d = d = Function(V)
        self._Q = dolfin.FunctionSpace(mesh, 'DG', 0)

        self._mf = dolfin.MeshFunction('size_t', mesh, mesh.geometric_dimension())
        self._dx_penalty = dx(subdomain_id=1, subdomain_data=self._mf, domain=mesh)

        v0 = dolfin.TestFunction(V)
        v1 = dolfin.TrialFunction(V)

        target_gradient = Constant(1.0)
        scaled_penalty = penalty/mesh.hmax()

        lhs_F0 = l0*dot(grad(v0), grad(v1))*dx \
            + scaled_penalty*v0*v1*self._dx_penalty

        rhs_F0 = v0*target_gradient*dx

        problem = dolfin.LinearVariationalProblem(lhs_F0, rhs_F0, d)
        self._linear_solver = dolfin.LinearVariationalSolver(problem)
        self._linear_solver.parameters["symmetric"] = True

        F = v0*(grad(d)**2-target_gradient)*dx \
            + viscosity*l0*dot(grad(v0), grad(d))*dx \
            + scaled_penalty*v0*d*self._dx_penalty

        J = dolfin.derivative(F, d, v1)

        problem = dolfin.NonlinearVariationalProblem(F, d, bcs=None, J=J)
        self._nonlinear_solver = dolfin.NonlinearVariationalSolver(problem)
        self._nonlinear_solver.parameters['nonlinear_solver'] = 'newton'
        self._nonlinear_solver.parameters['symmetric'] = False

        self._solve_initdist_problem = self._linear_solver.solve
        self._solve_distance_problem = self._nonlinear_solver.solve

    def _mark_zero_distance_subdomain(self, func, threshold):
        '''Mark the subdomain that defines the pseudo zero-distance boundy.'''

        self._mf.array()[:] = np.array(
            interpolate(func, self._Q).vector()
            .get_local() > threshold, np.uint)

        if not self._mf.array().any():
            raise RuntimeError('Could not mark any cells inside the '
                               'domain given the value of threshold')

    def compute_distance_function(self, func, threshold, init=True):
        '''Compute the distance function to the thresholded function.'''

        self._mark_zero_distance_subdomain(func, threshold)

        if init:

            self._solve_initdist_problem()

            try:
                self._solve_distance_problem()
            except RuntimeError:
                logger.error('Unable to solve distance problem')
                raise

        else:

            try:
                self._solve_distance_problem()
            except RuntimeError:
                logger.warning('Could not solve distance problem; '
                               're-initializing and trying again.')

                self._solve_initdist_problem()

                try:
                    self._solve_distance_problem()
                except RuntimeError:
                    logger.error('Unable to solve distance problem')
                    raise

        return self._d

    def get_distance_function(self):
        return self._d

    def set_viscosity(self, value):
        self._viscosity.assign(value)

    def set_penalty(self, value):
        self._penalty.assign(value)


class AlgebraicDistanceSolver:

    def __init__(self, p0, p1, num_cells_x, num_cells_y, num_cells_z=None):

        if not isinstance(p0, np.ndarray): p0 = np.array(p0)
        if not isinstance(p1, np.ndarray): p1 = np.array(p1)

        if num_cells_z is None:

            dim = 2

            if not (len(p0) == len(p1) == dim):
                raise RuntimeError('Expected parameters `p0` and `p1` to be '
                                   'points in 2D space since `nz` was not given')

            mesh = dolfin.RectangleMesh(
                dolfin.Point(p0), dolfin.Point(p1), num_cells_x, num_cells_y)

            self._V = dolfin.FunctionSpace(mesh, 'CG', 1)

            L, H = p1 - p0

            self._nx = nx = num_cells_x + 1
            self._ny = ny = num_cells_y + 1
            self._nz = None

            self._hx = hx = L / num_cells_x
            self._hy = hy = H / num_cells_y
            self._hz = None

            self._nn = nx * ny
            self._nxny = self._nn

            xs = np.linspace(p0[0], p1[0], nx)
            ys = np.linspace(p0[1], p1[1], ny)

            xs_grid = np.array(sum([xs.tolist(),] * ny, []))
            ys_grid = np.array(sum([[yi,] * nx for yi in ys], []))

            pt_dofs = self._V.tabulate_dof_coordinates()

            self._index_dof_to_pos = np.array(
                [yi/hy*nx + xi/hx + 0.5 for xi, yi in pt_dofs-p0], np.uint)

            self._index_pos_to_subs = np.array(
                [(pos%nx, pos//nx) for pos in range(self._nn)])

            self._update_candidates = self.__update_candidates_2d
            self._update_distances = self.__update_distances_2d

        else:

            dim = 3

            if not (len(p0) == len(p1) == dim):
                raise RuntimeError('Expected parameters `p0` and `p1` to be '
                                   'points in 3D space since `nz` was given')

            mesh = dolfin.BoxMesh(dolfin.Point(p0), dolfin.Point(p1),
                                  num_cells_x, num_cells_y, num_cells_z)

            self._V = dolfin.FunctionSpace(mesh, 'CG', 1)

            L, H, W = p1 - p0

            self._hx = hx = L / num_cells_x
            self._hy = hy = H / num_cells_y
            self._hz = hz = W / num_cells_z

            self._nx = nx = num_cells_x + 1
            self._ny = ny = num_cells_y + 1
            self._nz = nz = num_cells_z + 1

            self._nn = nx * ny * nz;
            self._nxny = nxny = nx*ny

            xs = np.linspace(p0[0], p1[0], nx)
            ys = np.linspace(p0[1], p1[1], ny)
            zs = np.linspace(p0[2], p1[2], nz)

            xs_grid = np.array(sum([xs.tolist(),] * (ny*nz), []))
            ys_grid = np.array(sum([[yi,] * nx for zi in zs for yi in ys], []))
            zs_grid = np.array(sum([[zi,] * nxny for zi in zs], []))

            pt_dofs = self._V.tabulate_dof_coordinates()

            self._index_dof_to_pos = np.array(
                [zi/hz*nxny + yi/hy*nx + xi/hx + 0.5
                for xi, yi, zi in pt_dofs-p0], np.uint)

            self._index_pos_to_subs = np.array(
                [(pos%nxny%nx, pos%nxny//nx, pos//nxny)
                for pos in range(self._nn)])

            self._update_candidates = self.__update_candidates_3d
            self._update_distances = self.__update_distances_3d


    def compute_distance_function(self, func, threshold,
            out=None, V_out=None, maximum_distance=None):

        if out is not None:
            if V_out is not None:
                logger.warning('Parameter `V_out` is redundant '
                               'since parameter `out` is given')
            V_out = out.function_space()

        elif V_out is None:
            V_out = func.function_space()

        require_interpolation = \
            not self._equivalent_function_spaces(V_out, self._V)

        if require_interpolation:
            func = interpolate(func, self._V)

        dof_index = np.flatnonzero(func.vector().get_local() > threshold)

        if not dof_index.size:
            raise RuntimeError('`func` (`dolfin.Function`) is '
                               'entirely below `threshold`')

        pos_index = self._index_dof_to_pos[dof_index]

        mask_candidates = np.zeros((self._nn,), bool)
        mask_candidates[pos_index] = True

        mask_knowns = np.zeros((self._nn,), bool)
        mask_knowns[pos_index] = True

        distances = np.full((self._nn,), np.inf)
        distances[pos_index] = 0.0

        if self._nz:
            travelpaths = np.empty((self._nn, 3), np.uint)
            travelpaths[:,0] = self._nx-1
            travelpaths[:,1] = self._ny-1
            travelpaths[:,2] = self._nz-1
            travelpaths[pos_index,:] = 0.0
        else:
            travelpaths = np.empty((self._nn, 2), np.uint)
            travelpaths[:,0] = self._nx-1
            travelpaths[:,1] = self._ny-1

        travelpaths[pos_index,:] = 0.0

        if maximum_distance is not None:
            while not mask_knowns.all():
                self._update_candidates(mask_candidates, mask_knowns)
                self._update_distances(distances, travelpaths, mask_candidates, mask_knowns)
                if (distances[mask_candidates] > maximum_distance).all():
                    distances[~mask_knowns] = distances[mask_candidates].max()
                    break
        else:
            while not mask_knowns.all():
                self._update_candidates(mask_candidates, mask_knowns)
                self._update_distances(distances, travelpaths, mask_candidates, mask_knowns)

        if require_interpolation:

            if out is not None:
                tmp = Function(self._V)
                tmp.vector()[:] = distances[self._index_dof_to_pos]
                out.vector()[:] = interpolate(tmp, V_out).vector()
            else:
                tmp = Function(self._V)
                tmp.vector()[:] = distances[self._index_dof_to_pos]
                out = interpolate(tmp, V_out)

        else:

            if out is not None:
                out.vector()[:] = distances[self._index_dof_to_pos]
            else:
                out = Function(V_out)
                out.vector()[:] = distances[self._index_dof_to_pos]

        return out


    def __update_candidates_2d(self, mask_candidates, mask_knowns):

        index = np.flatnonzero(mask_candidates)
        subs = self._index_pos_to_subs[index]

        assert mask_knowns[index].all(), \
            'Expected "candidates" to have been visited'

        mask_candidates[index[subs[:,0]>0] - 1] = True # Left
        mask_candidates[index[subs[:,0]<self._nx-1] + 1] = True # Right

        mask_candidates[index[subs[:,1]>0] - self._nx] = True # Bottom
        mask_candidates[index[subs[:,1]<self._ny-1] + self._nx] = True # Top

        # Reset candidates nodes to non-candidate status
        mask_candidates[mask_knowns] = False


    def __update_candidates_3d(self, mask_candidates, mask_knowns):

        index = np.flatnonzero(mask_candidates)
        subs = self._index_pos_to_subs[index]

        assert mask_knowns[index].all(), \
            'Expected "candidates" to have been visited'

        mask_candidates[index[subs[:,0]>0] - 1] = True # Left
        mask_candidates[index[subs[:,0]<self._nx-1] + 1] = True # Right

        mask_candidates[index[subs[:,1]>0] - self._nx] = True # Bottom
        mask_candidates[index[subs[:,1]<self._ny-1] + self._nx] = True # Top

        mask_candidates[index[subs[:,2]>0] - self._nxny] = True # candidates
        mask_candidates[index[subs[:,2]<self._nz-1] + self._nxny] = True # Back

        # Reset candidates nodes to non-candidate status
        mask_candidates[mask_knowns] = False


    def __update_distances_2d(self, distances, travelpaths, mask_candidates, mask_knowns):

        index = np.flatnonzero(mask_candidates)
        subs = self._index_pos_to_subs[index]

        assert not mask_knowns[index].any(), \
            'Expected "candidates" to be unvisited'

        distances_ = distances[index].copy()
        travelpaths_ = travelpaths[index].copy()

        assert (distances_ == np.inf).all(), \
            'Expected candidate distances to be unknown'


        trial = np.flatnonzero(subs[:,0] > 0)
        travelpaths_[trial,:] = travelpaths[index[trial]-1,:]
        travelpaths_[trial,0] += 1 # Advance once along "x"

        distances_[trial] = (travelpaths_[trial,0]*self._hx)**2 \
                          + (travelpaths_[trial,1]*self._hy)**2


        trial = np.flatnonzero(subs[:,0] < self._nx-1)
        travelpaths_trial = travelpaths[index[trial]+1,:].copy()
        travelpaths_trial[:,0] += 1 # Advance once along "-x"

        distances_trial = (travelpaths_trial[:,0]*self._hx)**2 \
                        + (travelpaths_trial[:,1]*self._hy)**2

        tmp = np.flatnonzero(distances_[trial] > distances_trial)
        distances_[trial[tmp]] = distances_trial[tmp]
        travelpaths_[trial[tmp]] = travelpaths_trial[tmp]


        trial = np.flatnonzero(subs[:,1] > 0)
        travelpaths_trial = travelpaths[index[trial]-self._nx,:].copy()
        travelpaths_trial[:,1] += 1 # Advance once along "y"

        distances_trial = (travelpaths_trial[:,0]*self._hx)**2 \
                        + (travelpaths_trial[:,1]*self._hy)**2

        tmp = np.flatnonzero(distances_[trial] > distances_trial)
        distances_[trial[tmp]] = distances_trial[tmp]
        travelpaths_[trial[tmp]] = travelpaths_trial[tmp]


        trial = np.flatnonzero(subs[:,1] < self._ny-1)
        travelpaths_trial = travelpaths[index[trial]+self._nx,:].copy()
        travelpaths_trial[:,1] += 1 # Advance once along "-y"

        distances_trial = (travelpaths_trial[:,0]*self._hx)**2 \
                        + (travelpaths_trial[:,1]*self._hy)**2

        tmp = np.flatnonzero(distances_[trial] > distances_trial)
        distances_[trial[tmp]] = distances_trial[tmp]
        travelpaths_[trial[tmp]] = travelpaths_trial[tmp]


        distances[index] = np.sqrt(distances_)
        travelpaths[index] = travelpaths_
        mask_knowns[index] = True

        assert (distances_ < np.inf).all(), 'Expected distances to be finite'


    def __update_distances_3d(self, distances, travelpaths, mask_candidates, mask_knowns):

        index = np.flatnonzero(mask_candidates)
        subs = self._index_pos_to_subs[index]

        assert not mask_knowns[index].any(), \
            'Expected candidates to be unvisited'

        distances_ = distances[index].copy()
        travelpaths_ = travelpaths[index].copy()

        assert (distances_ == np.inf).all(), \
            'Expected candidate distances to be unknown'


        trial = np.flatnonzero(subs[:,0] > 0)
        travelpaths_[trial,:] = travelpaths[index[trial]-1,:]
        travelpaths_[trial,0] += 1 # Advance once along "x"

        distances_[trial] = (travelpaths_[trial,0]*self._hx)**2 \
                          + (travelpaths_[trial,1]*self._hy)**2 \
                          + (travelpaths_[trial,2]*self._hz)**2


        trial = np.flatnonzero(subs[:,0] < self._nx-1)
        travelpaths_trial = travelpaths[index[trial]+1,:].copy()
        travelpaths_trial[:,0] += 1 # Advance once along "-x"

        distances_trial = (travelpaths_trial[:,0]*self._hx)**2 \
                        + (travelpaths_trial[:,1]*self._hy)**2 \
                        + (travelpaths_trial[:,2]*self._hz)**2

        tmp = np.flatnonzero(distances_[trial] > distances_trial)
        distances_[trial[tmp]] = distances_trial[tmp]
        travelpaths_[trial[tmp]] = travelpaths_trial[tmp]


        trial = np.flatnonzero(subs[:,1] > 0)
        travelpaths_trial = travelpaths[index[trial]-self._nx,:].copy()
        travelpaths_trial[:,1] += 1 # Advance once along "y"

        distances_trial = (travelpaths_trial[:,0]*self._hx)**2 \
                        + (travelpaths_trial[:,1]*self._hy)**2 \
                        + (travelpaths_trial[:,2]*self._hz)**2

        tmp = np.flatnonzero(distances_[trial] > distances_trial)
        distances_[trial[tmp]] = distances_trial[tmp]
        travelpaths_[trial[tmp]] = travelpaths_trial[tmp]


        trial = np.flatnonzero(subs[:,1] < self._ny-1)
        travelpaths_trial = travelpaths[index[trial]+self._nx,:].copy()
        travelpaths_trial[:,1] += 1 # Advance once along "-y"

        distances_trial = (travelpaths_trial[:,0]*self._hx)**2 \
                        + (travelpaths_trial[:,1]*self._hy)**2 \
                        + (travelpaths_trial[:,2]*self._hz)**2

        tmp = np.flatnonzero(distances_[trial] > distances_trial)
        distances_[trial[tmp]] = distances_trial[tmp]
        travelpaths_[trial[tmp]] = travelpaths_trial[tmp]


        trial = np.flatnonzero(subs[:,2] > 0)
        travelpaths_trial = travelpaths[index[trial]-self._nxny,:].copy()
        travelpaths_trial[:,2] += 1 # Advance once along "z"

        distances_trial = (travelpaths_trial[:,0]*self._hx)**2 \
                        + (travelpaths_trial[:,1]*self._hy)**2 \
                        + (travelpaths_trial[:,2]*self._hz)**2

        tmp = np.flatnonzero(distances_[trial] > distances_trial)
        distances_[trial[tmp]] = distances_trial[tmp]
        travelpaths_[trial[tmp]] = travelpaths_trial[tmp]


        trial = np.flatnonzero(subs[:,2] < self._nz-1)
        travelpaths_trial = travelpaths[index[trial]+self._nxny,:].copy()
        travelpaths_trial[:,2] += 1 # Advance once along "-z"

        distances_trial = (travelpaths_trial[:,0]*self._hx)**2 \
                        + (travelpaths_trial[:,1]*self._hy)**2 \
                        + (travelpaths_trial[:,2]*self._hz)**2

        tmp = np.flatnonzero(distances_[trial] > distances_trial)
        distances_[trial[tmp]] = distances_trial[tmp]
        travelpaths_[trial[tmp]] = travelpaths_trial[tmp]


        distances[index] = np.sqrt(distances_)
        travelpaths[index] = travelpaths_
        mask_knowns[index] = True

        assert (distances_ < np.inf).all(), 'Expected distances to be finite'


    @staticmethod
    def _equivalent_function_spaces(A, B):
        if A.dim() != B.dim() or \
           A.ufl_element() != B.ufl_element() or \
           A.mesh().num_vertices() != B.mesh().num_vertices() or \
           not np.allclose(A.mesh().coordinates(), B.mesh().coordinates()):
            return False
        return True


class FastMarchingMethod:

    def __init__(self, p0, p1, stepsize):

        if isinstance(p0, np.ndarray):
            p0 = p0.copy()
        else:
            p0 = np.array(p0)

        if isinstance(p1, np.ndarray):
            p1 = p1.copy()
        else:
            p1 = np.array(p1)

        dim = len(p0)

        if dim != 2 and dim != 3:
            raise TypeError('Points `p0` and `p1` can either lie in 2D or 3D space')

        if len(p1) != dim:
            raise TypeError('Inconsistent dimensions of points `p0` and `p1`')

        if not (isinstance(stepsize, (float, int)) and stepsize > 0):
            raise ValueError('Parameter `stepsize` must be a positive value')

        self._stepsize = stepsize

        if dim == 2:

            L, H = p1 - p0

            num_cells_x = int(L/stepsize + 1.0)
            num_cells_y = int(H/stepsize + 1.0)

            L_new = num_cells_x * stepsize
            H_new = num_cells_y * stepsize

            assert L_new >= L
            assert H_new >= H

            cover = (L_new - L) * 0.5
            if cover > L * EPS:
                p0[0] -= cover
                p1[0] += cover

            cover = (H_new - H) * 0.5
            if cover > H * EPS:
                p0[1] -= cover
                p1[1] += cover

            mesh = dolfin.RectangleMesh(dolfin.Point(p0), dolfin.Point(p1),
                                        num_cells_x, num_cells_y)

            self._V = dolfin.FunctionSpace(mesh, 'CG', 1)

            self._nx = nx = num_cells_x + 1
            self._ny = ny = num_cells_y + 1
            self._nz = None

            self._nn = nx * ny
            self._nxny = self._nn

            xs = np.linspace(p0[0], p1[0], nx)
            ys = np.linspace(p0[1], p1[1], ny)

            xs_grid = np.array(sum([xs.tolist(),] * ny, []))
            ys_grid = np.array(sum([[yi,] * nx for yi in ys], []))

            pt_dofs = self._V.tabulate_dof_coordinates()

            self._index_dof_to_pos = np.array(
                [yi/stepsize*nx + xi/stepsize + 0.5
                for xi, yi in pt_dofs-p0], np.uint)

            self._index_pos_to_subs = np.array(
                [(pos%nx, pos//nx) for pos in range(self._nn)])

            self._update_candidates = self.__update_candidates_2d
            self._update_distances = self.__update_distances_2d

        else: # dim == 3

            L, H, W = p1 - p0

            num_cells_x = int(L/stepsize + 1.0)
            num_cells_y = int(H/stepsize + 1.0)
            num_cells_z = int(W/stepsize + 1.0)

            L_new = num_cells_x * stepsize
            H_new = num_cells_y * stepsize
            W_new = num_cells_z * stepsize

            assert L_new >= L
            assert H_new >= H
            assert W_new >= W

            cover = (L_new - L) * 0.5
            if cover > L * EPS:
                p0[0] -= cover
                p1[0] += cover

            cover = (H_new - H) * 0.5
            if cover > H * EPS:
                p0[1] -= cover
                p1[1] += cover

            cover = (W_new - W) * 0.5
            if cover > W * EPS:
                p0[2] -= cover
                p1[2] += cover

            mesh = dolfin.BoxMesh(dolfin.Point(p0), dolfin.Point(p1),
                                  num_cells_x, num_cells_y, num_cells_z)

            self._V = dolfin.FunctionSpace(mesh, 'CG', 1)

            self._nx = nx = num_cells_x + 1
            self._ny = ny = num_cells_y + 1
            self._nz = nz = num_cells_z + 1

            self._nn = nx * ny * nz;
            self._nxny = nxny = nx*ny

            xs = np.linspace(p0[0], p1[0], nx)
            ys = np.linspace(p0[1], p1[1], ny)
            zs = np.linspace(p0[2], p1[2], nz)

            xs_grid = np.array(sum([xs.tolist(),] * (ny*nz), []))
            ys_grid = np.array(sum([[yi,] * nx for zi in zs for yi in ys], []))
            zs_grid = np.array(sum([[zi,] * nxny for zi in zs], []))

            pt_dofs = self._V.tabulate_dof_coordinates()

            self._index_dof_to_pos = np.array(
                [zi/stepsize*nxny + yi/stepsize*nx + xi/stepsize + 0.5
                for xi, yi, zi in pt_dofs-p0], np.uint)

            self._index_pos_to_subs = np.array(
                [(pos%nxny%nx, pos%nxny//nx, pos//nxny)
                for pos in range(self._nn)])

            self._update_candidates = self.__update_candidates_3d
            self._update_distances = self.__update_distances_3d


    def compute_distance_function(self, func, threshold,
            out=None, V_out=None, maximum_distance=None):

        if out is not None:
            if V_out is not None:
                logger.warning('Parameter `V_out` is redundant '
                               'since parameter `out` is given')
            V_out = out.function_space()

        elif V_out is None:
            V_out = func.function_space()

        require_interpolation = \
            not self._equivalent_function_spaces(V_out, self._V)

        if require_interpolation:
            func.set_allow_extrapolation(True)
            func = interpolate(func, self._V)

        dof_index = np.flatnonzero(func.vector().get_local() > threshold)

        if not dof_index.size:
            raise RuntimeError('`func` (`dolfin.Function`) is '
                               'entirely below `threshold`')

        pos_index = self._index_dof_to_pos[dof_index]

        mask_candidates = np.zeros((self._nn,), bool)
        mask_candidates[pos_index] = True

        mask_knowns = np.zeros((self._nn,), bool)
        mask_knowns[pos_index] = True

        distances = np.full((self._nn,), np.inf)
        distances[pos_index] = 0.0

        if maximum_distance is not None:
            while not mask_knowns.all():
                self._update_candidates(mask_candidates, mask_knowns)
                self._update_distances(distances, mask_candidates, mask_knowns)
                if (distances[mask_candidates] > maximum_distance).all():
                    distances[~mask_knowns] = distances[mask_candidates].max()
                    break
        else:
            while not mask_knowns.all():
                self._update_candidates(mask_candidates, mask_knowns)
                self._update_distances(distances, mask_candidates, mask_knowns)

        if require_interpolation:

            if out is not None:
                tmp = Function(self._V)
                tmp.vector()[:] = distances[self._index_dof_to_pos]
                out.vector()[:] = interpolate(tmp, V_out).vector()
            else:
                tmp = Function(self._V)
                tmp.vector()[:] = distances[self._index_dof_to_pos]
                out = interpolate(tmp, V_out)

        else:

            if out is not None:
                out.vector()[:] = distances[self._index_dof_to_pos]
            else:
                out = Function(V_out)
                out.vector()[:] = distances[self._index_dof_to_pos]

        return out


    def __update_candidates_2d(self, mask_candidates, mask_knowns):

        index = np.flatnonzero(mask_candidates)
        subs = self._index_pos_to_subs[index]

        assert mask_knowns[index].all(), \
            'Expected "candidates" to have been visited'

        mask_candidates[index[subs[:,0]>0] - 1] = True # Left
        mask_candidates[index[subs[:,0]<self._nx-1] + 1] = True # Right

        mask_candidates[index[subs[:,1]>0] - self._nx] = True # Bottom
        mask_candidates[index[subs[:,1]<self._ny-1] + self._nx] = True # Top

        # Reset visited nodes to non-candidate status
        mask_candidates[mask_knowns] = False


    def __update_candidates_3d(self, mask_candidates, mask_knowns):

        index = np.flatnonzero(mask_candidates)
        subs = self._index_pos_to_subs[index]

        assert mask_knowns[index].all(), \
            'Expected "candidates" to have been visited'

        mask_candidates[index[subs[:,0]>0] - 1] = True # Left
        mask_candidates[index[subs[:,0]<self._nx-1] + 1] = True # Right

        mask_candidates[index[subs[:,1]>0] - self._nx] = True # Bottom
        mask_candidates[index[subs[:,1]<self._ny-1] + self._nx] = True # Top

        mask_candidates[index[subs[:,2]>0] - self._nxny] = True # Front
        mask_candidates[index[subs[:,2]<self._nz-1] + self._nxny] = True # Back

        # Reset visited nodes to non-candidate status
        mask_candidates[mask_knowns] = False


    def __update_distances_2d(self, distances, mask_candidates, mask_knowns):

        index = np.flatnonzero(mask_candidates)
        subs = self._index_pos_to_subs[index]

        assert not mask_knowns[index].any(), \
            'Expected "candidates" to be unvisited'

        assert (distances[index] == np.inf).all(), \
            'Expected candidate distances to be unknown'

        dist_x = np.full((len(index), 2), np.inf)
        dist_y = np.full((len(index), 2), np.inf)

        trial = subs[:,0] > 0
        dist_x[trial,0] = distances[index[trial]-1]

        trial = subs[:,0] < self._nx-1
        dist_x[trial,1] = distances[index[trial]+1]

        trial = subs[:,1] > 0
        dist_y[trial,0] = distances[index[trial]-self._nx]

        trial = subs[:,1] < self._ny-1
        dist_y[trial,1] = distances[index[trial]+self._nx]

        dist_x = dist_x.min(1)
        dist_y = dist_y.min(1)

        trial = (dist_x < np.inf) * (dist_y == np.inf)
        distances[index[trial]] = dist_x[trial] + self._stepsize

        trial = (dist_x == np.inf) * (dist_y < np.inf)
        distances[index[trial]] = dist_y[trial] + self._stepsize

        trial = (dist_x < np.inf) * (dist_y < np.inf)
        distances[index[trial]] = self.__update_formula_for_2d(
            dist_x[trial], dist_y[trial], self._stepsize)

        mask_knowns[index] = True

        assert (distances[mask_knowns] < np.inf).all(), \
            'Expected distances to be finite'


    def __update_distances_3d(self, distances, mask_candidates, mask_knowns):

        index = np.flatnonzero(mask_candidates)
        subs = self._index_pos_to_subs[index]

        assert not mask_knowns[index].any(), \
            'Expected "candidates" to be unvisited'

        assert (distances[index] == np.inf).all(), \
            'Expected candidate distances to be unknown'

        dist_x = np.full((len(index), 2), np.inf)
        dist_y = np.full((len(index), 2), np.inf)
        dist_z = np.full((len(index), 2), np.inf)

        trial = subs[:,0] > 0
        dist_x[trial,0] = distances[index[trial]-1]

        trial = subs[:,0] < self._nx-1
        dist_x[trial,1] = distances[index[trial]+1]

        trial = subs[:,1] > 0
        dist_y[trial,0] = distances[index[trial]-self._nx]

        trial = subs[:,1] < self._ny-1
        dist_y[trial,1] = distances[index[trial]+self._nx]

        trial = subs[:,2] > 0
        dist_z[trial,0] = distances[index[trial]-self._nxny]

        trial = subs[:,2] < self._nz-1
        dist_z[trial,1] = distances[index[trial]+self._nxny]

        dist_x = dist_x.min(1)
        dist_y = dist_y.min(1)
        dist_z = dist_z.min(1)

        trial = (dist_x < np.inf) * (dist_y == np.inf) * (dist_z == np.inf)
        distances[index[trial]] = dist_x[trial] + self._stepsize

        trial = (dist_x == np.inf) * (dist_y < np.inf) * (dist_z == np.inf)
        distances[index[trial]] = dist_y[trial] + self._stepsize

        trial = (dist_x == np.inf) * (dist_y == np.inf) * (dist_z < np.inf)
        distances[index[trial]] = dist_z[trial] + self._stepsize

        trial = (dist_x < np.inf) * (dist_y < np.inf) * (dist_z == np.inf)
        distances[index[trial]] = self.__update_formula_for_2d(
            dist_x[trial], dist_y[trial], self._stepsize)

        trial = (dist_x == np.inf) * (dist_y < np.inf) * (dist_z < np.inf)
        distances[index[trial]] = self.__update_formula_for_2d(
            dist_y[trial], dist_z[trial], self._stepsize)

        trial = (dist_x < np.inf) * (dist_y == np.inf) * (dist_z < np.inf)
        distances[index[trial]] = self.__update_formula_for_2d(
            dist_x[trial], dist_z[trial], self._stepsize)

        trial = (dist_x < np.inf) * (dist_y < np.inf) * (dist_z < np.inf)
        distances[index[trial]] = self.__update_formula_for_3d(
            dist_x[trial], dist_y[trial], dist_z[trial], self._stepsize)

        mask_knowns[index] = True

        assert (distances[mask_knowns] < np.inf).all(), \
            'Expected distances to be finite'


    @staticmethod
    def __update_formula_for_2d(d1, d2, h):
        d_sum = d1 + d2
        square = d_sum**2 - (d1**2 + d2**2 - h**2) * 2.0
        assert (square >= 0.0).all()
        return (d_sum + np.sqrt(square)) * 0.5

    @staticmethod
    def __update_formula_for_3d(d1, d2, d3, h):
        d_sum = d1 + d2 + d3
        square = d_sum**2 - (d1**2 + d2**2 + d3**2 - h**2) * 3.0
        invalid = square < 0.0
        if invalid.any():
            logger.error('Distance could not be updated accurately')
            square[invalid] = 0
        return (d_sum + np.sqrt(square)) / 3.0

    @staticmethod
    def _equivalent_function_spaces(A, B):
        if A.dim() != B.dim() or \
           A.ufl_element() != B.ufl_element() or \
           A.mesh().num_vertices() != B.mesh().num_vertices() or \
           not np.allclose(A.mesh().coordinates(), B.mesh().coordinates()):
            return False
        return True


if __name__ == "__main__":

    import time
    from . import filter
    import matplotlib.pyplot as plt
    plt.interactive(True)

    def integration_measure_over_expression(mesh, expr, threshold):

        mf = dolfin.MeshFunction('size_t', mesh, mesh.geometric_dimension())
        dx_subdomains = dolfin.dx(subdomain_data=mf, domain=mesh)

        V_DG = dolfin.FunctionSpace(mesh, 'DG', 0)

        mf.array()[:] = np.array(
            dolfin.project(expr, V_DG).vector()
            .get_local() > threshold, np.uint)

        dx_interior = dx_subdomains(1)
        dx_exterior = dx_subdomains(0)

        return dx_interior, dx_exterior


    methods = ['variational', 'algebraic', 'fmm']
    # methods = ['algebraic', 'fmm']
    # methods = ['variational']
    # methods = ['algebraic']
    # methods = ['fmm']

    problem_dimension = 3

    threshold = 1/3

    L, H, W = 1, 1, 1

    R = H * 0.25 # Defect radius
    # R = 0

    p_norm = 1.5

    if problem_dimension == 2:

        nx = 150
        ny = round(nx*H/L)
        nz = None

        p0 = np.array([-L/2, -H/2])
        p1 = np.array([ L/2,  H/2])

        if R == 0:
            expr = dolfin.Expression(f"fabs(x[0]+{L/2}) < {L*EPS} && fabs(x[1]+{H/2}) < {H*EPS}", degree=1)
        else:
            if p_norm == np.inf:
                expr = dolfin.Expression(f"fabs(x[0]) < {R} && fabs(x[1]) < {R}", degree=1) # L_inf-norm
            else:
                expr = dolfin.Expression(f"pow(fabs(x[0]),{p_norm}) + pow(fabs(x[1]),{p_norm}) "
                                         f"< pow({R},{p_norm})", degree=1)  # Lp-norm

        mesh = dolfin.RectangleMesh(dolfin.Point(p0), dolfin.Point(p1), nx, ny, diagonal="crossed")
        # mesh = dolfin.RectangleMesh(dolfin.Point(p0), dolfin.Point(p1), nx, ny, diagonal="left")
        # mesh = dolfin.RectangleMesh(dolfin.Point(p0), dolfin.Point(p1), nx, ny)

    elif problem_dimension == 3:

        nx = 40
        ny = round(nx*H/L)
        nz = round(nx*W/L)

        p0 = np.array([-L/2, -H/2, -W/2])
        p1 = np.array([ L/2,  H/2,  W/2])

        if R == 0:
            expr = dolfin.Expression(f"fabs(x[0]+{L/2}) < {L*EPS} && fabs(x[1]+{H/2}) < {H*EPS} "
                                     f"&& fabs(x[2]+{W/2}) < {W*EPS}", degree=1)
        else:
            if p_norm is np.inf:
                expr = dolfin.Expression(f"fabs(x[0]+{L/2}) < {R} && fabs(x[1]+{H/2}) < {R} "
                                         f"&& fabs(x[2]+{W/2}) < {R}", degree=1) # L_inf-norm
            else:
                expr = dolfin.Expression(f"pow(fabs(x[0]+{L/2}),{p_norm}) + pow(fabs(x[1]+{H/2}),{p_norm}) "
                                         f"+ pow(fabs(x[2]+{W/2}),{p_norm}) < pow({R},{p_norm})", degree=1) # Lp-norm

        mesh = dolfin.BoxMesh(dolfin.Point(p0), dolfin.Point(p1), nx, ny, nz)

    else:
        raise ValueError('Parameter `problem_dimension` must equal either `2` or `3`')

    V = dolfin.FunctionSpace(mesh, 'CG', 1)
    phasefield_function = dolfin.project(expr, V)

    # filter.apply_diffusive_smoothing(phasefield_function, 1e-4)

    _, dx_error = integration_measure_over_expression(
        mesh, phasefield_function, threshold)

    if problem_dimension == 2:
        fig = plt.figure(1); fig.clear()
        dolfin.plot(phasefield_function)
        plt.title(f'Phasefield Function')
        plt.axis('tight')
        plt.axis('equal')
        plt.show()

    for i, method_i in enumerate(methods):

        if method_i == 'variational':

            t0 = time.perf_counter()

            solve_distance_i, distance_function_i, distance_function_arr_i = \
                variational_distance_solver(phasefield_function)

            t1 = time.perf_counter()

            solve_distance_i()

            t2 = time.perf_counter()

            gradient_expression_i = dolfin.sqrt(dolfin.grad(distance_function_i)**2)
            gradient_function_i = dolfin.project(gradient_expression_i, V)

            distance_function_variational = distance_function_i
            distance_function_variational.rename('variational','')

            gradient_function_variational = gradient_function_i
            gradient_function_variational.rename('variational (grad)','')

        elif method_i == 'algebraic':

            t0 = time.perf_counter()

            solve_distance_i, distance_function_i, distance_function_arr_i = \
                algebraic_distance_solver(phasefield_function)

            t1 = time.perf_counter()

            solve_distance_i()

            t2 = time.perf_counter()

            gradient_expression_i = dolfin.sqrt(dolfin.grad(distance_function_i)**2)
            gradient_function_i = dolfin.project(gradient_expression_i, V)

            distance_function_algebraic = distance_function_i
            distance_function_algebraic.rename('algebraic','')

            gradient_function_algebraic = gradient_function_i
            gradient_function_algebraic.rename('algebraic (grad)','')

        elif method_i == 'fmm':

            t0 = time.perf_counter()

            solve_distance_i, distance_function_i, distance_function_arr_i = \
                fast_marching_method(phasefield_function)

            t1 = time.perf_counter()

            solve_distance_i()

            t2 = time.perf_counter()

            gradient_expression_i = dolfin.sqrt(dolfin.grad(distance_function_i)**2)
            gradient_function_i = dolfin.project(gradient_expression_i, V)

            distance_function_fmm = distance_function_i
            distance_function_fmm.rename('fmm','')

            gradient_function_fmm = gradient_function_i
            gradient_function_fmm.rename('fmm (grad)','')

        else:
            raise RuntimeError

        ### This measure of error is too strong
        # gradient_error_i = math.sqrt(dolfin.assemble(
        #     (gradient_function_i-Constant(1.0))**2*dx_error)
        #     / dolfin.assemble(1*dx_error))

        ### This measure of error is more useful
        gradient_error_i = (dolfin.assemble(
            (gradient_function_i-Constant(1.0))*dx_error)
            / dolfin.assemble(1*dx_error))

        print(f'\nMethod: "{method_i.upper()}"\n'
              f'  Init. time : {t1-t0:g},\n'
              f'  Solve time : {t2-t1:g},\n'
              f'  Grad error : {gradient_error_i:g}')

        if problem_dimension == 2:

            fig = plt.figure(i+10); fig.clear()
            dolfin.plot(distance_function_i)
            plt.title(f'Distance Solution ({method_i} method)')
            plt.axis('equal')

            fig = plt.figure(i+100); fig.clear()
            dolfin.plot(gradient_function_i)
            plt.title(f'Distance Gradient ({method_i} method)')
            plt.axis('equal')

            plt.show()
