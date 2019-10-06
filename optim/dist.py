
import math
import dolfin
import numpy as np

from dolfin import interpolate
from dolfin import grad, dot, dx

from . import config
logger = config.logger

EPS = 1e-12


threshold = config.parameters_distance_solver['variational_solver']['threshold']
viscosity = config.parameters_distance_solver['variational_solver']['viscosity']
penalty   = config.parameters_distance_solver['variational_solver']['penalty']

if not (isinstance(threshold, (float, int)) and 0.0 < threshold < 1.0):
    raise TypeError('`config.parameters_distance_solver[\'threshold\']`')

if not (isinstance(viscosity, (float, int)) and viscosity > 0.0):
    raise TypeError('`config.parameters_distance_solver[\'viscosity\']`')

if not (isinstance(penalty, (float, int)) and penalty > 0.0):
    raise TypeError('`config.parameters_distance_solver[\'penalty\']`')



def phasefield_distance_solver(phasefield_functions, zero_distance_threshold, method="fmm"):

    if not isinstance(method, str):
        raise TypeError('Parameter `method` must be a `str`')

    if isinstance(phasefield_functions, tuple):
        return_distance_functions_as_tuple = True

    elif isinstance(phasefield_functions, list):
        phasefield_functions = tuple(phasefield_functions)
        return_distance_functions_as_tuple = True

    else:
        phasefield_functions = (phasefield_functions,)
        return_distance_functions_as_tuple =  False

    if not all(isinstance(func_i, dolfin.Function) for func_i in phasefield_functions):
        raise TypeError('Parameter `phasefield_functions` must contain `dolfin.Function`s')

    V = phasefield_functions[0].function_space()

    distance_functions = tuple(dolfin.Function(V)
        for _ in range(len(phasefield_functions)))

    if method == "fmm":
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

        solver = DistanceSolverFMM(p0, p1, nx, ny, nz)

        def solve_distances():
            for p_i, d_i in zip(phasefield_functions, distance_functions):
                solver.solve(p_i, zero_distance_threshold, d_i)

    elif method == "pde":

        params = config.parameters_distance_solver
        solver = DistanceSolverPDE(V, zero_distance_threshold,
            viscosity=params['viscosity'], penalty=params['penalty'])

        def solve_distances():
            for p_i, d_i in zip(phasefield_functions, distance_functions):

                if not solver._mark_zero_distance_cells(p_i):
                    raise RuntimeError("Could not mark any cells")

                if not d_i.vector().norm('l2'):
                    logger.info('Initial solve (approximation)')
                    solver.compute_initdist_dofs(d_i.vector())

                solver.compute_distance_dofs(d_i.vector())

    else:
        raise ValueError('Parameter `method` must either equal "fmm" or "pde"')

    return (distance_functions if return_distance_functions_as_tuple \
            else distance_functions[0]), solve_distances, solver



class VariationalDistanceSolver:

    def __init__(self, V, viscosity=1e-2, penalty=1e5):
        '''
        Parameters
        ----------
        V : dolfin.FunctionSpace
            Function space for the distance function.

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

        self._d = d = dolfin.Function(V)
        self._d_vec = d.vector()

        mesh = V.mesh()

        self._Q = dolfin.FunctionSpace(mesh, 'DG', 0)

        x = mesh.coordinates()
        l0 = (x.max(0)-x.min(0)).min()
        he = mesh.hmax()

        scaled_penalty = penalty / he
        target_gradient = dolfin.Constant(1.0)

        self._mf = dolfin.MeshFunction('size_t', mesh, mesh.geometric_dimension())
        self._dx_penalty = dx(subdomain_id=1, subdomain_data=self._mf, domain=mesh)

        v0 = dolfin.TestFunction(V)
        v1 = dolfin.TrialFunction(V)

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

    def _mark_zero_distance_cells(self, func, threshold):
        '''Mark the domain (cells) where the mean value of `func` is greater
        than the `threshold` value.'''

        self._mf.array()[:] = np.array(
            interpolate(func, self._Q).vector()
            .get_local() > threshold, np.uint)

        if not self._mf.array().any():
            raise RuntimeError('Could not mark any cells inside the '
                               'domain given the value of threshold')

    def compute_distance_function(self, func, threshold, init=False):
        '''Compute the distance (function) to the thresholded function.'''

        self._mark_zero_distance_cells(func, threshold)

        if init:
            self._solve_initdist_problem()
            self._solve_distance_problem()
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

        return self._d

    def get_distance_function(self):
        return self._d

    def set_viscosity(self, value):
        self._viscosity.assign(value)

    def set_penalty(self, value):
        self._penalty.assign(value)


class DistanceSolverFMM:
    '''Distance solver based on the Fast Marching Method (FMM).'''

    def __init__(self, p0, p1, num_cells_x, num_cells_y, num_cells_z=None):

        if not isinstance(p0, np.ndarray): p0 = np.array(p0)
        if not isinstance(p1, np.ndarray): p1 = np.array(p1)

        if num_cells_z is not None:

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

        else:

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

    def _initialize(self, func, zero_distance_threshold, criterion="above"):
        '''
        Todos
        -----
        The `dolfin.interpolate` is very expensive but it is needed if
        the function space of `func` is not the same as `self._V`.

        '''

        if not self._equivalent_function_spaces(func.function_space(), self._V):
            func = interpolate(func, self._V)

        if criterion == "above":
            dof_index = np.flatnonzero(func.vector() \
                .get_local() > zero_distance_threshold)

        elif criterion == "below":
            dof_index = np.flatnonzero(func.vector() \
                .get_local() < zero_distance_threshold)

        else:
            raise ValueError('Parameter `criterion` must be a `str` '
                             'equal to either \"above\" or \"below\"')

        if dof_index.size == 0:
            raise RuntimeError('Function `func` values do not meet '
                               '`zero_distance_threshold` value')

        pos_index = self._index_dof_to_pos[dof_index]

        candidate_mask = np.zeros((self._nn,), bool)
        visited_mask = np.zeros((self._nn,), bool)
        distances = np.full((self._nn,), np.inf)

        candidate_mask[pos_index] = True
        visited_mask[pos_index] = True
        distances[pos_index] = 0.0

        return distances, candidate_mask, visited_mask

    def __update_candidates_2d(self, candidate_mask, visited_mask):

        index = np.flatnonzero(candidate_mask)
        subs = self._index_pos_to_subs[index]

        assert visited_mask[index].all(), \
            'Expected all candidate nodes to have been visited'

        candidate_mask[index[subs[:,0]>0] - 1] = True # Left
        candidate_mask[index[subs[:,0]<self._nx-1] + 1] = True # Right

        candidate_mask[index[subs[:,1]>0] - self._nx] = True # Bottom
        candidate_mask[index[subs[:,1]<self._ny-1] + self._nx] = True # Top

        # Reset visited nodes to non-candidate status
        candidate_mask[visited_mask] = False

        subs_trial = self._index_pos_to_subs[candidate_mask]
        subs_known = self._index_pos_to_subs[visited_mask]
        subs_unknown = self._index_pos_to_subs[~visited_mask]

        # fh = plt.figure(1001); # fh.clear()
        # plt.plot(subs_unknown[:,0]*self._hx, subs_unknown[:,1]*self._hy, '.k')
        # plt.plot(subs_trial[:,0]*self._hx, subs_trial[:,1]*self._hy, 'ob')
        # plt.plot(subs_known[:,0]*self._hx, subs_known[:,1]*self._hy, '^r')
        # # plt.axis([0,0, 1,1])
        # # plt.axis('equal')
        # plt.show()

        # import ipdb; ipdb.set_trace()

    def __update_candidates_3d(self, candidate_mask, visited_mask):

        index = np.flatnonzero(candidate_mask)
        subs = self._index_pos_to_subs[index]

        assert visited_mask[index].all(), \
            'Expected all candidate nodes to have been visited'

        candidate_mask[index[subs[:,0]>0] - 1] = True # Left
        candidate_mask[index[subs[:,0]<self._nx-1] + 1] = True # Right

        candidate_mask[index[subs[:,1]>0] - self._nx] = True # Bottom
        candidate_mask[index[subs[:,1]<self._ny-1] + self._nx] = True # Top

        candidate_mask[index[subs[:,2]>0] - self._nxny] = True # Front
        candidate_mask[index[subs[:,2]<self._nz-1] + self._nxny] = True # Back

        # Reset visited nodes to non-candidate status
        candidate_mask[visited_mask] = False

    def __update_distances_2d(self, distances, candidate_mask, visited_mask):

        index = np.flatnonzero(candidate_mask)
        subs = self._index_pos_to_subs[index]

        assert not visited_mask[index].any(), \
            'Expected none of the candidate nodes to have been visited'

        assert (distances[index] == np.inf).all(), \
            'Expected none of the candidate node distances to be known'

        dist_x = np.full((len(index), 2), np.inf)
        dist_y = np.full((len(index), 2), np.inf)

        mask = subs[:,0] > 0
        dist_x[mask,0] = distances[index[mask]-1]

        mask = subs[:,0] < self._nx-1
        dist_x[mask,1] = distances[index[mask]+1]

        mask = subs[:,1] > 0
        dist_y[mask,0] = distances[index[mask]-self._nx]

        mask = subs[:,1] < self._ny-1
        dist_y[mask,1] = distances[index[mask]+self._nx]

        dist_x = dist_x.min(1)
        dist_y = dist_y.min(1)

        mask = (dist_x < np.inf) * (dist_y == np.inf)
        distances[index[mask]] = dist_x[mask] + self._hx

        mask = (dist_x == np.inf) * (dist_y < np.inf)
        distances[index[mask]] = dist_y[mask] + self._hy

        mask = (dist_x < np.inf) * (dist_y < np.inf)
        distances[index[mask]] = self.__fmm_formula_for_2d(
            dist_x[mask], dist_y[mask], self._hx, self._hy)

        visited_mask[index] = True

        assert (distances[visited_mask] < np.inf).all(), \
            'Expected all candidate node distances to be known'

    def __update_distances_3d(self, distances, candidate_mask, visited_mask):

        index = np.flatnonzero(candidate_mask)
        subs = self._index_pos_to_subs[index]

        assert not visited_mask[index].any(), \
            'Expected none of the candidate nodes to have been visited'

        assert (distances[index] == np.inf).all(), \
            'Expected none of the candidate node distances to be known'

        dist_x = np.full((len(index), 2), np.inf)
        dist_y = np.full((len(index), 2), np.inf)
        dist_z = np.full((len(index), 2), np.inf)

        mask = subs[:,0] > 0
        dist_x[mask,0] = distances[index[mask]-1]

        mask = subs[:,0] < self._nx-1
        dist_x[mask,1] = distances[index[mask]+1]

        mask = subs[:,1] > 0
        dist_y[mask,0] = distances[index[mask]-self._nx]

        mask = subs[:,1] < self._ny-1
        dist_y[mask,1] = distances[index[mask]+self._nx]

        mask = subs[:,2] > 0
        dist_z[mask,0] = distances[index[mask]-self._nxny]

        mask = subs[:,2] < self._nz-1
        dist_z[mask,1] = distances[index[mask]+self._nxny]

        dist_x = dist_x.min(1)
        dist_y = dist_y.min(1)
        dist_z = dist_z.min(1)

        mask = (dist_x < np.inf) * (dist_y == np.inf) * (dist_z == np.inf)
        distances[index[mask]] = dist_x[mask] + self._hx

        mask = (dist_x == np.inf) * (dist_y < np.inf) * (dist_z == np.inf)
        distances[index[mask]] = dist_y[mask] + self._hy

        mask = (dist_x == np.inf) * (dist_y == np.inf) * (dist_z < np.inf)
        distances[index[mask]] = dist_z[mask] + self._hz

        mask = (dist_x < np.inf) * (dist_y < np.inf) * (dist_z == np.inf)
        distances[index[mask]] = self.__fmm_formula_for_2d(
            dist_x[mask], dist_y[mask], self._hx, self._hy)

        mask = (dist_x == np.inf) * (dist_y < np.inf) * (dist_z < np.inf)
        distances[index[mask]] = self.__fmm_formula_for_2d(
            dist_y[mask], dist_z[mask], self._hy, self._hz)

        mask = (dist_x < np.inf) * (dist_y == np.inf) * (dist_z < np.inf)
        distances[index[mask]] = self.__fmm_formula_for_2d(
            dist_x[mask], dist_z[mask], self._hx, self._hz)

        mask = (dist_x < np.inf) * (dist_y < np.inf) * (dist_z < np.inf)
        distances[index[mask]] = self.__fmm_formula_for_3d(dist_x[mask],
            dist_y[mask], dist_z[mask], self._hx, self._hy, self._hz)

        visited_mask[index] = True

        assert (distances[visited_mask] < np.inf).all(), \
            'Expected all candidate node distances to be known'

    def solve(self, func, zero_distance_threshold, out=None,
              V_out=None, maximum_distance=None, criterion="above"):

        if out is not None:
            if V_out is not None:
                logger.warning('Parameter `V_out` is redundant since `out` is given')
            V_out = out.function_space()

        elif V_out is None:
            V_out = func.function_space()

        if maximum_distance is None:
            maximum_distance = 3

        distances, candidate_mask, visited_mask = \
            self._initialize(func, zero_distance_threshold, criterion)

        while not visited_mask.all():
            self._update_candidates(candidate_mask, visited_mask)
            self._update_distances(distances, candidate_mask, visited_mask)
            if (distances[candidate_mask] > maximum_distance).all():
                distances[~visited_mask] = maximum_distance
                logger.info("Reached the precribed maximum distance")
                break

        if self._equivalent_function_spaces(self._V, V_out):

            if out is not None:
                out.vector()[:] = distances[self._index_dof_to_pos]
            else:
                out = dolfin.Function(V_out)
                out.vector()[:] = distances[self._index_dof_to_pos]

        else:

            if out is not None:
                tmp = dolfin.Function(self._V)
                tmp.vector()[:] = distances[self._index_dof_to_pos]
                out.vector()[:] = interpolate(tmp, V_out).vector()
            else:
                tmp = dolfin.Function(self._V)
                tmp.vector()[:] = distances[self._index_dof_to_pos]
                out = interpolate(tmp, V_out)

        return out

    @staticmethod
    def __fmm_formula_for_2d(d1, d2, h1, h2):
        h1h1 = h1*h1
        h2h2 = h2*h2
        a = h1h1 + h2h2
        neg_b = d1*(h2h2/a) + d2*(h1h1/a)
        c = d2**2*(h1h1/a) + d1**2*(h2h2/a) - (h1h1*h2h2)/a
        det = neg_b**2 - c
        assert (det >= 0.0).all()
        return neg_b + np.sqrt(det)

    @staticmethod
    def __fmm_formula_for_3d(d1, d2, d3, h1, h2, h3):
        h1h1 = h1*h1
        h2h2 = h2*h2
        h3h3 = h3*h3
        a = h1h1*h2h2 + h2h2*h3h3 + h3h3*h1h1
        neg_b = d1*(h2h2*h3h3/a) + d2*(h3h3*h1h1/a) + d3*(h1h1*h2h2/a)
        c = d1**2*(h2h2*h3h3/a) + d2**2*(h3h3*h1h1/a) + d3**2*(h1h1*h2h2/a) - h1h1*h2h2*h3h3/a
        det = neg_b**2 - c
        assert (det >= 0.0).all()
        return neg_b + np.sqrt(det)

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

    problem_dimension = 2
    # methods = ['pde', 'fmm']
    methods = ['pde']

    threshold = 0.5

    L, H, W = 1, 1, 1
    R = H * 0.25

    power = 2.5

    if problem_dimension == 2:

        nx = 100
        ny = round(nx*H/L * 1.0)
        nz = None

        p0 = np.array([-L/2, -H/2])
        p1 = np.array([ L/2,  H/2])

        # expr = dolfin.Expression(f"(abs(x[0]-{R}) + abs(x[1]) < {R}) || "                    # L1-norm
        #                          f"(pow(x[0]+{R},2) + pow(x[1],2) < pow({R},2))", degree=1)  # L2-norm

        if power is np.inf:
            expr = dolfin.Expression(f"fabs(x[0]) < {R} && fabs(x[1]) < {R}", degree=1)  # L2-norm
        else:
            expr = dolfin.Expression(f"pow(fabs(x[0]),{power}) + pow(fabs(x[1]),{power}) < pow({R},{power})", degree=1)  # L2-norm

        # expr = dolfin.Expression(f'x[0] <= {p0[0]+L*0.5} && x[1] <= {p0[1]+H*0.5}', degree=1)  # L2-norm

        # mesh = dolfin.RectangleMesh(dolfin.Point(p0), dolfin.Point(p1), nx, ny, diagonal="crossed")
        # mesh = dolfin.RectangleMesh(dolfin.Point(p0), dolfin.Point(p1), nx, ny, diagonal="left")
        mesh = dolfin.RectangleMesh(dolfin.Point(p0), dolfin.Point(p1), nx, ny)

    elif problem_dimension == 3:

        nx = 50
        ny = round(nx*H/L * 1.25)
        nz = round(nx*W/L * 0.75)

        p0 = np.array([-L/2, -H/2, -W/2])
        p1 = np.array([ L/2,  H/2,  W/2])

        # expr = dolfin.Expression(f"(abs(x[0]-{L/4}) + abs(x[1]) + abs(x[2]) < {R}) || "                # L1-norm
        #                          f"(pow(x[0],2) + pow(x[1],2) + pow(x[2],2) < pow({R},2))", degree=1)  # L2-norm

        expr = dolfin.Expression(f'x[0] <= {p0[0]+L*0.5} && x[1] <= {p0[1]+H*0.5} && x[2] <= {p0[2]+W*0.5}', degree=1)  # L2-norm

        mesh = dolfin.BoxMesh(dolfin.Point(p0), dolfin.Point(p1), nx, ny, nz)

    else:
        raise ValueError('Parameter `problem_dimension` must equal either `2` or `3`')

    V = dolfin.FunctionSpace(mesh, 'CG', 1)
    func = dolfin.project(expr, V)
    filter.apply_diffusive_smoothing(func, kappa=1e-4)

    if problem_dimension == 2:
        fig = plt.figure(1); fig.clear()
        dolfin.plot(func)
        plt.title(f'Source Function')
        plt.axis('tight')
        plt.axis('equal')
        plt.show()

    for i, method_i in enumerate(methods):
        print(f'Solving distance problem using method: "{method_i}"')

        t0 = time.perf_counter()

        distance_function_i, solve_distances_i, _distance_solver_i = \
            phasefield_distance_solver(func, threshold, method_i)

        t1 = time.perf_counter()

        solve_distances_i()

        t2 = time.perf_counter()

        print('CPU times\n'
              f'  Init.: {t1-t0:g},\n'
              f'  Solve: {t2-t1:g}')

        gradient_function_i = dolfin.project(
            dolfin.sqrt(dolfin.grad(distance_function_i)**2), V)

        if method_i == 'pde':
            distance_function_pde = distance_function_i
            distance_function_pde.rename('pde','')
            gradient_function_pde = gradient_function_i
            gradient_function_pde.rename('pde (grad)','')
        elif method_i == 'fmm':
            distance_function_fmm = distance_function_i
            distance_function_fmm.rename('fmm','')
            gradient_function_fmm = gradient_function_i
            gradient_function_fmm.rename('fmm (grad)','')
        else:
            raise RuntimeError

        if problem_dimension == 2:

            fig = plt.figure(i+10); fig.clear()
            dolfin.plot(distance_function_i)
            plt.title(f'Distance Solution ({method_i.upper()})')
            plt.axis('tight')
            plt.axis('equal')

            fig = plt.figure(i+100); fig.clear()
            dolfin.plot(gradient_function_i)
            plt.title(f'Gradient of Distance ({method_i.upper()})')
            plt.axis('tight')
            plt.axis('equal')

            plt.show()
