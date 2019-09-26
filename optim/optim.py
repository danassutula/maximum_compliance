'''Optimize material distribution for maximum compliance (minimum stiffness) of
a hyper-elastic solid.

By Danas Sutula
University of Liberec, Czech Republic, 2018-2019

'''

import math
import dolfin
import numpy as np
import scipy.linalg as linalg

from dolfin import Constant
from dolfin import Function
from dolfin import assemble
from dolfin import derivative
from dolfin import interpolate
from dolfin import grad
from dolfin import dot
from dolfin import dx

from . import config
logger = config.logger

EPS = 1e-12


class TopologyOptimizer:
    def __init__(self, J, P, F, C, u, p, p_locals, bcs_u,
                 function_to_call_during_iterations=None):
        '''Minimize the objective functional `J(u(p),p)` (e.g. potential energy
        of a hyper-elastic solid) with respect to the phasefield `p` subject to
        satisfying the weak-form `F(u(p),p)` (e.g. static equilibrium) and the
        phasefield equality-constraint functionals `C(p)`.

        Parameters
        ----------
        J : dolfin.Form
            Cost functional to be minimized with respect to phasefield `p`.
        P : dolfin.Form
            Phasefield penalty (regularization) functional that solely depends
            on the phasefield `p`. The functional `P` should be a measure of the
            roughness of the phasefield `p`. The phasefield advance direction
            will be determined as the weighted-average direction of the cost
            gradient direction and the phasefield penalty gradient direction.
        F : dolfin.Form
            Variational problem, i.e. `F(u,p; v)==0` for all `v`.
        C : (sequence of) dolfin.Form(s)
            Linear phasefield equality-constraint functional(s) that solely
            depend on the phasefield `p`.
        p : dolfin.Function
            Global phasefield function.
        p_locals : (sequence of) dolfin.Function(s)
            Local phasefield functions. The sum of the local phasefield functions
            will be equivelent to the global phasefield function `p`.
        u : dolfin.Function
            The displacement function or a mixed field function.
        bcs_u : (sequence of) dolfin.DirichletBC's
            The dirichlet boundary conditions for the displacement or the mixed
            field function.
        function_to_call_during_iterations : function(self)
            Function to be called at each iteration of the solver. The `self`
            object will be passed to the function as the argument. This function
            will typically be responsible for recording the solution state.

        '''

        if function_to_call_during_iterations is not None \
           and not callable(function_to_call_during_iterations):
            raise TypeError('Parameter `function_to_call_during_iterations` must '
                            'be callable with the `self` object as the argument.')

        threshold = config.parameters_distance_solver['threshold']
        viscosity = config.parameters_distance_solver['viscosity']
        penalty   = config.parameters_distance_solver['penalty']

        if not (isinstance(threshold, (float, int)) and 0.0 < threshold < 1.0):
            raise TypeError('`config.parameters_distance_solver[\'threshold\']`')

        if not (isinstance(viscosity, (float, int)) and viscosity > 0.0):
            raise TypeError('`config.parameters_distance_solver[\'viscosity\']`')

        if not (isinstance(penalty, (float, int)) and penalty > 0.0):
            raise TypeError('`config.parameters_distance_solver[\'penalty\']`')

        self._V_u = u.function_space()
        self._V_p = p.function_space()

        self._u = u
        self._p = p
        self._p_locals = None

        self._p_vec_global = p.vector()
        self._p_vec_locals = None
        self._d_arr_locals = None

        self._collision_distance = None

        self._distance_solver = DistanceSolver(
            self._V_p, threshold, viscosity, penalty)

        self._initialize_local_phasefields(p_locals)

        self._domain_integral_dof_weights = \
            assemble(dolfin.TestFunction(self._V_p)*dx).get_local()

        self._J = J
        self._P = P

        self._dJdp = derivative(J, p)
        self._dPdp = derivative(P, p)
        self._dFdu = derivative(F, u)

        if isinstance(C, (tuple, list)):
            self._C = C if isinstance(C, tuple) else tuple(C)
            self._dCdp = tuple(derivative(Ci, p) for Ci in C)
        else:
            self._C = (C,)
            self._dCdp = (derivative(C, p),)

        self._dCdp_arr = tuple(assemble(dCidp).get_local() for dCidp in self._dCdp)
        self._tol_C = tuple(np.abs(dCidp).sum()*EPS for dCidp in self._dCdp_arr)
        self._dp_C_arr = self._constraint_correction_vectors(self._dCdp_arr)

        self._nonlinear_solver = dolfin.NonlinearVariationalSolver(
            dolfin.NonlinearVariationalProblem(F, u, bcs_u, self._dFdu))
        self._solve_equilibrium_problem = self._nonlinear_solver.solve

        self._update_parameters(
            self._nonlinear_solver.parameters,
            config.parameters_nonlinear_solver)

        self.parameters_topology_solver = \
            config.parameters_topology_solver.copy()

        self.function_to_call_during_iterations = function_to_call_during_iterations \
            if function_to_call_during_iterations is not None else lambda self : None


    def solve_equilibrium_problem(self):
        '''Solve for the equilibrium displacements.'''
        return self._solve_equilibrium_problem()


    def _initialize_local_phasefields(self, p_locals):

        if isinstance(p_locals, tuple):
            self._p_locals = p_locals
        elif isinstance(p_locals, Function):
            self._p_locals = (p_locals,)
        else:
            try:
                self._p_locals = tuple(p_locals)
            except:
                self._p_locals = None

        if self._p_locals is None:
            raise TypeError('Expected parameter `p_locals` to be '
                            'a (sequence of) `dolfin.Function`(s)')

        if not all(p_i.function_space() == self._V_p for p_i in self._p_locals):
            raise TypeError('Parameter `p_locals` must constrain `dolfin.Function`s '
                            'that are members of the same function space as `p`')

        self._p_vec_locals = tuple(p_i.vector() for p_i in self._p_locals)
        self._d_arr_locals = np.zeros((len(self._p_locals), self._V_p.dim()))

        self._initialize_phasefield_distances() # -> self._d_arr_locals (approximate)
        self._solve_phasefield_distances() # -> self._d_arr_locals (correct solution)

        if len(p_locals) == 1:
            self._apply_collision_prevention = lambda p_arr : None


    def optimize(self, stepsize, penalty_weight, collision_distance,
                 convergence_tolerance=None, minimum_convergences=None,
                 maximum_iterations=None):
        '''
        Parameters
        ----------
        stepsize : float
            The maximum phasefield nodal change per iteration, i.e. l_inf norm.
        penalty_weight : float
            The phasefield advance direction will be the weighted-average direction
            of the phasefield penalty gradient `dPdp` and the cost gradient `dJdp`.
        collision_distance : float
            Value for the minimum distance between local phasefields (`p_locals`).

        Returns
        -------
        iterations_count : int
            Number of iterations.
        cost_values : list of float's
            The value of the cost functional for each iteration.

        '''

        if stepsize <= 0.0:
            raise ValueError('Require `stepsize > 0`')

        if collision_distance < 0.0:
            raise ValueError('Require `collision_distance > 0`')

        if not 0.0 <= penalty_weight <= 1.0:
            raise ValueError('Require `0 <= penalty_weight <= 1`')

        if convergence_tolerance is None: convergence_tolerance = \
            self.parameters_topology_solver['convergence_tolerance']

        if minimum_convergences is None: minimum_convergences = \
            self.parameters_topology_solver['minimum_convergences']

        if maximum_iterations is None: maximum_iterations = \
            self.parameters_topology_solver['maximum_iterations']

        if convergence_tolerance < 0:
            raise ValueError('Require non-negative `convergence_tolerance`')

        if minimum_convergences < 0:
            raise ValueError('Require non-negative `minimum_convergences`')

        if maximum_iterations < 0:
            raise ValueError('Require non-negative `maximum_iterations`')

        if (self._d_arr_locals[:,0] == 0).all() or \
           (self._d_arr_locals[:,0] == np.inf).all():
            raise RuntimeError('Ill-defined phasefields.')

        def compute_phasefield_increment():

            x_J = assemble(self._dJdp).get_local()
            x_P = assemble(self._dPdp).get_local()

            # Orthogonalize with respect to (orthogonal) constraints
            for dCdp_i, dp_C_i in zip(self._dCdp_arr, self._dp_C_arr):
                x_J -= dp_C_i * (x_J.dot(dCdp_i) / dp_C_i.dot(dCdp_i))
                x_P -= dp_C_i * (x_P.dot(dCdp_i) / dp_C_i.dot(dCdp_i))

            # Weighted-average phasefield advance direction
            dp_arr = x_J * (-weight_J/math.sqrt(x_J.dot(x_J))) \
                   + x_P * (-weight_P/math.sqrt(x_P.dot(x_P)))

            dp_arr *= stepsize / np.abs(dp_arr).max()

            return dp_arr

        weight_P = penalty_weight
        weight_J = 1.0 - weight_P

        convergences_count = 0
        iterations_count = 0
        progress_count = 0

        self._collision_distance = collision_distance
        p_arr = sum(self._p_vec_locals).get_local()

        self._solve_phasefield_distances()
        self._apply_phasefield_constraints(p_arr)
        self._assign_phasefield_values(p_arr)

        self._solve_phasefield_distances()
        self._apply_collision_prevention(p_arr)
        self._apply_phasefield_constraints(p_arr)
        self._assign_phasefield_values(p_arr)

        self._solve_equilibrium_problem()

        J_min = assemble(self._J)
        cost_values = [J_min,]

        # convergence_tolerance *= abs(J_min)
        # convergence_factor = 1.0 - convergence_tolerance

        # cost_values = []
        # J_min = np.inf

        while True:

            iterations_count += 1

            dp_arr = compute_phasefield_increment()
            p_arr_prv = p_arr; p_arr = p_arr + dp_arr

            self._solve_phasefield_distances()
            self._apply_collision_prevention(p_arr)
            self._apply_phasefield_constraints(p_arr)
            self._assign_phasefield_values(p_arr)

            self._solve_equilibrium_problem()

            ### Report iteration

            J_cur = assemble(self._J)
            cost_values.append(J_cur)

            dp_arr = p_arr - p_arr_prv
            abs_dp_arr = np.abs(dp_arr)
            max_dp_arr = abs_dp_arr.max()

            normL1_dp = self._domain_integral_dof_weights.dot(abs_dp_arr)

            logger.info(
                f'k:{iterations_count:3d}, '
                f'J:{J_cur: 11.5e}, '
                f'|dp|_inf:{max_dp_arr: 8.2e}, '
                f'|dp|_1:{normL1_dp: 8.2e}'
                )

            self.function_to_call_during_iterations(self)

            ### Assess convergence

            # if J_min - J_cur > convergence_tolerance:
            # if J_cur < J_min * convergence_factor:
            if J_min - J_cur > convergence_tolerance * abs(J_cur) * normL1_dp:

                J_min = J_cur

                progress_count += 1
                if progress_count == 0:
                    convergences_count = 0

            else:

                progress_count = -3
                convergences_count += 1

                if convergences_count > minimum_convergences:
                    logger.info('Reached minimum number of convergences')
                    break

            if iterations_count >= maximum_iterations:
                logger.warning('Reached maximum number of iterations')
                break

        return iterations_count, cost_values


    def _initialize_phasefield_distances(self):
        '''Initialize distances to local phasefields.'''

        compute_dofs = self._distance_solver.compute_initdist_dofs
        mark_cells = self._distance_solver.mark_zero_distance_cells

        for p_i, d_arr_i in zip(self._p_locals, self._d_arr_locals):

            if mark_cells(p_i):
                compute_dofs(d_arr_i)
            else:
                d_arr_i[:] = np.inf


    def _solve_phasefield_distances(self):
        '''Update distances to local phasefields.'''

        compute_dofs = self._distance_solver.compute_distance_dofs
        mark_cells = self._distance_solver.mark_zero_distance_cells

        for p_i, d_arr_i in zip(self._p_locals, self._d_arr_locals):

            if mark_cells(p_i):
                compute_dofs(d_arr_i)
            else:
                d_arr_i[:] = np.inf


    def _apply_collision_prevention(self, p_arr):

        # NOTE: The "distance" needs to be defined as the second smallest distance
        # because the smallest value just is a reference to a particular phasefield.

        mask = (self._d_arr_locals < self._collision_distance).sum(0) > 1
        s = np.sort(self._d_arr_locals[:,mask], 0)[1] / self._collision_distance
        p_arr[mask] *= self._collision_smoothing_weight(s)


    def _apply_phasefield_constraints(self, p_arr):
        '''Apply constraints on the estimated phasefield.

        Parameters
        ----------
        p_arr : numpy.ndarray (1D)
            Estimated phasefield dof/nodal values.

        '''

        p0_arr = self._p_vec_global.get_local()

        for C_i, dCdp_i, dp_C_i, tol_C_i in zip(
                self._C, self._dCdp_arr, self._dp_C_arr, self._tol_C):

            C_i = assemble(C_i)
            dp_C_i = dp_C_i.copy()

            ind_lwr = p_arr < 0.0
            ind_upr = p_arr > 1.0

            p_arr[ind_lwr] = 0.0
            p_arr[ind_upr] = 1.0

            R_i = C_i + dCdp_i.dot(p_arr-p0_arr)

            while abs(R_i) > tol_C_i:

                dRdp_C_i = dCdp_i.dot(dp_C_i)

                if abs(dRdp_C_i) < tol_C_i:
                    raise RuntimeError('Unable to enforce equality constraints.')

                dp_C_i *= -R_i/dRdp_C_i

                p_arr += dp_C_i

                ind_lwr = p_arr < 0.0
                ind_upr = p_arr > 1.0

                dp_C_i[ind_lwr] = 0.0
                dp_C_i[ind_upr] = 0.0

                p_arr[ind_lwr] = 0.0
                p_arr[ind_upr] = 1.0

                R_i = C_i + dCdp_i.dot(p_arr-p0_arr)


    def _assign_phasefield_values(self, p_arr):

        self._p_vec_global[:] = p_arr

        assert all(abs(assemble(C_i)) < tol_C_i * 10.0
            for C_i, tol_C_i in zip(self._C, self._tol_C))

        # Closest phasefield indices (at dof positions)
        phasefield_dofmap = self._d_arr_locals.argmin(0)

        for i, p_vec_i in enumerate(self._p_vec_locals):
            dofs_i = np.flatnonzero(phasefield_dofmap==i)
            p_vec_i[:]=0.0; p_vec_i[dofs_i]=p_arr[dofs_i]


    @staticmethod
    def _constraint_correction_vectors(dCdp):
        '''Construct constraint correction vectors.

        The equality constraints are assumed to be the constraints for a fixed
        phasefield fraction within each of the subdomains where the constraints
        are defined.

        Parameters
        ----------
        dCdp : Sequence of `numpy.ndarray`s (1D)
            Gradients of the constraint equations.

        Returns
        -------
        dp_C : `list` of `numpy.ndarray`s (1D)
            Constraint correction vectors.

        '''

        # Phasefield dofs affected by constraints
        A = [dCidp.astype(bool) for dCidp in dCdp]

        # Decouple constraint activities
        for i, A_i in enumerate(A[:-1]):
            for A_j in A[i+1:]:
                A_j[A_i] = False

        # Decoupled constraint correction vectors
        dp_C = tuple(A_i.astype(float) for A_i in A)

        assert all(abs(v_i.dot(v_j)) < EPS*min(v_j.dot(v_j), v_i.dot(v_i))
            for i, v_i in enumerate(dp_C[:-1]) for v_j in dp_C[i+1:]), \
            'Constraint correction vectors are not mutually orthogonal.'

        return dp_C


    @staticmethod
    def _collision_smoothing_weight(s):
        '''Regularized step function.'''
        return 3*s**2 - 2*s**3


    @classmethod
    def _update_parameters(cls, target, source):
        '''Update dict-like `target` with dict-like `source`.'''

        for k in source.keys():

            if k not in target.keys():
                raise KeyError(k)

            if hasattr(target[k], 'keys'):

                if not hasattr(source[k], 'keys'):
                    raise TypeError(f'`source[{k}]` must be dict-like.')
                else:
                    cls._update_parameters(target[k], source[k])

            elif hasattr(source[k], 'keys'):
                raise TypeError(f'`source[{k}]` can not be dict-like.')

            else:
                target[k] = source[k]


class DistanceSolver:
    def __init__(self, V, threshold, viscosity=1e-2, penalty=1e5):
        '''
        Parameters
        ----------
        V : dolfin.FunctionSpace
            Function space for the distance function.
        threshold : float
            Threshold marking the zero-distance boundary.
        viscosity: float or dolfin.Constant
            Stabilization for the unique solution of the distance problem.
        penalty: float or dolfin.Constant
            Penalty for weakly enforcing the zero-distance boundary conditions.

        '''

        if not isinstance(V, dolfin.FunctionSpace):
            raise TypeError('Parameter `V` must be a `dolfin.FunctionSpace`.')

        if not isinstance(threshold, float):
            if not isinstance(threshold, int):
                raise TypeError('Parameter `threshold`')
            threshold = float(threshold)

        if not isinstance(viscosity, Constant):
            if not isinstance(viscosity, (float, int)):
                raise TypeError('`Parameter `viscosity`')
            viscosity = Constant(viscosity)

        if not isinstance(penalty, Constant):
            if not isinstance(penalty, (float, int)):
                raise TypeError('Parameter `penalty`')
            penalty = Constant(penalty)

        self._threshold = threshold
        self._viscosity = viscosity
        self._penalty   = penalty

        self._d = d = Function(V)
        self._d_vec = d.vector()

        mesh = V.mesh()

        self._Q = dolfin.FunctionSpace(mesh, 'DG', 0)

        x = mesh.coordinates()
        l0 = (x.max(0)-x.min(0)).min()
        he = mesh.hmax()

        scaled_penalty = penalty / he
        target_gradient = Constant(1.0)

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

        J = derivative(F, d, v1)

        problem = dolfin.NonlinearVariationalProblem(F, d, bcs=None, J=J)
        self._nonlinear_solver = dolfin.NonlinearVariationalSolver(problem)
        self._nonlinear_solver.parameters['nonlinear_solver'] = 'newton'
        self._nonlinear_solver.parameters['symmetric'] = False

        self._solve_initdist_problem = self._linear_solver.solve
        self._solve_distance_problem = self._nonlinear_solver.solve

    def set_threshold_value(self, value):
        self._threshold = float(value)

    def set_viscosity_value(self, value):
        self._viscosity.assign(value)

    def set_penalty_value(self, value):
        self._penalty.assign(value)

    def get_distance_function(self):
        return self._d

    def mark_zero_distance_cells(self, p):
        '''Mark cells as zero-distance if the phasefield value
        is greater than the threshold value `p_max * threshold`.'''

        self._mf.array()[:] = np.array(
            interpolate(p, self._Q).vector()
            .get_local() > self._threshold, int)

        return self._mf.array().any()

    def compute_initdist_dofs(self, x):
        self._solve_initdist_problem()
        x[:] = self._d_vec.get_local()

    def compute_distance_dofs(self, x):

        self._d_vec[:] = x

        try:
            self._solve_distance_problem()
        except RuntimeError:
            logger.error('Could not solve distance problem. '
                         'Attempting to re-initialize and re-solve.')

            self._solve_initdist_problem()

            try:
                self._solve_distance_problem()
            except RuntimeError:
                logger.error('Unable to solve distance problem. '
                             'Reusing the previous distance solution.')

                return

        x[:] = self._d_vec.get_local()

    def compute_distance(self, p, init=True):
        '''Compute distance to thresholded phasefield boundary.'''

        if not self.mark_zero_distance_cells(p):
            raise RuntimeError('Could not mark any zero-distance cells.')

        if init:
            self._solve_initdist_problem()

        self._solve_distance_problem()

        return self._d
