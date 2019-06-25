'''Topology optimization based on phase-fields. (biomech-optimtop)

By Danas Sutula
University of Liberec, Czech Republic, 2018-2019

'''

import math
import dolfin
import numpy as np

from dolfin import Constant
from dolfin import Function
from dolfin import assemble
from dolfin import derivative
from dolfin import interpolate
from dolfin import grad
from dolfin import dot
from dolfin import dx

from . import config
from . import utility


EPS = 1e-12

SEQUENCE_TYPES = (tuple, list)
logger = config.logger


class TopologyOptimizer:
    '''Minimize a cost functional.'''

    def __init__(self, W, P, C, p, p_locals, u, bcs_u, external_function=None):
        '''

        Parameters
        ----------
        W : dolfin.Form
            Potential energy to be minimized.
        P : dolfin.Form
            Functional that is a measure of the roughness of the phasefield.
            Note, this functional acts like penaly; that is, it modifies the
            phasefield increment vector so as to make the solution more smooth.
        C : (sequence of) dolfin.Form(s)
            Phasefield equality-constraint functionals. Note, each constraint
            functional must be linear in `p` and independent of `u`.
        p : dolfin.Function
            Global phasefield function.
        p_locals : (sequence of) dolfin.Function(s)
            Local phasefield functions. Note, the sum of the local phasefield
            functions should be equal to the global phasefield function.
        u : dolfin.Function
            Displacement field.
        bcs_u : (sequence of) dolfin.DirichletBC(s)
            Displacement field boundary conditions.
        alpha : float
            Parameter that is used to determine the implicit boundary of a
            phasefield. The value must be in range `(0, 1)`. The lower bound
            value of `0` means the phasefield boundary is the boundary of the
            domain where the phasefield is defined; the upper bound value of
            `1` -- the boundary of the domain where the phasefield is 1.0.
        kappa : float or dolfin.Constant
            Viscosity/diffusion-like parameter that regularizes the phasefield
            distance problem so that it can be solved uniquely.
        gamma : float or dolfin.Constant
            Penalty-like parameter used to weakly impose the zero-distance
            Dirichlet boundary condition on the phasefield distance solution.
        external_function : callable
            Function that is to be called at each iteration of the solver.

        '''

        if external_function is not None and not callable(external_function):
            raise TypeError('Parameter `external_function` must be callable.')

        alpha = config.parameters_distance_solver['alpha']
        kappa = config.parameters_distance_solver['kappa']
        gamma = config.parameters_distance_solver['gamma']

        if not isinstance(alpha, (float, int)):
            raise TypeError('`config.parameters_distance_solver[\'alpha\']`')

        if not 0.0 < alpha < 1.0:
            raise ValueError('`config.parameters_distance_solver[\'alpha\']`')

        if not isinstance(kappa, Constant):
            if not isinstance(kappa, (float, int)):
                raise TypeError('`config.parameters_distance_solver[\'kappa\']`')
            kappa = Constant(kappa)

        if not isinstance(gamma, Constant):
            if not isinstance(gamma, (float, int)):
                raise TypeError('`config.parameters_distance_solver[\'gamma\']`')
            gamma = Constant(gamma)

        self._alpha = alpha
        self._kappa = kappa
        self._gamma = gamma

        self._W = W
        self._P = P

        self._dWdp = derivative(W, p)
        self._dPdp = derivative(P, p)

        if isinstance(C, SEQUENCE_TYPES):
            self._C = C if isinstance(C, tuple) else tuple(C)
            self._dCdp = tuple(derivative(Ci, p) for Ci in C)
        else:
            self._C = (C,)
            self._dCdp = (derivative(C, p),)

        self._dCdp_arr = tuple(assemble(dCidp).get_local() for dCidp in self._dCdp)
        self._abstol_C = tuple(np.abs(dCidp).sum()*EPS for dCidp in self._dCdp_arr)
        self._dp_C_arr = self._constraint_correction_vectors(self._dCdp_arr)

        F = derivative(W, u)
        dFdu = derivative(F, u)

        self._nonlinear_solver = dolfin.NonlinearVariationalSolver(
            dolfin.NonlinearVariationalProblem(F, u, bcs_u, dFdu))
        self._solve_equilibrium_problem = self._nonlinear_solver.solve

        self._distance_solver = DistanceSolver(
            p.function_space(), kappa, gamma)

        self._u = u
        self._p = p

        self._p_vec_global = p.vector()

        self.initialize_local_phasefields(p_locals)

        utility.update_lhs_parameters_from_rhs(
            self._nonlinear_solver.parameters,
            config.parameters_nonlinear_solver)

        self.parameters_topology_solver = \
            config.parameters_topology_solver.copy()

        self.external_function = external_function \
            if external_function is not None else lambda : None


    def solve_equilibrium_problem(self):
        '''Solve for the equilibrium displacements.'''
        return self._solve_equilibrium_problem()


    def initialize_phasefield_distances(self):
        '''Initialize local phasefield distances.'''
        self._solve_phasefield_distances(init=True)


    def initialize_local_phasefields(self, p_locals):

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
                            'a (sequence of) `dolfin.Function`(s).')

        V_p = self._p.function_space()

        if not all(p_i.function_space() == V_p for p_i in self._p_locals):
            raise TypeError('Parameter `p_locals` must constrain `dolfin.Functions` '
                            'that are members of the same function space as `p`.')

        self._p_vec_locals = tuple(p_i.vector() for p_i in self._p_locals)
        self._d_arr_locals = np.zeros((len(self._p_locals), V_p.dim()))

        # Initialize local phasefield distances
        self._solve_phasefield_distances(init=True)


    def optimize(self, stepsize, smoothing_weight, minimum_distance,
                 convergence_tolerance=None, maximum_iterations=None,
                 maximum_divergences=None):
        '''
        Parameters
        ----------
        stepsize : float
            Phasefield stepsize. More precisely, it is the maximum nodal change
            in the phasefield per iteration, i.e. the l_inf norm of the change.
        smoothing_weight : float
            Factor weighting the amount of phasefield regularity (`P`) relative
            to the amount of phasefield strain energy (`W`) dissipationt.
        minimum_distance : float
            Minimum distance between local phasefields. The distance acts as
            a constraint that prevents phasefield collisions/intersections.

        Returns
        -------
        iterations_count : int
            Number of iterations.
        is_converged : bool
            Wheather the solver converged.
        potentials : list of float's
            The value of potential energy for each iteration.

        '''

        if stepsize <= 0.0:
            raise ValueError('Require `stepsize` to be positive')

        if minimum_distance < 0.0:
            raise ValueError('Parameter `minimum_distance` must be non-negative.')

        if not 0.0 <= smoothing_weight <= 1.0:
            raise ValueError('Parameter `smoothing_weight` must be between `0` and `1`.')

        prm = self.parameters_topology_solver

        if maximum_iterations is None:
            maximum_iterations = prm['maximum_iterations']

        if maximum_divergences is None:
            maximum_divergences = prm['maximum_divergences']

        if convergence_tolerance is None:
            convergence_tolerance = prm['convergence_tolerance']

        if maximum_iterations < 0:
            raise ValueError('Require non-negative `maximum_iterations`.')

        if maximum_divergences < 0:
            raise ValueError('Require non-negative `maximum_divergences`.')

        if convergence_tolerance < 0:
            raise ValueError('Require non-negative `convergence_tolerance`.')

        weight_P = smoothing_weight
        weight_W = 1.0 - weight_P

        p_arr = sum(self._p_vec_locals).get_local()
        dp_arr = 0.0 # np.zeros((len(p_arr,))) # Dummy

        # This solves for `self._d_arr_locals`
        self._solve_phasefield_distances()

        if (self._d_arr_locals[:,0] == np.inf).all():
            raise RuntimeError('Ill-defined phasefields.')

        divergences_count = 0
        iterations_count = 0
        is_converged = False

        potentials = []
        W_cur = np.inf

        while True:

            p_arr_prv = p_arr
            p_arr = p_arr + dp_arr

            ### Update phasefields (estimated)

            self._apply_phasefield_constraints(p_arr)
            self._assign_phasefield_values(p_arr)

            ### Phasefield collisions

            self._solve_phasefield_distances() # Updates `self._d_arr_locals`
            p_arr[(self._d_arr_locals < minimum_distance).sum(0) > 1] = 0.0

            ### Update phasefields (corrected)

            self._apply_phasefield_constraints(p_arr)
            self._assign_phasefield_values(p_arr)

            ### Solve equilibrium problem

            _, b = self._solve_equilibrium_problem()

            if not b:
                logger.error('Displacement problem could not be solved')
                break

            ### Report

            W_prv = W_cur
            W_cur = assemble(self._W)
            potentials.append(W_cur)

            normL1_p = p_arr_prv.sum()
            normL1_dp = np.abs(p_arr-p_arr_prv).sum()
            change_p = normL1_dp / normL1_p

            logger.info(f'k:{iterations_count:3d}, '
                        f'potential:{W_cur: 11.5e}, '
                        f'|dp|/|p|:{change_p: 8.2e}')

            self.external_function()

            ### Assess convergence

            if change_p < convergence_tolerance and iterations_count:
                logger.info('Negligible phase-field change (break)')
                is_converged = True
                break

            if W_prv < W_cur:
                logger.warning('Iterations diverged (energy did not decrease)')

                divergences_count += 1

                if divergences_count > maximum_divergences:
                    logger.error('Reached maximum number of diverged iterations')

                    # Assign previous phasefield solution
                    # NOTE: `p_arr_prv` satisfies constraints
                    self._assign_phasefield_values(p_arr_prv)

                    _, b = self._solve_equilibrium_problem()

                    if not b:
                        logger.error('Displacement problem could not be solved')

                    break

            if iterations_count == maximum_iterations:
                logger.error('Reached maximum number of iterations (break)')
                break

            iterations_count += 1

            ### Estimate phasefield change

            x_W = assemble(self._dWdp).get_local()
            x_P = assemble(self._dPdp).get_local()

            # Orthogonalize with respect to constraint vectors
            for dCdp_i, dp_C_i in zip(self._dCdp_arr, self._dp_C_arr):
                x_W -= dp_C_i * (x_W.dot(dCdp_i) / dp_C_i.dot(dCdp_i))
                x_P -= dp_C_i * (x_P.dot(dCdp_i) / dp_C_i.dot(dCdp_i))

            dp_arr = x_W * (-weight_W/math.sqrt(x_W.dot(x_W))) \
                   + x_P * (-weight_P/math.sqrt(x_P.dot(x_P)))

            dp_arr[(p_arr == 0.0) & (dp_arr < 0.0)] = 0.0
            dp_arr[(p_arr == 1.0) & (dp_arr > 0.0)] = 0.0

            dp_arr *= stepsize / np.abs(dp_arr).max()

        if not is_converged:
            if prm['error_on_nonconvergence']:
                raise RuntimeError('Iterations did not converge.')

        return iterations_count, is_converged, potentials


    def _solve_phasefield_distances(self, init=False):
        '''Compute distances to local phasefields.

        Parameters
        ----------
        init : bool (optional)
            Wheather to just initialize or solve the distance problem.

        '''

        if init: compute_dofs = self._distance_solver.compute_initdist_dofs
        else: compute_dofs = self._distance_solver.compute_distance_dofs
        mark_cells = self._distance_solver.mark_zero_distance_cells

        for p_i, d_arr_i in zip(self._p_locals, self._d_arr_locals):

            if mark_cells(p_i, self._alpha):
                compute_dofs(d_arr_i)
            else:
                d_arr_i[:] = np.inf


    def _apply_phasefield_constraints(self, p_arr):
        '''Apply constraints on the estimated phasefield.

        Parameters
        ----------
        p_arr : numpy.ndarray (1D)
            Estimated phasefield dof/nodal values.

        '''

        p0_arr = self._p_vec_global.get_local()

        for C_i, dCdp_i, dp_C_i, abstol_C_i in zip(
                self._C, self._dCdp_arr, self._dp_C_arr, self._abstol_C):

            C_i = assemble(C_i)
            dp_C_i = dp_C_i.copy()

            ind_lwr = p_arr < 0.0
            ind_upr = p_arr > 1.0

            p_arr[ind_lwr] = 0.0
            p_arr[ind_upr] = 1.0

            R_i = C_i + dCdp_i.dot(p_arr-p0_arr)

            while abs(R_i) > abstol_C_i:

                dRdp_C_i = dCdp_i.dot(dp_C_i)

                if abs(dRdp_C_i) < abstol_C_i:
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

        assert all(abs(assemble(C_i)) < abstol_C_i
            for C_i, abstol_C_i in zip(self._C, self._abstol_C))

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

        # Constraint correction vectors
        dp_C = tuple(A_i.astype(float) for A_i in A)

        assert all(abs(v_i.dot(v_j)) < EPS*min(v_j.dot(v_j), v_i.dot(v_i))
            for i, v_i in enumerate(dp_C[:-1]) for v_j in dp_C[i+1:]), \
            'Constraint correction vectors are not mutually orthogonal.'

        return dp_C


class DistanceSolver:
    def __init__(self, V, viscosity=1e-2, penalty=1e4):
        '''
        Parameters
        ----------
        V : dolfin.FunctionSpace
            Function space for the discritezation of the distance function.
        viscosity: float or dolfin.Constant
            Necessary stabilization for the solution of the distance equation.
        penalty: float or dolfin.Constant
            Penalty parameter for the weakly enforcement the zero-distance
            value Dirichlet boundary conditions.

        '''

        if not isinstance(V, dolfin.FunctionSpace):
            raise TypeError('Parameter `V` must be a `dolfin.FunctionSpace`.')

        self._d = d = Function(V)
        self._d_vec = d.vector()

        mesh = V.mesh()

        self._Q = dolfin.FunctionSpace(mesh, 'DG', 0)

        if not isinstance(viscosity, Constant):
            viscosity = Constant(viscosity)

        if not isinstance(penalty, Constant):
            penalty = Constant(penalty)

        self._viscosity = viscosity
        self._penalty = penalty

        x = mesh.coordinates()
        l0 = (x.max(0)-x.min(0)).min()

        scaled_penalty = penalty / l0
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

        F = v0*(dolfin.sqrt(grad(d)**2)-target_gradient)*dx \
            + viscosity*l0*dot(grad(v0), grad(d))*dx \
            + scaled_penalty*v0*d*self._dx_penalty

        J = derivative(F, d, v1)

        problem = dolfin.NonlinearVariationalProblem(F, d, bcs=None, J=J)
        self._nonlinear_solver = dolfin.NonlinearVariationalSolver(problem)
        self._nonlinear_solver.parameters['nonlinear_solver'] = 'newton'
        self._nonlinear_solver.parameters['symmetric'] = False

        self._solve_initdist_problem = self._linear_solver.solve
        self._solve_distance_problem = self._nonlinear_solver.solve

    def set_viscosity_value(self, value):
        self._viscosity.assign(value)

    def set_penalty_value(self, value):
        self._penalty.assign(value)

    def get_distance_function(self):
        return self._d

    def mark_zero_distance_cells(self, p, alpha):
        '''Mark cells as zero-distance if the phasefield value
        is greater than the threshold value `p_max * alpha`.'''

        self._mf.array()[:] = np.array(
            interpolate(p, self._Q).vector()
            .get_local() > alpha, int)

        return self._mf.array().any()

    def compute_initdist_dofs(self, x=None):

        if x is not None:
            self._solve_initdist_problem()
            x[:] = self._d_vec.get_local()
        else:
            self._solve_initdist_problem()
            x = self._d_vec.get_local()

        return x

    def compute_distance_dofs(self, x=None):

        if x is not None:
            self._d_vec[:] = x
            self._solve_distance_problem()
            x[:] = self._d_vec.get_local()
        else:
            self._solve_distance_problem()
            x = self._d_vec.get_local()

        return x

    def compute_distance_function(self, p, alpha, init=True):
        '''Compute distance to thresholded phasefield boundary.'''

        if not self.mark_zero_distance_cells(p, alpha):
            raise RuntimeError('Could not mark any zero-distance cells.')

        if init:
            self._solve_initdist_problem()

        self._solve_distance_problem()

        return self._d
