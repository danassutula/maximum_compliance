'''Optimize material distribution for maximum compliance (minimum stiffness) of
a hyper-elastic solid.

By Danas Sutula
University of Liberec, Czech Republic, 2018-2019


Notes
-----
* Assumining to never have to solve the adjoint problem because the objective
functional to be minimized corresponds to the energy of the deformation.

'''

import sys
import math
import dolfin
import numpy as np

from dolfin import Constant
from dolfin import Function
from dolfin import assemble

from . import config
from . import dist

logger = config.logger
EPS = 1e-12


class TopologyOptimizer:

    MAXIMUM_DISTANCES_UPDATE_PERIOD = 2**(sys.int_info.bits_per_digit-1)

    def __init__(self, J, P, C, p, p_locals, equilibrium_solve, callback_function=None):
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
        C : (sequence of) dolfin.Form(s)
            Linear phasefield equality-constraint functional(s) that solely
            depend on the phasefield `p`.
        p : dolfin.Function
            Global phasefield function.
        p_locals : (sequence of) dolfin.Function(s)
            Local phasefield functions. The sum of the local phasefield functions
            will be equivelent to the global phasefield function `p`.
        equilibrium_solve : function()
            To be called with each iteration for solving the equilibrium problem.
        callback_function (optional): function()
            To be called with each iteration e.g. for writing out solution state.

        '''

        if not callable(equilibrium_solve):
            raise TypeError('Parameter `equilibrium_solve` must '
                            'be callable without any arguments')

        if callback_function is None:
            callback_function = lambda : None

        elif not callable(callback_function):
            raise TypeError('Parameter `callback_function` '
                            'must be callable without any arguments')

        self.equilibrium_solve = equilibrium_solve
        self.callback_function = callback_function

        self._p = p
        self._p_vec = p.vector()

        self._p_locals = None
        self._d_locals = None

        self._p_vec_locals = None
        self._d_arr_locals = None

        self._collision_distance = None

        self._initialize_local_phasefields(p_locals)
        assert self._p_locals is not None
        assert self._p_vec_locals is not None

        self._initialize_phasefield_distance_solver()
        assert self._d_locals is not None
        assert self._d_arr_locals is not None

        self._J = J
        self._P = P

        self._dJdp = dolfin.derivative(J, p)
        self._dPdp = dolfin.derivative(P, p)

        if isinstance(C, (tuple, list)):
            self._C = C if isinstance(C, tuple) else tuple(C)
            self._dCdp = tuple(dolfin.derivative(Ci, p) for Ci in C)
        else:
            self._C = (C,)
            self._dCdp = (dolfin.derivative(C, p),)

        self._dCdp_arr = tuple(assemble(dCidp).get_local() for dCidp in self._dCdp)
        self._tol_C = tuple(np.abs(dCidp).sum()*EPS for dCidp in self._dCdp_arr)
        self._dp_C_arr = self._constraint_correction_vectors(self._dCdp_arr)

        self.parameters_topology_solver = config.parameters_topology_solver.copy()


    def _initialize_local_phasefields(self, p_locals):

        if isinstance(p_locals, tuple):
            self._p_locals = p_locals
        elif isinstance(p_locals, Function):
            self._p_locals = (p_locals,)
        else:
            try:
                self._p_locals = tuple(p_locals)
            except:
                raise TypeError('Parameter `p_locals` must be a '
                                '(sequence of) `dolfin.Function`(s)')

        V = self._p.function_space()

        if not all(p_i.function_space() == V for p_i in self._p_locals):
            raise TypeError('Parameter `p_locals` must contain `dolfin.Function`s '
                            'that are members of the same function space as `p`')

        self._p_vec_locals = tuple(p_i.vector() for p_i in self._p_locals)


    def _initialize_phasefield_distance_solver(self):

        if config.parameters_distance_solver['method'] == 'variational':
            factory_distance_solver = dist.variational_distance_solver
        elif config.parameters_distance_solver['method'] == 'algebraic':
            factory_distance_solver = dist.algebraic_distance_solver
        else:
            raise ValueError('`config.parameters_distance_solver[\'method\']`')

        assert isinstance(self._p_locals, tuple)

        self._solve_phasefield_distances, self._d_locals, self._d_arr_locals = \
            factory_distance_solver(self._p_locals)

        if len(self._p_locals) > 1:
            self._solve_phasefield_distances()


    def optimize(self, stepsize, penalty_weight, collision_distance,
                 convergence_tolerance=None, minimum_convergences=None,
                 maximum_iterations=None, distances_update_frequency=None):
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
        distances_update_frequency : float
            The phasefield distance problem will be solved at approximately every
            `1 / distances_update_frequency` iterations. The value must be in range
            [0, 1] where the lower and upper bounds mean that the distance problem
            is 'solved never' and 'solved at every iteration' respectively.

        Returns
        -------
        num_iterations : int
            Number of iterations.
        cost_values : list of float's
            Values of the cost functional.

        '''

        if stepsize <= 0.0:
            raise ValueError('Require `stepsize > 0.0`')

        if not 0.0 <= penalty_weight <= 1.0:
            raise ValueError('Require `0.0 <= penalty_weight <= 1.0`')

        if collision_distance < 0.0:
            raise ValueError('Require `collision_distance > 0.0`')

        if convergence_tolerance is None: convergence_tolerance = \
            self.parameters_topology_solver['convergence_tolerance']

        if minimum_convergences is None: minimum_convergences = \
            self.parameters_topology_solver['minimum_convergences']

        if maximum_iterations is None: maximum_iterations = \
            self.parameters_topology_solver['maximum_iterations']

        if distances_update_frequency is None: distances_update_frequency = \
            self.parameters_topology_solver['distances_update_frequency']

        if convergence_tolerance < 0:
            raise ValueError('Require non-negative `convergence_tolerance`')

        if not (isinstance(minimum_convergences, int) and minimum_convergences > 0):
            raise ValueError('Require positive integer `minimum_convergences`')

        if not (isinstance(maximum_iterations, int) and maximum_iterations > 0):
            raise ValueError('Require positive integer `maximum_iterations`')

        if not 0.0 <= distances_update_frequency <= 1.0:
            raise ValueError('Require `0.0 <= distances_update_frequency <= 1.0`')

        if len(self._p_locals) > 1 and (
           (self._d_arr_locals[:,0] == 0).all() or
           (self._d_arr_locals[:,0] == np.inf).all()):
            raise RuntimeError('Ill-defined phasefields.')

        weight_P = penalty_weight
        weight_J = 1.0 - weight_P

        distances_update_period = int(1.0/distances_update_frequency) if \
            distances_update_frequency > 1/self.MAXIMUM_DISTANCES_UPDATE_PERIOD \
            else self.MAXIMUM_DISTANCES_UPDATE_PERIOD

        solve_phasefield_distances = self._solve_phasefield_distances \
            if len(self._p_locals) > 1 else lambda : None # Dummy callable

        self._collision_distance = collision_distance
        p_arr = sum(self._p_vec_locals).get_local()

        self._apply_phasefield_constraints(p_arr)
        self._assign_phasefield_values(p_arr)

        dp_arr = 0.0
        J_cur = np.inf
        cost_values = []
        num_iterations = 0

        while True:

            ### Update phasefields

            p_arr_prv = p_arr
            p_arr = p_arr + dp_arr

            if not num_iterations % distances_update_period:
                solve_phasefield_distances()

            self._apply_collision_constraints(p_arr)
            self._apply_phasefield_constraints(p_arr)
            self._assign_phasefield_values(p_arr)

            ### Solve

            self.equilibrium_solve()
            self.callback_function()

            J_cur = assemble(self._J)
            cost_values.append(J_cur)

            logger.info(f'n:{num_iterations:3d}, '
                        f'J:{J_cur: 11.5e}')

            num_iterations += 1

            try:
                # Break if the mean cost decrease over the recent 
                # `minimum_convergences` iterations is small enough.
                if (cost_values[-minimum_convergences-1] - J_cur) / \
                   minimum_convergences < abs(J_cur)*convergence_tolerance:
                    logger.info('Reached minimum number of convergences')
                    break
            except IndexError:
                pass

            if num_iterations == maximum_iterations:
                logger.error('Reached maximum number of iterations')
                break

            ### Estimate phasefield change

            dJdp = assemble(self._dJdp).get_local()
            dPdp = assemble(self._dPdp).get_local()

            # Orthogonalize with respect to constraints
            for dCdp_i, dp_C_i in zip(self._dCdp_arr, self._dp_C_arr):
                dJdp -= dp_C_i * (dJdp.dot(dCdp_i) / dp_C_i.dot(dCdp_i))
                dPdp -= dp_C_i * (dPdp.dot(dCdp_i) / dp_C_i.dot(dCdp_i))

            norm_dJdp = math.sqrt(dJdp.dot(dJdp))
            norm_dPdp = math.sqrt(dPdp.dot(dPdp))

            try:

                # Weighted-average advance vector
                dp_arr = dJdp * (-weight_J / norm_dJdp) \
                       + dPdp * (-weight_P / norm_dPdp)

            except ZeroDivisionError:

                if norm_dJdp == 0.0:
                    logger.error('Cost gradient is zero')
                    break

                if norm_dPdp == 0.0:
                    raise RuntimeError('Penalty gradient is zero')

            dp_arr *= stepsize / np.abs(dp_arr).max()

        return num_iterations, cost_values


    def _apply_collision_constraints(self, p_arr):

        if len(self._p_locals) > 1:
            mask = (self._d_arr_locals < self._collision_distance).sum(0) > 1
            s = np.sort(self._d_arr_locals[:,mask], 0)[1] / self._collision_distance  # NOTE: Should not have to sort; just need second smallest element
            p_arr[mask] *= self._collision_smoothing_weight(s)


    def _apply_phasefield_constraints(self, p_arr):
        '''Apply constraints on the estimated phasefield.

        Parameters
        ----------
        p_arr : numpy.ndarray (1D)
            Estimated phasefield dof/nodal values.

        '''

        p0_arr = self._p_vec.get_local()

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
                    if dp_C_i.any():
                        raise RuntimeError('Could not satisfy phasefield constraint; '
                            'constraint gradient is orthogonal to correction vector.')
                    break

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

        self._p_vec[:] = p_arr

        if len(self._p_locals) > 1:

            # Closest phasefield indices (at dof positions)
            phasefield_dofmap = self._d_arr_locals.argmin(0)
            for i, p_vec_i in enumerate(self._p_vec_locals):
                dofs_i = np.flatnonzero(phasefield_dofmap==i)
                p_vec_i[:]=0.0; p_vec_i[dofs_i]=p_arr[dofs_i]

        else:
            self._p_vec_locals[0][:] = p_arr


    @staticmethod
    def _constraint_correction_vectors(dCdp):
        '''Constraint correction vectors.

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

        # Orthogonalized constraint correction vectors
        dp_C = tuple(A_i.astype(float) for A_i in A)

        if any(dp_Ci.dot(dp_Ci) < len(dp_Ci)*EPS for dp_Ci in dp_C):
            raise RuntimeError('Constraints are linearly dependent.')

        assert all(abs(vi.dot(vj)) < EPS*min(vj.dot(vj), vi.dot(vi))
            for i, vi in enumerate(dp_C[:-1]) for vj in dp_C[i+1:]), \
            'Could not orthogonalize constraint correction vectors.'

        return dp_C


    @staticmethod
    def _collision_smoothing_weight(s):
        '''Regularized step function.'''
        return 3*s**2 - 2*s**3
