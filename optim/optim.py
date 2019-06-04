'''Topology optimization based on phase-fields. (biomech-optimtop)

By Danas Sutula
University of Liberec, Czech Republic, 2018-2019

NOTE: Assigning a vector rather than array is significantly faster

TODO:
    - What should be the right balance:
        `dp_arr = dp_hat_W*weight_W + dp_hat_P*weight_P`. This balance depends
        on the state of the solution. If the solution is overly diffusive
        than, unless `weight_P` is very small, the solution continues on to
        diffuse out. The reverse happens when the solution is very sharp; i.e.
        in such a case, the solution becomes sharper unless `weight_W` is very
        small. In summary, the solution tends either towards over-diffusion or
        ultra-sharpness.

    - Try optimizing over a subdomain only. How about having a 2D array of
    subdomains.

    - Acceleration of convergence using the history of the sesnitivities.
    - External phasefield evolution function could be beneficial. Adding it
    will decouple the potential energy and the regularizing penalty in the
    total cost functional.
    - When the cost functional is compoxed of multiple terms, the relative
    size of the terms becomes import. How to make it independent ? One way is
    to have a separate solver for the phase field -- we are back at it again.

NOTES:
    - Assuming C:=f(p) does not depend on `u`; hence,
    dCdu := derivative(C, u) = 0

    - having the phasefield evolution inside the total cost functional requires
    to solve the adjoint problem. The relative magnitues of different terms in
    the total cost function matter; e.g. if the BC are changed the these terms
    will grow appart.

    - If the phase field could be written in a separate function the
    adjoint would not need to be solved; however, the phasefield equation would
    need to be solved separately; thus, need to privded another functional.

    IMPORTANT
    - Penalty in total cost does not work so well becaus the action of pentlay is
    explicit, i.e. it has not effect at the start of the simulation. Need to be
    able to penalize after the gradient is computed based on energy dissipation.

    - The downside of smoothing is that the evolution of phasefield is not maximal
    because the energetically favourable vector is diffused.

    - So, what should you smooth? Energ gradient or the phasefield advance vector?

    - Note, the dissipation should be very local, so you should not diffuse too much.
    When using diffusion, need to recover the locality of dissipation, so map the
    dissipaton using an exponential function.

    - Thresholding phaseifled increments does not work so well

    - There was a proble with enforcing the equality constraint after the increment
    in phasefield because the phasefield increment would decrease to enforce constraint.
    This means phasefield almost never reaches 1.0.

    Problems
    --------
    - Mesh dependence, increasing diffusivity does not help; actually, it makes
    the problem worse, the diffusion becomes highly anisotropic. The remedy is
    to refine the mesh.

'''

import math
import dolfin
import numpy as np

from dolfin import Constant
from dolfin import Function
from dolfin import assemble
from dolfin import derivative
from dolfin import grad
from dolfin import dot
from dolfin import dx

from . import config
from . import utility


EPS = 1e-14
PI = math.pi

PHASEFIELD_LOWER_BOUND = -EPS
PHASEFIELD_UPPER_BOUND = 1.0 + EPS
DEGREES_TO_RADIANS =  PI / 180.0

SEQUENCE_TYPES = (tuple, list)

logger = config.logger


class TopologyOptimizer:
    '''Minimize a cost functional.'''

    def __init__(self, W, P, C, p, ps, u, bcs_u, kappa, userfunc=None):
        '''

        Parameters
        ----------
        W : dolfin.Form
            Potential energy to be minimized.
        P : dolfin.Form
            Penalty energy to be minimized.
        C : dolfin.Form or a sequence of dolfin.Form's
            Equality constraint(s) as integral equations.
        p : Function
            Global phasefield function.
        ps : sequence of Function's
            Local phasefield functions.
        kappa : float or dolfin.Constant
            Phasefield proximity sensing parameter. Individual phasefields are
            diffused and a measure of the overlap serves as a distance proxy.

        Notes
        -----
        The equality constraints `C` are assumed to be linear in `p` and
        independent of `u`.

        '''

        if not isinstance(ps, SEQUENCE_TYPES): ps = (ps,)
        if all(isinstance(ps_i, Function) for ps_i in ps):
            self._ps = ps if isinstance(ps, tuple) else tuple(ps)
        else:
            raise TypeError('Parameter `ps` must be a `Function` '
                            'or a sequence of `Function`s.')

        if not isinstance(kappa, Constant):
            kappa = Constant(float(kappa))

        if userfunc is not None and not callable(userfunc):
            raise TypeError('Parameter `userfunc` must be callable.')

        self._p = p
        self._u = u

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

        # Assuming constraints are linear (i.e. gradients are constant vectors)
        self._dCdp_arr = tuple(assemble(dCidp).get_local() for dCidp in self._dCdp)
        self._abstol_C = tuple(np.abs(dCidp).sum()*EPS for dCidp in self._dCdp_arr)
        self._dp_C_arr = self._constraint_correction_vectors(self._dCdp_arr)

        F = derivative(W, u)
        dFdu = derivative(F, u)

        nonlinear_problem = dolfin.NonlinearVariationalProblem(F, u, bcs_u, dFdu)
        self.nonlinear_solver = dolfin.NonlinearVariationalSolver(nonlinear_problem)

        v = dolfin.TestFunction(p.function_space())
        f = dolfin.TrialFunction(p.function_space())

        # Mass-like matrix for smoothing gradients
        self._M = assemble(f*v*dx)

        self._smoothing_solver = dolfin.LUSolver(self._M, "mumps")
        self._smoothing_solver.parameters["symmetric"] = True

        if bool(kappa.values()):
            A = assemble((f*v + kappa*dot(grad(f),grad(v)))*dx)
            self._diffusion_solver = dolfin.LUSolver(A, "mumps")
            self._diffusion_solver.parameters["symmetric"] = True
        else:
            self._diffusion_solver = self._smoothing_solver

        utility.update_lhs_parameters_from_rhs(
            self.nonlinear_solver.parameters,
            config.parameters_nonlinear_solver)

        self.parameters_topology_solver = \
            config.parameters_topology_solver.copy()

        self.userfunc = userfunc


    def optimize(self, stepsize, regularization,
        maximum_iterations=None, maximum_divergences=None,
        collision_threshold=None, convergence_tolerance=None):
        '''
        Parameters
        ----------
        regularization : float
            Penalty energy weight factor relative to strain energy.

        '''

        if stepsize <= 0.0:
            raise ValueError('Require `stepsize` to be positive')

        if not 0.0 <= regularization <= 1.0:
            raise ValueError('Parameter `regularization` out of range.')

        prm = self.parameters_topology_solver

        if maximum_iterations is None:
            maximum_iterations = prm['maximum_iterations']

        if maximum_divergences is None:
            maximum_divergences = prm['maximum_divergences']

        if collision_threshold is None:
            collision_threshold = prm['collision_threshold']

        if convergence_tolerance is None:
            convergence_tolerance = prm['convergence_tolerance']

        weight_P = regularization
        weight_W = 1.0 - weight_P

        p_vec = self._p.vector()
        p_arr = p_vec.get_local()
        dp_arr = 0.0 # Dummy value

        ps_vec = tuple(p.vector() for p in self._ps) # DOFs of local phasefields
        ws_arr = np.empty((len(ps_vec), len(p_arr))) # Diffused phasefield values

        divergences_count = 0
        is_converged = False
        error_message = ''

        W_val = np.inf
        k_itr = 0

        while True:

            p_arr_prv = p_arr
            p_arr = p_arr + dp_arr

            ### Diffuse phasefields to find interactions

            for i, p_vec_i in enumerate(ps_vec):
                x = self._M * p_vec_i
                self._diffusion_solver.solve(x, x)
                ws_arr[i,:] = x.get_local()

            dominant_phasefield_ids = ws_arr.argmax(0)
            competing_phasefield_dofs = np.flatnonzero(
                (ws_arr > collision_threshold).sum(0) > 1)

            p_arr[competing_phasefield_dofs] = 0.0

            # Correct `p_arr` so that constraints are satisfied
            self._enforce_phasefield_constraints(p_arr, p_arr_prv)

            # Actual phasefield change
            dp_arr = p_arr - p_arr_prv

            ### Update phasefields

            # Update global phasefield
            p_vec[:] = p_arr

            # Update local phasefields
            for i, pi_vec in enumerate(ps_vec):
                dofs_i = np.flatnonzero(dominant_phasefield_ids == i)
                pi_vec[dofs_i] = p_arr[dofs_i]

            ### Solve equilibrium problem

            _, b = self.nonlinear_solver.solve()

            if not b:
                logger.error('Unable to solve nonlinear problem')
                error_message = 'nonlinear_solver'
                break

            ### Report

            self.userfunc()

            W_val_prv = W_val
            W_val = assemble(self._W)

            normL1_p = p_arr.sum()
            normL1_dp = np.abs(dp_arr).sum()
            phasefield_change = normL1_dp / normL1_p

            logger.info(
                  f'k:{k_itr:3d}, '
                  f'W:{W_val: 11.5e}, '
                  f'|dp|/|p|:{phasefield_change: 8.2e}, '
                  f'stepsize:{stepsize: 0.4f} '
                  )

            ### Assess convergence

            if phasefield_change < convergence_tolerance and k_itr > 0:
                logger.info('Negligible phase-field change (break)')
                is_converged = True
                break

            if k_itr == maximum_iterations:
                logger.warning('Reached maximum number of iterations')
                error_message = 'maximum_iterations'
                break

            if W_val_prv < W_val:

                logger.warning('Iteration diverged.')
                divergences_count += 1

                if divergences_count > maximum_divergences:
                    logger.error('Exceeded maximum number of divergences.')
                    error_message = 'maximum_diverged_iterations'
                    break

            k_itr += 1

            ### Estimate phasefield change

            x_W = assemble(self._dWdp)
            x_P = assemble(self._dPdp)

            x = x_W * (weight_W/math.sqrt(x_W.inner(x_W))) \
              + x_P * (weight_P/math.sqrt(x_P.inner(x_P)))

            self._smoothing_solver.solve(x, x)

            dp_arr = -x.get_local()

            dp_arr[(p_arr == 0.0) & (dp_arr < 0.0)] = 0.0
            dp_arr[(p_arr == 1.0) & (dp_arr > 0.0)] = 0.0

            dp_arr *= stepsize / np.abs(dp_arr).max()


        if not is_converged:
            if prm['error_on_nonconvergence']:
                raise RuntimeError(error_message)
            else:
                logger.error('Iterations did not converge')

        return k_itr, is_converged, error_message


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


    def _enforce_phasefield_constraints(self, p_hat, p_cur):
        '''Correct `p_hat` to satisfy phasefield constraints.

        Parameters
        ----------
        p_hat : numpy.ndarray (1D)
            Estimated phasefield values.
        p_cur : numpy.ndarray (1D)
            Current phasefield values.

        '''

        # p_cur = self._p.vector().get_local()

        for C_i, dCdp_i, dp_C_i, abstol_C_i in zip(
                self._C, self._dCdp_arr, self._dp_C_arr, self._abstol_C):

            C_i = assemble(C_i)
            dp_C_i = dp_C_i.copy()
            R_i = C_i + dCdp_i.dot(p_hat-p_cur)

            ind_lwr = np.array((), bool)
            ind_upr = np.array((), bool)

            while abs(R_i) > abstol_C_i:

                dp_C_i[ind_lwr] = 0.0
                dp_C_i[ind_upr] = 0.0

                dRdp_C_i = dCdp_i.dot(dp_C_i)

                if abs(dRdp_C_i) < abstol_C_i:
                    raise RuntimeError('Can not enforce equality constraints.')

                dp_C_i *= -R_i/dRdp_C_i

                p_hat += dp_C_i

                ind_lwr = p_hat < 0.0
                ind_upr = p_hat > 1.0

                p_hat[ind_lwr] = 0.0
                p_hat[ind_upr] = 1.0

                R_i = C_i + dCdp_i.dot(p_hat-p_cur)
