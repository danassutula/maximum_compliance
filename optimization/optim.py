'''Topology optimization based on phase-fields. (biomech-optimtop)

By Danas Sutula
University of Liberec, Czech Republic, 2018-2019

TODO:
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
    dCdu := dolfin.derivative(C, u) = 0

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
    - Mesh dependence, increasing diffusivity does not hely; actually, it makes
    the problem worse, the diffusion becomes highly anisotropic. The remedy is
    to refine the mesh.

'''

import os
import time
import math
import dolfin
import numpy as np

from dolfin import assemble

from . import config
from . import filter
from . import utility

EPS = 1e-14
PI = math.pi

PHASEFIELD_LOWER_BOUND = -EPS
PHASEFIELD_UPPER_BOUND = 1.0 + EPS
DEGREES_TO_RADIANS =  PI / 180.0

# TODO:
# form_compiler_parameters = []


class TopologyOptimizer:
    '''Minimize a cost functional.'''

    def __init__(self, J, W, C, p, u, bcs_u, kappa=0.0,
        iteration_state_recording_function = lambda: None,
        phasefields=None):
        '''

        Parameters
        ----------
        J : dolfin.Form
            Cost functional to be minimized.
        W : dolfin.Form
            Potential energy to be minimized.
        C : dolfin.Form
            Integral form of equality constraint. Note, `C` is assumed to be
            independent of `u`.
        D : dolfin.Form
            Filter problem.
        kappa : float or dolfin.Constant (optional)
            Diffusion-like coefficient for smoothing the cost gradient.

        '''

        # TEMP
        self._phasefields = phasefields
        self._phasefields_velocity_expr = [grad(p_i)**2 for p_i in phasefields]

        # Phasefield
        self._p = p

        # Displacements
        self._u = u

        # Generic function to be passed to a filter
        self._f = dolfin.Function(p.function_space())

        # Diffusion-like constant for smoothing nodal dissipation
        self._kappa = kappa if isinstance(kappa, dolfin.Constant) \
                            else dolfin.Constant(kappa)

        self._J = J
        self._dJdp = dolfin.derivative(J, p)

        if isinstance(C, (list, tuple)):
            self._C = C if isinstance(C, tuple) else tuple(C)
            self._dCdp = tuple(dolfin.derivative(C_i, p) for C_i in C)
        else:
            self._C = (C,)
            self._dCdp = (dolfin.derivative(C, p),)

        F = dolfin.derivative(W, u)
        dFdu = dolfin.derivative(F, u)

        nonlinear_problem = dolfin.NonlinearVariationalProblem(F, u, bcs_u, dFdu)
        self.nonlinear_solver = dolfin.NonlinearVariationalSolver(nonlinear_problem)
        self.diffusion_filter = filter.make_diffusion_filter(self._f, self._kappa)

        utility.update_lhs_parameters_from_rhs(
            self.nonlinear_solver.parameters,
            config.parameters_nonlinear_solver)

        utility.update_lhs_parameters_from_rhs(
            self.diffusion_filter.parameters,
            config.parameters_linear_solver)

        self.parameters_nonlinear_solver = self.nonlinear_solver.parameters
        self.parameters_diffusion_filter = self.diffusion_filter.parameters
        self.parameters_topology_solver = config.parameters_topology_solver.copy()
        self.record_iteration_state = iteration_state_recording_function

    def optimize(self, stepsize, phasefield_tolerance=None,
        maximum_divergences=None, maximum_iterations=None):

        if stepsize <= 0.0:
            raise ValueError('Require positive `stepsize`')

        prm = self.parameters_topology_solver

        if phasefield_tolerance is None:
            phasefield_tolerance = prm['phasefield_tolerance']

        if maximum_divergences is None:
            maximum_divergences = prm['maximum_divergences']

        if maximum_iterations is None:
            maximum_iterations = prm['maximum_iterations']

        rtol_C = prm['constraint_tolerance']
        atol_C = [rtol_C * np.abs(assemble(dCdp_i)[:]).sum()
                  for dCdp_i in self._dCdp]

        # Whether to smooth the cost gradient `dJdp`
        require_filtering = bool(self._kappa.values())

        J_val_prv = np.inf

        p_vec = self._p.vector()
        f_vec = self._f.vector()


        # TEMP
        ps_arr = self._phasefields_dof
        dps_arr = [p_i.copy() for p_i in ps_arr]


        p_arr = p_vec.get_local()
        dp_arr = p_arr.copy()
        normL2_dp = np.inf

        divergences_count = 0
        is_converged = False
        error_message = ''

        # Compute initial displacements `u`
        _, b = self.nonlinear_solver.solve()

        if not b:
            raise RuntimeError('Unable to solve nonlinear problem')

        for k_itr in range(maximum_iterations):

            # Compute adjoint solution `z`
            # self.adjoint_solver.solve()

            J_val = assemble(self._J)
            C_val = [assemble(C_i) for C_i in self._C]

            dJdp_arr = assemble(self._dJdp).get_local()
            dCdp_arr = [assemble(dCdp_i).get_local() for dCdp_i in self._dCdp]

            if require_filtering:

                f_vec[:] = dJdp_arr
                self.diffusion_filter.apply()
                dJdp_arr = f_vec.get_local()

                # ind_nnz = np.flatnonzero(dJdp_arr)
                # dJdp_arr[ind_nnz] = f_vec[ind_nnz]

            # User-defined recorder function
            self.record_iteration_state()


            ### Check convergence

            if J_val_prv < J_val and all(abs(C_val_i) < atol_C_i
                for C_val_i, atol_C_i in zip(C_val, atol_C)):

                print('INFO: Iteration diverged.')
                divergences_count += 1

                if divergences_count > maximum_divergences:
                    print('INFO: Refining stepsize.')
                    divergences_count = 0
                    stepsize /= 2.0

            J_val_prv  = J_val
            p_arr_prv  = p_arr
            dp_arr_prv = dp_arr


            ### Estimate phasefield change

            # Compute nodal phasefield change `dp_arr`
            # such that norm(dp_arr, inf) -> stepsize

            # Dissipation potentiak

            dJdp_max = np.abs(dJdp_arr).max() + EPS
            dp_arr = dJdp_arr * (-stepsize / dJdp_max)

            p_arr = p_arr + dp_arr


            ### Enforce phasefield bounds

            p_arr[p_arr < 0.0] = 0.0
            p_arr[p_arr > 1.0] = 1.0

            dp_arr = p_arr - p_arr_prv


            ### Enforce equality constraint

            # Constraint correction vectors (initialize)
            # msk_dp_aux = [dCdp_arr_i.astype(bool) for dCdp_arr_i in dCdp_arr]
            # ind_dp_aux = [np.flatnonzero(msk_i) for msk_i in msk_dp_aux]
            # dp_aux = [msk_i.astype(float) for msk_i in msk_dp_aux]

            dp_aux = [dCdp_arr_i.astype(bool).astype(float)
                      for dCdp_arr_i in dCdp_arr]

            # Need to prescribe zero change in some regionsself.

            # Vectors must be exactly orthogonal
            assert all(dp_aux[i].dot(dp_aux_j) == 0.0
                       for i in range(len(dp_aux)-1)
                       for dp_aux_j in dp_aux[i+1:])

            for C_val_i, dCdp_arr_i, dp_aux_i, atol_C_i \
                in zip(C_val, dCdp_arr, dp_aux, atol_C):

                dCdp_aux_i = dCdp_arr_i.dot(dp_aux_i)
                while abs(dCdp_aux_i) > atol_C_i:

                    # Solve residual equation for correct magnitude of `dp_aux_i`
                    dp_aux_i *= -(C_val_i + dCdp_arr_i.dot(dp_arr)) / dCdp_aux_i

                    # Superpose correction
                    p_arr += dp_aux_i

                    mask_lwr = p_arr < 0.0
                    mask_upr = p_arr > 1.0

                    # Enforce phasefield constraints
                    if np.any(mask_upr) or np.any(mask_lwr):

                        p_arr[mask_lwr] = 0.0
                        p_arr[mask_upr] = 1.0

                        dp_aux_i[mask_lwr] = 0.0
                        dp_aux_i[mask_upr] = 0.0

                        dp_arr = p_arr - p_arr_prv

                    else:

                        dp_arr += dp_aux_i
                        break

                    dCdp_aux_i = dCdp_arr_i.dot(dp_aux_i)

                else:
                    print('\nERROR: Unable to enforce equality constraint\n')
                    error_message = 'equality_constraint'
                    break

            assert p_arr.min() > PHASEFIELD_LOWER_BOUND
            assert p_arr.max() < PHASEFIELD_UPPER_BOUND


            ### Distribute phasefield

            # TODO


            ### Phasefield change cosine similarity

            normL2_dp_prv = normL2_dp
            normL2_dp = math.sqrt(dp_arr.dot(dp_arr))
            cossim_dp = dp_arr.dot(dp_arr_prv) / (normL2_dp*normL2_dp_prv)


            ### Convergence assessment

            normL1_p = p_arr.sum()
            normL1_dp = np.abs(dp_arr).sum()
            phasefield_change = normL1_dp / normL1_p

            print('INFO: '
                  f'k:{k_itr:3d}, '
                  f'J:{J_val: 11.5e}, '
                  f'C:{np.abs(C_val).max(): 9.3e}, '
                  f'|dp|/|p|:{phasefield_change: 8.2e}, '
                  f'cossim_dp:{cossim_dp: 0.3f}, '
                  f'stepsize:{stepsize: 0.4f} '
                  )

            # Update phasefield
            p_vec[:] = p_arr

            # Update displacement solution `u`
            _, b = self.nonlinear_solver.solve()

            if not b:
                print('\nERROR: Unable to solve nonlinear problem\n')
                error_message = 'nonlinear_solver'
                break

            if phasefield_change < phasefield_tolerance:
                print('INFO: Negligible phase-field change (break)')
                is_converged = True # Do not update phasefield again
                break

        else:

            print('\nWARNING: Reached maximum number of iterations\n')
            error_message = 'maximum_iterations'
            is_converged = False

        if not is_converged:
            if prm['error_on_nonconvergence']:
                raise RuntimeError(error_message)
            else:
                print('\nERROR: Iterations did not converge\n')

        return k_itr, is_converged, error_message


    @property
    def kappa(self):
        '''Diffusion filter parameter.'''
        return float(self._kappa.values())

    @kappa.setter
    def kappa(self, value):
        self._kappa.assign(float(value))

    # @property
    # def p_mean(self):
    #     '''Diffusion filter parameter.'''
    #     return float(self._p_mean.values())

    # @p_mean.setter
    # def p_mean(self, value):
    #     self._p_mean.assign(float(value))
