''' Topology optimization based on phase-fields.

By Danas Sutula

TODO:
    - Acceleration of convergence using the history of the sesnitivities.
    - External phasefield evolution function could be beneficial. Adding it
    will decouple the potential energy and the regularizing penalty in the
    total cost functional.
    - When the cost functional is compoxed of multiple terms, the relative
    size of the terms becomes import. How to make it independent ? One way is
    to have a separate solver for the phase field -- we are back at it again.

NOTES:
    - hving the phasefield evolution inside the total cost functional requires
    to solve the adjoint problem. The relative magnitues of different terms in
    the total cost function matter; e.g. if the BC are chenged the these terms
    will growh appart.

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
    ========
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


class TopologyOptimizer:
    '''Minimize a cost functional.'''

    def __init__(self, W, P, p, p_mean, u, bcs_u, diffusivity=None,
        iteration_state_recording_function = lambda: None):
        '''
        diffusivity (float or dolfin.Constant): Diffusion-like coefficient for
            spatially smoothing out the cost gradient.

        W (dolfin.Form) : Potential energy to be minimized.
        P (dolfin.Form) : Penalty functional to be minimized.
        C (dolfin.Form) : Integral form of equality constraint. Note, `C` is
            assumed to be independent of `u`.

        '''

        # Displacement
        self._u = u

        # Phasefield
        self._p = p

        # Phasefield target mean
        self._p_mean = p_mean

        # Diffusion-like constant for spatially smoothing
        # discrete (nodal) phasefield dissipation rates
        if not isinstance(diffusivity, dolfin.Constant):
            if diffusivity is None: diffusivity = 0.0
            else: diffusivity = float(diffusivity)
            diffusivity = dolfin.Constant(diffusivity)

        self._diffusivity = diffusivity

        # Energy gradient as function (will need filtering)
        self._g = g = dolfin.Function(p.function_space())

        # Adjoint variable for computing cost sensitivity
        self._z = z = dolfin.Function(u.function_space())

        bcs_z = [dolfin.DirichletBC(bc) for bc in bcs_u]
        for bc in bcs_z: bc.homogenize() # zero bc dofs

        # Phasefield constraint
        C = (p - p_mean) * dolfin.dx

        # TODO
        # Constraint relative amount of phasefield diffusion
        C2 = (dolfin.grad(p)**2 - Constant(0.1)*p_mean) * dolfin.dx

        # Total cost
        J = W + P

        self._J = J
        self._W = W
        self._P = P
        self._C = C

        F    = dolfin.derivative(W, u)
        dFdu = dolfin.derivative(F, u)
        dFdp = dolfin.derivative(F, p)

        dJdp = dolfin.derivative(J, p)
        dPdu = dolfin.derivative(P, u)
        dCdp = dolfin.derivative(C, p)

        dJdu = dPdu # NOTE: Disregard `dWdu` (`F`)
        # because it will vanish for any solution

        # Total cost dissipation with respect to phasefield
        self._dJdp = dJdp - dolfin.action(dolfin.adjoint(dFdp), z)
        self._dCdp = dCdp # NOTE: assuming `C` is independent of `u`

        adjoint_problem = dolfin.LinearVariationalProblem(dFdu, dJdu, z, bcs_z)
        nonlinear_problem = dolfin.NonlinearVariationalProblem(F, u, bcs_u, dFdu)

        self.adjoint_solver = dolfin.LinearVariationalSolver(adjoint_problem)
        self.nonlinear_solver = dolfin.NonlinearVariationalSolver(nonlinear_problem)
        self.diffusion_filter = filter.make_diffusion_filter(g, diffusivity)

        utility.update_parameters(self.adjoint_solver.parameters, config.parameters_linear_solver)
        utility.update_parameters(self.nonlinear_solver.parameters, config.parameters_nonlinear_solver)
        utility.update_parameters(self.diffusion_filter.parameters, config.parameters_linear_solver)

        self.parameters_adjoint_solver = self.adjoint_solver.parameters
        self.parameters_nonlinear_solver = self.nonlinear_solver.parameters
        self.parameters_diffusion_filter = self.diffusion_filter.parameters

        self.parameters_topology_solver = config.parameters_topology_solver.copy()
        self.record_iteration_state = iteration_state_recording_function


    def optimize(self, stepsize, rtol=None, nmax=None):

        prm = self.parameters_topology_solver

        atol_J = prm['absolute_tolerance_energy']
        atol_C = prm['absolute_tolerance_constraint']
        rtol_p = prm['relative_tolerance_phasefield']

        maximum_iterations = prm['maximum_iterations']
        maximum_diverged   = prm['maximum_diverged']

        if stepsize <= 0.0:
            raise ValueError('Require positive `stepsize`')

        # Diffusion value for cost gradient smoothing
        diffusivity = float(self._diffusivity.values())

        # W_val_prv = np.inf
        J_val_prv = np.inf
        C_val_prv = np.inf

        g_vec = self._g.vector()
        p_vec = self._p.vector()

        p_arr = p_vec.get_local()
        dp_arr = p_arr.copy()

        normL1_p = p_arr.sum() # `abs` not needed
        normL2_dp = math.sqrt(dp_arr.dot(dp_arr))

        if not normL1_p: normL1_p = 1.0
        if not normL2_dp: normL2_dp = 1.0

        error_message = 'None'
        is_converged = False
        num_diverged = 0

        for k_itr in range(maximum_iterations):

            # Compute displacement field `u`
            _, b = self.nonlinear_solver.solve()

            if not b:
                print('\nERROR: Nonlinear problem could not be solved.\n')
                error_message = 'nonlinear_solver'
                is_converged = False
                break

            # Compute adjoint variable `z`
            self.adjoint_solver.solve()

            # W_val = assemble(self._W)
            J_val = assemble(self._J)
            C_val = assemble(self._C)

            dJdp_arr = assemble(self._dJdp).get_local()
            dCdp_arr = assemble(self._dCdp).get_local()

            if diffusivity:
                w_arr = filter.weight(p_arr)
                g_vec[:] = w_arr * dJdp_arr
                self.diffusion_filter.apply()
                dJdp_arr = g_vec.get_local()

            # User-defined recorder function
            self.record_iteration_state()


            ### Check convergence

            if J_val_prv < J_val + atol_J and \
              abs(C_val_prv) < abs(C_val) + atol_C:
                print('\nWARNING: Iteration diverged\n')
                if num_diverged == maximum_diverged:
                    print('\nERROR: Reached maximum times diverged\n')
                    error_message = 'maximum_times_diverged'
                    is_converged = False
                    break
                num_diverged += 1
                stepsize /= 2.0

            J_val_prv = J_val
            C_val_prv = C_val


            ### Update phasefield solution

            p_arr_prv = p_arr
            dp_arr_prv = dp_arr

            dJdp_max = np.abs(dJdp_arr).max()

            if not dJdp_max:
                if k_itr:
                    print('\nINFO: Zero dissipation (break)\n')
                    is_converged = True
                    break
                else:
                    print('\nERROR: Zero dissipation on first iteration\n')
                    error_message = 'zero_dissipation_on_first_iteration'
                    is_converged = False
                    break

            dp_arr = dJdp_arr * (-stepsize / dJdp_max)
            p_arr = p_arr + dp_arr


            ### Enforce phasefield bounds

            p_arr[p_arr < 0.0] = 0.0
            p_arr[p_arr > 1.0] = 1.0

            dp_arr = p_arr - p_arr_prv


            ### Enforce equality constraint

            # `dp_arr` correction vector
            dp_aux = np.ones_like(dp_arr)

            dot_dCdp_dp_aux = dCdp_arr.dot(dp_aux)

            while dot_dCdp_dp_aux:

                # Solve residual equation for correctional scale factor
                # C + \grad C \dot [\delta p + \delta p_aux scale] = 0
                # Apply scale factor on the correctional change dp_aux

                dp_aux *= -(C_val + dCdp_arr.dot(dp_arr)) / dot_dCdp_dp_aux
                # scale = math.sqrt(dp_arr.dot(dp_arr) / dp_aux.dot(dp_aux))

                # Aim for |dp_aux| <= |dp_arr|
                # if scale < 1.0: dp_aux *= scale

                # Apply correction
                p_arr += dp_aux

                mask_lwr = p_arr < 0.0
                mask_upr = p_arr > 1.0

                if np.any(mask_upr) or np.any(mask_lwr):
                    # Enforce phasefield constraints

                    p_arr[mask_lwr] = 0.0
                    p_arr[mask_upr] = 1.0

                    dp_aux[mask_lwr] = 0.0
                    dp_aux[mask_upr] = 0.0

                    dp_arr = p_arr - p_arr_prv

                else:
                    # Constraints enforced
                    dp_arr += dp_aux
                    break

                dot_dCdp_dp_aux = dCdp_arr.dot(dp_aux)

            else:
                raise RuntimeError('Can not enforce equality constraint')


            ### Phasefield change cosine similarity

            normL2_dp_prv = normL2_dp
            normL2_dp = math.sqrt(dp_arr.dot(dp_arr))
            cossim_dp = dp_arr.dot(dp_arr_prv) / (normL2_dp*normL2_dp_prv)


            ### Convergence assessment

            if abs(C_val) > atol_C:
                # assert np.all(p_arr>=0)
                normL1_p = p_arr.sum()

            normL1_dp = np.abs(dp_arr).sum()
            rchange_p = normL1_dp / normL1_p

            print('INFO: '
                  f'k:{k_itr:3d}, '
                  f'J:{J_val: 9.3e}, '
                  f'C:{C_val: 9.3e}, '
                  f'|dp|/|p|:{rchange_p: 8.2e}, '
                  f'cossim_dp:{cossim_dp: 0.3f}, '
                  f'stepsize:{stepsize: 0.3f} '
                  )

            assert p_arr.min() > PHASEFIELD_LOWER_BOUND
            assert p_arr.max() < PHASEFIELD_UPPER_BOUND

            if rchange_p < rtol_p:
                print('INFO: Negligible phase-field change (break)')
                is_converged = True # Should not update `p_vec`
                break

            p_vec[:] = p_arr


            ### Adapt phasefield stepsize

            # if cossim < cossim_refinement \
            #     and stepsize > stepsize_min + EPS:
            #     stepsize *= stepsize_decrease_factor
            #
            #     if stepsize > stepsize_min:
            #         print('INFO: Decrease step-size')
            #         # dp_arr *= stepsize_decrease_factor
            #
            #     else:
            #         print('INFO: Setting minimum step-size.')
            #         stepsize /= stepsize_decrease_factor
            #         # scaler = stepsize_min / stepsize
            #         stepsize = stepsize_min
            #         # dp_arr *= scaler
            #
            # elif cossim > cossim_coarsening \
            #     and stepsize < stepsize_max - EPS:
            #     stepsize *= stepsize_increase_factor
            #
            #     if stepsize < stepsize_max:
            #         print('INFO: Increase step-size.')
            #         # dp_arr *= stepsize_increase_factor
            #
            #     else:
            #         print('INFO: Setting maximum step-size.')
            #         stepsize /= stepsize_increase_factor
            #         # scaler = stepsize_max / stepsize
            #         stepsize = stepsize_max
            #         # dp_arr *= scaler


        else:

            print('\nWARNING: Reached maximum number of iterations\n')
            error_message = 'maximum_iterations'
            is_converged = False

        if not is_converged:
            if prm['error_on_nonconvergence']:
                raise RuntimeError('Iterations did not converge')
            else:
                print('\nERROR: Iterations did not converge\n')

        return k_itr, is_converged, error_message


    @property
    def diffusivity(self):
        '''Diffusion filter parameter.'''
        return float(self._diffusivity.values())

    @property
    def p_mean(self):
        '''Diffusion filter parameter.'''
        return float(self._p_mean.values())

    @diffusivity.setter
    def diffusivity(self, value):
        self._diffusivity.assign(float(value))

    @p_mean.setter
    def p_mean(self, value):
        self._p_mean.assign(float(value))
