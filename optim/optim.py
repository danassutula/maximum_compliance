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
    - Mesh dependence, increasing diffusivity does not help; actually, it makes
    the problem worse, the diffusion becomes highly anisotropic. The remedy is
    to refine the mesh.

'''

import os
import time
import math
import dolfin
import numpy as np

from dolfin import assemble
from dolfin import project
from dolfin import solve

# TEMP
import matplotlib.pyplot as plt
plt.interactive(True)

# TEMP
def plot(*args, **kwargs):
    '''Plot either `np.ndarray`s or something plottable from `dolfin`.'''
    plt.figure(kwargs.pop('name', None))
    if isinstance(args[0], (np.ndarray,list, tuple)):
        plt.plot(*args, **kwargs)
    else: # just try it anyway
        dolfin.plot(*args, **kwargs)
    plt.show()

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

# NOTE: Assigning a vector rather than array is significantly faster


class TopologyOptimizer:
    '''Minimize a cost functional.'''

    def __init__(self, W, P, C, p, ps, u, bcs_u,
                 weight_P, kappa_W, kappa_P, kappa_I,
                 recorder_function = lambda: None):
        '''

        Parameters
        ----------
        W : dolfin.Form
            Potential energy to be minimized.
        P : dolfin.Form
            Penalty energy to be minimized.
        C : dolfin.Form or a sequence of dolfin.Form's
            Equality constraint(s) as integral equations.
        p : dolfin.Function
            Global phasefield function.
        ps : sequence of dolfin.Function's
            Local phasefield functions.
        weight_P : float
            Penalty energy weight factor relative to strain energy.
        kappa_W : float or dolfin.Constant
            Diffusion-like coefficient for smoothing the energy gradient.
        kappa_P : float or dolfin.Constant
            Diffusion-like coefficient for smoothing the penalty gradient.
        kappa_I : float or dolfin.Constant
            Diffusion-like coefficient for detecting the interaction among
            phasefields by spreading out the phasefields and detecting overlap.

        Notes
        -----
        The equality constraints `C` are assumed to be linear in `p` and
        independent of `u`.

        '''

        if not -EPS < weight_P < 1.0 + EPS:
            raise ValueError('Parameter `weight_P`.')

        # Sequence of local phasefields
        if isinstance(ps, (list, tuple)) and all(
           isinstance(ps_i, dolfin.Function) for ps_i in ps):
            self._ps = ps if isinstance(ps, tuple) else tuple(ps)

        elif isinstance(ps, dolfin.Function):
            self._ps = (ps,)

        else:
            raise TypeError('Parameter `ps` is neither a `dolfin.Function` '
                            'nor a sequence of `dolfin.Function`s.')

        # Global phasefield
        p.assign(sum(ps))

        if np.any(p.vector() < PHASEFIELD_LOWER_BOUND): raise ValueError
        if np.any(p.vector() > PHASEFIELD_UPPER_BOUND): raise ValueError

        self._p = p
        self._u = u

        if not isinstance(kappa_W, dolfin.Constant):
            kappa_W = dolfin.Constant(kappa_W)

        if not isinstance(kappa_P, dolfin.Constant):
            kappa_P = dolfin.Constant(kappa_P)

        if not isinstance(kappa_I, dolfin.Constant):
            kappa_I = dolfin.Constant(kappa_I)

        self._require_filtering_W = bool(float(kappa_W.values()))
        self._require_filtering_P = bool(float(kappa_P.values()))
        self._require_filtering_I = bool(float(kappa_I.values()))

        self.weight_P = weight_P

        self._W = W
        self._P = P

        self._dWdp = dolfin.derivative(W, p)
        self._dPdp = dolfin.derivative(P, p)

        if isinstance(C, (list, tuple)):
            self._C = C if isinstance(C, tuple) else tuple(C)
            self._dCdp = tuple(dolfin.derivative(Ci, p) for Ci in C)
        else:
            self._C = (C,)
            self._dCdp = (dolfin.derivative(C, p),)

        # Assuming constraints are linear (i.e. gradients are constant vectors)
        self._dCdp_arr = [assemble(dCidp).get_local() for dCidp in self._dCdp]
        self._dp_C_arr = self._constraint_correction_vectors(self._dCdp_arr)

        F = dolfin.derivative(W, u)
        dFdu = dolfin.derivative(F, u)

        nonlinear_problem = dolfin.NonlinearVariationalProblem(F, u, bcs_u, dFdu)
        self.nonlinear_solver = dolfin.NonlinearVariationalSolver(nonlinear_problem)

        v = dolfin.TestFunction(p.function_space())
        f = dolfin.TrialFunction(p.function_space())

        self._filter_KW = assemble((f * v + kappa_W * dolfin.dot(dolfin.grad(f), dolfin.grad(v))) * dolfin.dx)
        self._filter_KP = assemble((f * v + kappa_P * dolfin.dot(dolfin.grad(f), dolfin.grad(v))) * dolfin.dx)
        self._filter_KI = assemble((f * v + kappa_I * dolfin.dot(dolfin.grad(f), dolfin.grad(v))) * dolfin.dx)

        self._f = dolfin.Function(p.function_space())
        self._filter_rhs = self._f * v * dolfin.dx

        utility.update_lhs_parameters_from_rhs(
            self.nonlinear_solver.parameters,
            config.parameters_nonlinear_solver)

        self.parameters_topology_solver = \
            config.parameters_topology_solver.copy()

        self.recorder_function = recorder_function


    def optimize(self, stepsize, phasefield_tolerance=None,
        influence_threshold=None, maximum_iterations=None,
        maximum_divergences=None):

        if stepsize <= 0.0:
            raise ValueError('Require positive `stepsize`')

        prm = self.parameters_topology_solver

        if phasefield_tolerance is None:
            phasefield_tolerance = prm['phasefield_tolerance']

        if influence_threshold is None:
            influence_threshold = prm['influence_threshold']

        if maximum_iterations is None:
            maximum_iterations = prm['maximum_iterations']

        if maximum_divergences is None:
            maximum_divergences = prm['maximum_divergences']

        weight_P = self.weight_P
        weight_W = 1.0 - weight_P

        dCdp_arr = self._dCdp_arr
        dp_C_arr = self._dp_C_arr

        rtol_C = prm['constraint_tolerance']
        atol_C = [rtol_C * np.abs(dCidp_arr).sum() for dCidp_arr in dCdp_arr]

        p_vec = self._p.vector()
        f_vec = self._f.vector()

        p_arr = p_vec.get_local()
        dp_arr = p_arr.copy()

        W_val_prv = np.inf
        normL2_dp = np.inf

        ps_vec = [p_i.vector() for p_i in self._ps]
        ws_arr = np.empty((len(ps_vec), len(p_vec))) # diffused phasefield values
        ms_arr = np.empty(ws_arr.shape, dtype=bool)  # phasefield activity masks

        divergences_count = 0
        is_converged = False
        error_message = ''

        # Compute initial displacements `u`
        _, b = self.nonlinear_solver.solve()

        if not b:
            raise RuntimeError('Unable to solve nonlinear problem')

        for k_itr in range(maximum_iterations):

            W_val = assemble(self._W)
            C_val = [assemble(C_i) for C_i in self._C]

            dWdp_arr = assemble(self._dWdp).get_local()
            dPdp_arr = assemble(self._dPdp).get_local()

            if self._require_filtering_W:

                f_vec[:] = dWdp_arr
                solve(self._filter_KW, f_vec, f_vec)
                dWdp_arr = f_vec.get_local()

            if self._require_filtering_P:

                f_vec[:] = dPdp_arr
                solve(self._filter_KP, f_vec, f_vec)
                dPdp_arr = f_vec.get_local()

            # User-defined recorder
            self.recorder_function()


            ### Check convergence

            if W_val_prv < W_val and all(abs(C_val_i) < atol_C_i
                for C_val_i, atol_C_i in zip(C_val, atol_C)):

                print('INFO: Iteration diverged.')
                divergences_count += 1

                if divergences_count > maximum_divergences:
                    print('INFO: Refining stepsize.')
                    divergences_count = 0
                    stepsize /= 2.0

            W_val_prv  = W_val
            p_arr_prv  = p_arr
            dp_arr_prv = dp_arr


            ### Estimate phasefield change

            # dWdp_arr /= np.abs(dWdp_arr).max() + EPS
            # dPdp_arr /= np.abs(dPdp_arr).max() + EPS

            dWdp_arr /= (math.sqrt(dWdp_arr.dot(dWdp_arr)) + EPS)
            dPdp_arr /= (math.sqrt(dPdp_arr.dot(dPdp_arr)) + EPS)

            dp_arr = dWdp_arr*weight_W + dPdp_arr*weight_P
            dp_arr *= (-stepsize) / np.abs(dp_arr).max()
            p_arr = p_arr + dp_arr


            ### Enforce phasefield bounds

            p_arr[p_arr < 0.0] = 0.0
            p_arr[p_arr > 1.0] = 1.0

            dp_arr = p_arr - p_arr_prv


            ### Phasefield activity weights

            for i, p_vec_i in enumerate(ps_vec):

                f_vec[:] = p_vec_i
                rhs = assemble(self._filter_rhs)
                solve(self._filter_KI, f_vec, rhs)
                ws_arr[i,:] = f_vec.get_local()


            ### Mark maximum-value phasefields as active

            # Find maximum value phasefields
            argmax_ws = ws_arr.argmax(axis=0)

            for i in range(len(ps_vec)):
                ms_arr[i,:] = argmax_ws == i


            ### Enforce zero phasefield change for competing phasefields

            competing_phasefields = \
                (ws_arr > influence_threshold).sum(axis=0) > 1

            ms_arr[:,competing_phasefields] = False
            phasefield_inactivity = ~ms_arr.any(axis=0)

            # All phasefield activities should be orthogonal
            assert all(ms_arr[i,:].dot(ms_arr[j,:]) < 1e-12
                       for i in range(len(ps_vec)-1)
                       for j in range(i+1, len(ps_vec)))

            p_arr_prv[phasefield_inactivity] = 0.0
            dp_arr[phasefield_inactivity] = 0.0
            p_arr = p_arr_prv + dp_arr


            ### Enforce equality constraint(s)

            self._enforce_equality_constraints(dp_arr, p_arr,
                p_arr_prv, dp_C_arr, dCdp_arr, C_val, atol_C)

            assert p_arr.min() > PHASEFIELD_LOWER_BOUND
            assert p_arr.max() < PHASEFIELD_UPPER_BOUND


            ### Assign updated phasefields

            for p_vec_i, ms_arr_i in zip(ps_vec, ms_arr):

                p_arr_i = np.zeros((len(p_arr),))
                p_arr_i[ms_arr_i] = p_arr[ms_arr_i]
                p_vec_i[:] = p_arr_i


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
                  f'W:{W_val: 11.5e}, '
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
        dp_C = [A_i.astype(float) for A_i in A]

        assert all(abs(v_i.dot(v_j)) < EPS*min(v_j.dot(v_j), v_i.dot(v_i))
            for i, v_i in enumerate(dp_C[:-1]) for v_j in dp_C[i+1:]), \
            'Constraint correction vectors are not mutually orthogonal.'

        return dp_C


    @staticmethod
    def _enforce_equality_constraints(dp, p, p_prv, dp_C, dCdp, C, atol_C):
        '''Correct `p` and `dp` so that the constraints `C` are satisfied.

        Parameters
        ----------
        p : numpy.ndarray (1D)
            Nodal phasefield values.
        dp : numpy.ndarray (1D)
            Increment in the nodel phasefield values.
        dp_C : sequence of numpy.ndarray (1D)
            Initial (trial) phasefield corrections due to constraints.
        dCdp : sequence of numpy.ndarray (1D)
            Gradients of the equality constraint equations.
        C : sequence of float's
            Residual values of the constraint equations.
        atol_C : sequence of float's
            Convergence tolerances (absolute) for the constraint equations.

        Returns
        -------
        None

        '''

        # Require copy because it will be mutated
        dp_C = (dp_C_i.copy() for dp_C_i in dp_C)

        for dp_C_i, dCdp_i, C_i, atol_C_i in zip(dp_C, dCdp, C, atol_C):

            R_i = C_i + dCdp_i.dot(dp)
            while abs(R_i) > atol_C_i:

                dRdp_C_i = dCdp_i.dot(dp_C_i)
                if abs(dRdp_C_i) < atol_C_i:
                    print('ERROR: Equality constraint can not be enforced.')
                    break

                # Magnitude correction
                dp_C_i *= -R_i/dRdp_C_i

                # Superpose correction
                p += dp_C_i

                mask_lwr = p < 0.0
                mask_upr = p > 1.0

                # Enforce phasefield constraints
                if np.any(mask_upr) or np.any(mask_lwr):

                    p[mask_lwr] = 0.0
                    p[mask_upr] = 1.0

                    dp_C_i[mask_lwr] = 0.0
                    dp_C_i[mask_upr] = 0.0

                    dp[:] = p
                    dp -= p_prv

                else:

                    dp += dp_C_i
                    break

                R_i = C_i + dCdp_i.dot(dp)