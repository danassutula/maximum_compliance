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

'''

import os
import math
import dolfin
import numpy as np

from dolfin import assemble

from .config import parameters_linear_solver
from .config import parameters_nonlinear_solver
from .config import parameters_topology_solver


EPS = 1e-14

PHASEFIELD_LIM_LOWER = -EPS
PHASEFIELD_LIM_UPPER = 1.0 + EPS

DEGREES_TO_RADIANS = math.pi / 180.0

ITERATION_OUTFILE_NAME_U = "u"
ITERATION_OUTFILE_NAME_P = "p"

# TODO:
# form_compiler_parameters = []


def phasefield_filter(p, k0):
    '''As a minimization problem.

    For a given rough solution p0, find the smooth solution p that minimizes:
        J := 1/2 * ((p-p0)**2 + k0*grad(p)**2) * dx

    The minimization problem is solved by solving the associated stationary
    (variational) problem:
        F := dJdp = 0

    No boundary conditions are required for the solution.

    '''

    if not isinstance(k0, dolfin.Constant):
        k0 = dolfin.Constant(k0)

    if float(kappa.values()) < 0:
        raise ValueError('Require k0 > 0')

    dx = dolfin.dx
    dot = dolfin.dot
    grad = dolfin.grad

    # Unfiltered function
    p0 = p

    V = p.function_space()
    v = dolfin.TestFunction(V)
    p = dolfin.TrialFunction(V)

    # Minimization problem as a stationary problem
    F = ((p-p0)*v + k0*dot(grad(p),grad(v))) * dx

    a = dolfin.lhs(F) # bilinear form, a(p,v)
    L = dolfin.rhs(F) # linear form, L(v)

    return a, L


def phasefield_filter_residual(p, p0, k0):

    if not isinstance(k0, dolfin.Constant):
        k0 = dolfin.Constant(k0)

    if k0.values()[0] < 0:
        raise ValueError('Require k0 > 0')

    dx = dolfin.dx
    dot = dolfin.dot
    grad = dolfin.grad

    V = p.function_space()
    v = dolfin.TestFunction(V)
    p = dolfin.TrialFunction(V)

    # Minimization problem as a stationary problem
    F = ((p-p0)*v + k0*dot(grad(p),grad(v))) * dx

    a = dolfin.lhs(F) # bilinear form, a(p,v)
    L = dolfin.rhs(F) # linear form, L(v)

    return a, L


class TopologyOptimizer:
    '''Minimize a cost functional.'''

    def __init__(self, W, C, u, p, k0, bcs_u,
        iteration_state_recording_function = lambda : None):
        '''

        W (dolfin.Form) : Potential energy of solid
        J (dolfin.Form) : Cost functional to be minimized
        C (dolfin.Form) : Integral form of constraint equation

        '''

        if C is None:
            domain = p.function_space().mesh()
            p_mean = assemble(p*dolfin.dx(domain)) \
                   / assemble(1*dolfin.dx(domain))
            C = (p - p_mean) * dolfin.dx

        self._u = u
        self._p = p

        self._W = W # Potential energy
        self._C = C # Phasefield constraint

        self._dWdp = dolfin.derivative(W, p)
        self._dCdp = dolfin.derivative(C, p)

        F = dolfin.derivative(W, u)
        K = dolfin.derivative(F, u)

        nonlinear_problem = dolfin.NonlinearVariationalProblem(F, u, bcs_u, K)
        self.nonlinear_solver = dolfin.NonlinearVariationalSolver(nonlinear_problem)

        if not isinstance(k0, dolfin.Constant):
            k0 = dolfin.Constant(k0)

        self._k0 = k0

        a, L = phasefield_filter(p, k0)

        phasefield_problem = dolfin.LinearVariationalProblem(a, L, p, bcs=None)
        self.phasefield_solver = dolfin.LinearVariationalSolver(phasefield_problem)

        update_lhs_parameters_from_rhs(self.nonlinear_solver.parameters, parameters_nonlinear_solver)
        update_lhs_parameters_from_rhs(self.phasefield_solver.parameters, parameters_linear_solver)

        self.parameters_nonlinear_solver = self.nonlinear_solver.parameters
        self.parameters_linear_solver = self.phasefield_solver.parameters
        self.parameters_topology_solver = parameters_topology_solver.copy()
        self.record_iteration_state = iteration_state_recording_function

        if self.parameters_topology_solver['write_solutions']:
            self._iteration_outfile_u = dolfin_file(ITERATION_OUTFILE_NAME_U)
            self._iteration_outfile_p = dolfin_file(ITERATION_OUTFILE_NAME_P)
        else:
            self._iteration_outfile_u = None
            self._iteration_outfile_p = None


    def optimize(self, stepsize_ini=None, stepsize_min=None, stepsize_max=None):

        prm = self.parameters_topology_solver

        maximum_diverged         = prm['maximum_diverged']
        maximum_iterations       = prm['maximum_iterations']
        stepsize_decrease_factor = prm['stepsize_decrease_factor']
        stepsize_increase_factor = prm['stepsize_increase_factor']
        coarsening_delta_angle   = prm['coarsening_delta_angle']
        refinement_delta_angle   = prm['refinement_delta_angle']
        write_solutions          = prm['write_solutions']

        atol_W = prm['absolute_tolerance_energy']
        atol_C = prm['constraint_tolerance']
        rtol_p = prm['phasefield_tolerance']

        if write_solutions:
            if not self._iteration_outfile_u or not self._iteration_outfile_p:
                self._iteration_outfile_u = dolfin_file(ITERATION_OUTFILE_NAME_U)
                self._iteration_outfile_p = dolfin_file(ITERATION_OUTFILE_NAME_P)

        if stepsize_decrease_factor < 0.0 or stepsize_decrease_factor > 1.0:
            raise ValueError('Step-size decrease factor must be between 0.0 and 1.0')

        if stepsize_increase_factor < 1.0:
            raise ValueError('Step-size increase factor must be greater than 1.0')

        if coarsening_delta_angle > refinement_delta_angle:
            raise ValueError('`coarsening_delta_angle` must be '
                             'less than `refinement_delta_angle`')

        if stepsize_ini is None:
            stepsize_ini = prm['initial_stepsize']

        if stepsize_min is None:
            stepsize_min = prm['minimum_stepsize']

        if stepsize_max is None:
            stepsize_max = prm['maximum_stepsize']

        if EPS > stepsize_ini:
            raise ValueError('Require stepsize_ini > 0.0')

        if stepsize_min < 0.0:
            raise ValueError('Require stepsize_min >= 0.0')

        if stepsize_max > 1.0:
            raise ValueError('Require stepsize_max <= 1.0')

        if stepsize_min > stepsize_ini:
            print('INFO: Setting `stepsize_min` to `stepsize_ini`')
            stepsize_min = stepsize_ini

        if stepsize_max < stepsize_ini:
            print('INFO: Setting `stepsize_max` to `stepsize_ini`')
            stepsize_max = stepsize_ini

        coarsening_cossim = math.cos(coarsening_delta_angle * DEGREES_TO_RADIANS)
        refinement_cossim = math.cos(refinement_delta_angle * DEGREES_TO_RADIANS)

        W_val_prv = np.inf
        C_val_prv = np.inf

        dWdp_arr_prv = None
        sqrd_dWdp_prv = None

        p_vec = self._p.vector()
        p_arr = p_vec.get_local()

        stepsize = stepsize_ini
        error_message = 'None'
        is_converged = False
        num_diverged = 0

        for k_itr in range(maximum_iterations):

            _, b = self.nonlinear_solver.solve()

            if not b:
                print('\nERROR: Nonlinear problem could not be solved.\n')
                error_message = 'nonlinear_solver'
                is_converged = False
                break

            self.record_iteration_state()

            if write_solutions:
                self._iteration_outfile_u << self._u
                self._iteration_outfile_p << self._p

            W_val = assemble(self._W)
            C_val = assemble(self._C)

            dWdp_arr = assemble(self._dWdp).get_local()
            dCdp_arr = assemble(self._dCdp).get_local()


            ### Update phasefield Solution

            is_constraint_converging = \
                abs(C_val_prv) > abs(C_val) > atol_C

            if W_val_prv > W_val or is_constraint_converging:

                if W_val_prv - W_val < atol_W and abs(C_val) < atol_C:
                    print('INFO: Negligible change in cost functional')
                    is_converged = True
                    break

                # Scale for target dissipation
                sqrd_dWdp = dWdp_arr.dot(dWdp_arr)
                scale = W_val/sqrd_dWdp * stepsize

                # Phasefield value increment
                dp_arr = dWdp_arr * (-scale)

                if sqrd_dWdp_prv:
                    cossim = (dWdp_arr.dot(dWdp_arr_prv) /
                        math.sqrt(sqrd_dWdp * sqrd_dWdp_prv))

                else: # k_itr == 0
                    # Initialize `cossim` so that `stepsize` will not be changed
                    # Requirement: refinement_cossim < cossim < coarsening_cossim
                    cossim = coarsening_cossim - EPS

                W_val_prv = W_val
                C_val_prv = C_val

                dWdp_arr_prv = dWdp_arr
                sqrd_dWdp_prv = sqrd_dWdp

            else:

                print('\nWARNING: Solution diverged (try backtracking)\n')

                if stepsize < stepsize_min + EPS:
                    print('\nWARNING: Step-size already minimum (will not backtrack)\n')
                    error_message = 'phasefield_backtrace'
                    is_converged = False
                    break

                if num_diverged == maximum_diverged:
                    print('\nERROR: Reached maximum times diverged (will not backtrack)\n')
                    error_message = 'maximum_diverged'
                    is_converged = False
                    break

                else:
                    num_diverged += 1

                # Previous phasefield values
                p_arr -= dp_arr # old `dp_arr`

                # Signal refinement
                cossim = -1.0


            ### Adapt phasefield increment

            if cossim < refinement_cossim \
                and stepsize > stepsize_min + EPS:
                stepsize *= stepsize_decrease_factor

                if stepsize > stepsize_min:
                    print('INFO: Decrease step-size')
                    dp_arr *= stepsize_decrease_factor

                else:
                    print('INFO: Setting minimum step-size.')
                    stepsize /= stepsize_decrease_factor
                    scaler = stepsize_min / stepsize
                    stepsize = stepsize_min
                    dp_arr *= scaler

            elif cossim > coarsening_cossim \
                and stepsize < stepsize_max - EPS:
                stepsize *= stepsize_increase_factor

                if stepsize < stepsize_max:
                    print('INFO: Increase step-size.')
                    dp_arr *= stepsize_increase_factor

                else:
                    print('INFO: Setting maximum step-size.')
                    stepsize /= stepsize_increase_factor
                    scaler = stepsize_max / stepsize
                    stepsize = stepsize_max
                    dp_arr *= scaler


            ### Enforce equality constraint

            p_arr_ref = p_arr
            p_arr = p_arr + dp_arr
            dp_aux = dCdp_arr.copy()

            # To attempt corrections while nonzero
            dCdp_dot_dp_aux = dCdp_arr.dot(dp_aux)

            while dCdp_dot_dp_aux:

                # Solve residual equation for correctional scale factor
                # C + \grad C \dot [\delta p + \delta p_aux scale] = 0
                # Apply scale factor on the correctional change dp_aux

                dp_aux *= -(C_val + dCdp_arr.dot(dp_arr)) / dCdp_dot_dp_aux
                scale = math.sqrt(dp_arr.dot(dp_arr) / dp_aux.dot(dp_aux))

                # Aim for |dp_aux| <= |dp_arr|
                if scale < 1.0: dp_aux *= scale

                # Apply correction
                p_arr += dp_aux

                mask_lwr = p_arr < 0.0
                mask_upr = p_arr > 1.0

                if np.any(mask_lwr) or np.any(mask_upr):
                    # Need to enforce lower and upper bounds

                    p_arr[mask_lwr] = 0.0
                    p_arr[mask_upr] = 1.0

                    dp_aux[mask_lwr] = 0.0
                    dp_aux[mask_upr] = 0.0

                    # Corrected increment
                    dp_arr = p_arr - p_arr_ref

                    # To re-attempt correction if nonzero
                    dCdp_dot_dp_aux = dCdp_arr.dot(dp_aux)

                else:
                    # Successful constraint correction
                    dp_arr += dp_aux
                    break

            else:
                raise RuntimeError('Can not enforce equality constraint')

            # dp_arr = p_arr - p_arr_ref
            # p_arr = p_arr_ref
            #
            # p_arr_ref = p_arr
            # p_arr = p_arr + dp_arr

            # import ipdb; ipdb.set_trace()
            # self._k0.assign(float(self._k0))

            # Apply smoothing filter
            p_vec[:] = p_arr
            self.phasefield_solver.solve()
            p_arr = p_vec.get_local()

            # Enforce bounds
            p_arr[p_arr < 0.0] = 0.0
            p_arr[p_arr > 1.0] = 1.0

            dp_arr = p_arr - p_arr_ref

            assert p_arr.min() > PHASEFIELD_LIM_LOWER
            assert p_arr.max() < PHASEFIELD_LIM_UPPER

            print('INFO: '
                  f'k:{k_itr:3d}, '
                  f'W:{W_val: 9.3e}, '
                  f'C:{C_val: 9.3e}, '
                  f'stepsize:{stepsize: 8.2e}, '
                  f'dp_min:{dp_arr.min(): 8.2e}, '
                  f'dp_max:{dp_arr.max(): 8.2e}, '
                  )

            # Check phasefield convergence
            if dp_arr.dot(dp_arr) < p_arr.dot(p_arr) * rtol_p**2:
                print('INFO: Negligible phase-field change (break)')
                is_converged = True
                break

            p_vec[:] = p_arr

        else:

            print('\nWARNING: Reached maximum number of iterations\n')
            error_message = 'maximum_iterations'
            is_converged = False

        if not is_converged:
            if prm['error_on_nonconvergence']:
                raise RuntimeError('Iterations did not converge')
            else:
                print('\nERROR: Iterations did not converge\n')

        return is_converged, error_message


    @property
    def k0(self):
        return float(self._k0.values())

    @k0.setter
    def k0(self, value):
        self._k0.assign(float(value))


def update_lhs_parameters_from_rhs(lhs, rhs):
    '''Recursively update values of dict-like `lhs` with those in `rhs`.'''

    for k in rhs.keys():

        if k not in lhs.keys():
            raise KeyError(k) # NOTE: `k` in `rhs.keys()` is not in `lhs.keys()`

        if hasattr(lhs[k], 'keys'):

            if not hasattr(rhs[k], 'keys'):
                raise TypeError(f'`rhs[{k}]` must be dict-like.')
            else:
                update_lhs_parameters_from_rhs(lhs[k], rhs[k])

        elif hasattr(rhs[k], 'keys'):
            raise TypeError(f'`rhs[{k}]` can not be dict-like.')
        else:
            lhs[k] = rhs[k]


def dolfin_file(name, ext='pvd'):

    curdir = os.getcwd()

    outdir = os.path.join(curdir, 'out')
    if not os.path.exists(outdir): os.mkdir(outdir)

    outdir = os.path.join(outdir, 'tmp')
    if not os.path.exists(outdir): os.mkdir(outdir)

    outdir += os.sep
    for oldfile in os.listdir(outdir):
        oldfile = outdir + oldfile
        if os.path.isfile(oldfile):
            os.remove(oldfile)

    filepath = os.path.join(outdir, name)
    filepath += os.extsep + ext

    return dolfin.File(filepath)
