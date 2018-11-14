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

'''

import os
import math
import dolfin
import numpy as np

from dolfin import assemble

from .config import parameters_adjoint_solver
from .config import parameters_nonlinear_solver
from .config import parameters_topology_solver

EPS = 1e-14
PHASEFIELD_LIM_LOWER = -EPS
PHASEFIELD_LIM_UPPER = 1.0 + EPS
RADIANS_PER_DEGREE = math.pi / 180.0
ITERATION_OUTFILE_NAME_U = "u"
ITERATION_OUTFILE_NAME_P = "p"

ERROR_ENUMERATION = {
    'nonlinear_solver': 1,
    'phasefield_backtrace': 2,
    'maximum_iterations': 3,
    }

ERROR_NUMBER_TO_NAME = {v : k for k, v
    in ERROR_ENUMERATION.items()}


class TopologyOptimizer:
    '''Minimize a cost functional.'''

    def __init__(self, W, J, C, u, p, bcs_u,
        iteration_state_recording_function = lambda : None):
        '''

        W (dolfin.Form) : Potential energy of solid
        J (dolfin.Form) : Cost functional to be minimized
        C (dolfin.Form) : Integral form of constraint equation

        '''

        self._u = u
        self._p = p

        self._z = z = dolfin.Function(u.function_space())
        bcs_z = [dolfin.DirichletBC(bc) for bc in bcs_u]
        for bc in bcs_z: bc.homogenize() # zero bc dofs

        if C is None:
            mesh = p.function_space().mesh()
            p_mean = assemble(p*dolfin.dx) / assemble(1*dolfin.dx(mesh))
            C = (p - p_mean) * dolfin.dx

        self._J = J
        self._C = C

        F    = dolfin.derivative(W, u)
        dFdu = dolfin.derivative(F, u)
        dFdp = dolfin.derivative(F, p)

        dJdu = dolfin.derivative(J, u)
        dJdp = dolfin.derivative(J, p)
        dCdp = dolfin.derivative(C, p)

        # Total energy dissipation with respect to phasefield
        self._dJdp = dJdp - dolfin.action(dolfin.adjoint(dFdp), z)
        self._dCdp = dCdp # NOTE: assuming `C` is independent of `u`

        self._DJ = dolfin.Function(p.function_space()) # FIXME
        self._Dp = dolfin.Function(p.function_space()) # FIXME

        # NOTE: `dFdu` is equivalent to `adjoint(dFdu)` because of symmetry
        adjoint_problem = dolfin.LinearVariationalProblem(dFdu, dJdu, z, bcs_z)
        nonlinear_problem = dolfin.NonlinearVariationalProblem(F, u, bcs_u, dFdu)
        # phasefield_problem = # TODO or NOT TODO ?

        self.adjoint_solver = dolfin.LinearVariationalSolver(adjoint_problem)
        update_parameters(self.adjoint_solver.parameters, parameters_adjoint_solver)
        self.parameters_adjoint_solver = self.adjoint_solver.parameters

        self.nonlinear_solver = dolfin.NonlinearVariationalSolver(nonlinear_problem)
        update_parameters(self.nonlinear_solver.parameters, parameters_nonlinear_solver)
        self.parameters_nonlinear_solver = self.nonlinear_solver.parameters

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

        rtol_p = prm['relative_tolerance_phasefield']
        atol_C = prm['absolute_tolerance_constraint']
        atol_J = prm['absolute_tolerance_energy']

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

        if EPS > stepsize_ini:
            raise ValueError('Require stepsize_ini > 0.0')

        if stepsize_min is None:
            stepsize_min = prm['minimum_stepsize']

        if stepsize_min < 0.0:
            raise ValueError('Require stepsize_min >= 0.0')

        if stepsize_max is None:
            stepsize_max = prm['maximum_stepsize']

        if stepsize_max > 1.0:
            raise ValueError('Require stepsize_max <= 1.0')

        if stepsize_min > stepsize_ini:
            print('INFO: Setting `stepsize_min` to `stepsize_ini`')
            stepsize_min = stepsize_ini

        if stepsize_max < stepsize_ini:
            print('INFO: Setting `stepsize_max` to `stepsize_ini`')
            stepsize_max = stepsize_ini

        coarsening_cossim = math.cos(RADIANS_PER_DEGREE * coarsening_delta_angle)
        refinement_cossim = math.cos(RADIANS_PER_DEGREE * refinement_delta_angle)

        J_val_prv = np.inf
        C_val_prv = np.inf

        dJdp_arr_prv = None
        norm_dJdp_prv = None

        p_vec = self._p.vector()
        p_arr = p_vec.get_local()

        stepsize = stepsize_ini
        is_converged = False
        num_diverged = 0
        error_number = 0

        for k_itr in range(maximum_iterations):

            _, b = self.nonlinear_solver.solve()

            if not b:
                print('\nERROR: Nonlinear problem could not be solved.\n')
                error_number = ERROR_ENUMERATION['nonlinear_solver']
                is_converged = False
                break

            self.adjoint_solver.solve()
            self.record_iteration_state()

            if write_solutions:
                self._iteration_outfile_u << self._u
                self._iteration_outfile_p << self._p

            J_val = assemble(self._J)
            C_val = assemble(self._C)

            dJdp_arr = assemble(self._dJdp).get_local()
            dCdp_arr = assemble(self._dCdp).get_local()

            # Scale for target dissipation
            norm_dJdp = dJdp_arr.dot(dJdp_arr)
            scale = stepsize*J_val/norm_dJdp

            # Advance in opposite direction
            dp_arr = dJdp_arr * (-scale)


            ### Check convergence

            is_constraint_converging = \
                abs(C_val_prv) > abs(C_val) > atol_C

            if J_val_prv > J_val or is_constraint_converging:

                if J_val_prv - J_val < atol_J and abs(C_val) < atol_C:
                    print('INFO: Negligible change in cost functional')
                    is_converged = True
                    break

                if norm_dJdp_prv:
                    cossim = (dJdp_arr.dot(dJdp_arr_prv) /
                        math.sqrt(norm_dJdp * norm_dJdp_prv))

                else: # k_itr == 0
                    # Initialize `cossim` so that `stepsize` will not be changed
                    # Requirement: refinement_cossim < cossim < coarsening_cossim
                    cossim = coarsening_cossim - EPS

                J_val_prv = J_val
                C_val_prv = C_val

                dJdp_arr_prv = dJdp_arr
                norm_dJdp_prv = norm_dJdp

            else:

                print('\nWARNING: Solution diverged (try backtracking)\n')

                if stepsize < stepsize_min + EPS:
                    print('\nWARNING: Step-size already minimum (will not backtrack)\n')
                    error_number = ERROR_ENUMERATION['phasefield_backtrace']
                    is_converged = False
                    break

                if num_diverged == maximum_diverged:
                    print('\nERROR: Reached maximum times diverged (will not backtrack)\n')
                    error_number = ERROR_ENUMERATION['phasefield_backtrace']
                    is_converged = False
                    break

                p_arr -= dp_arr_prv
                dp_arr = dp_arr_prv

                num_diverged += 1
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

            # Correction for dp_arr
            dp_aux = dCdp_arr.copy()

            # To attempt corrections while nonzero
            dCdp_dot_dp_aux = dCdp_arr.dot(dp_aux)

            while dCdp_dot_dp_aux:

                # Solve residual equation for correctional scale factor
                # C + \grad C \dot [\delta p - \delta p_aux scale] = 0
                # Apply scale factor on the correctional change dp_aux

                dp_aux *= (C_val + dCdp_arr.dot(dp_arr)) / dCdp_dot_dp_aux
                scale = math.sqrt(dp_arr.dot(dp_arr) / dp_aux.dot(dp_aux))

                # Aim for |dp_aux| <= |dp_arr|
                if scale < 1.0: dp_aux *= scale

                # Apply correction
                dp_arr -= dp_aux

                # Masks for lower and upper bounds
                mask_upr = p_arr + dp_arr > PHASEFIELD_LIM_UPPER
                mask_lwr = p_arr + dp_arr < PHASEFIELD_LIM_LOWER

                # Enforce lower and upper bounds
                if np.any(mask_upr) or np.any(mask_lwr):

                    dp_arr[mask_upr] = 1.0-p_arr[mask_upr]
                    dp_arr[mask_lwr] = -p_arr[mask_lwr]

                    dp_aux[mask_upr] = 0.0
                    dp_aux[mask_lwr] = 0.0

                else:
                    # Successful constraint correction
                    break

                # To re-attempt correction if nonzero
                dCdp_dot_dp_aux = dCdp_arr.dot(dp_aux)

            else:
                print('ERROR: Can not enforce equality constraint')
                error_number = ERROR_ENUMERATION['constraint_correction']
                is_converged = False
                break

            print('INFO: '
                  f'k:{k_itr:3d}, '
                  f'J:{J_val: 9.3e}, '
                  f'C:{C_val: 9.3e}, '
                  f'cossim:{cossim: 8.2e}, '
                  f'stepsize:{stepsize: 8.2e}, '
                  )

            # Check phasefield convergence
            if dp_arr.dot(dp_arr) < p_arr.dot(p_arr) * rtol_p**2:
                print('INFO: Negligible phase-field change (break)')
                is_converged = True
                break

            p_arr[:] += dp_arr
            dp_arr_prv = dp_arr
            p_vec.set_local(p_arr)

            assert p_arr.min() > PHASEFIELD_LIM_LOWER
            assert p_arr.max() < PHASEFIELD_LIM_UPPER

        else:

            print('\nWARNING: Reached maximum number of iterations\n')
            error_number = ERROR_ENUMERATION['maximum_iterations']
            is_converged = False

        if not is_converged:
            if prm['error_on_nonconvergence']:
                raise RuntimeError('Iterations did not converge')
            else:
                print('\nERROR: Iterations did not converge\n')

        return is_converged, error_number


    def enforce_phasefield_bounds(self, p_arr, dp_arr):
        '''Enforce bounds on the phasfield update.

        Enforce condition: 0 <= p_arr + dp_arr <= 1.0

        Parameters
        ----------
        p_arr (numpy.ndarray) : Nodal values of phasefield
        dp_arr (numpy.ndarray) : Phasefield increment values

        Returns
        -------
        bool : True if `dp_arr` was modified else False.

        '''

        mask_lwr = p_arr + dp_arr < PHASEFIELD_LIM_LOWER
        mask_upr = p_arr + dp_arr > PHASEFIELD_LIM_UPPER

        dp_arr[mask_lwr] = -p_arr[mask_lwr]
        dp_arr[mask_upr] = 1.0-p_arr[mask_upr]

        return any(mask_upr) or any(mask_lwr)


    def apply_pde_filter(self, k, degree=1):

        if not isinstance(degree, int) or degree < 1:
            raise ValueError('Require integer degree >= 0')

        V = self._p.function_space()

        if V.ufl_element().degree() != degree:
            V = dolfin.FunctionSpace(V.mesh(), 'CG', degree)

        p = dolfin.Function(V)
        pde_filter(self._p, p, k)
        self._p.interpolate(p)


# FIXME
def solve_phasefield_problem(p, Dp, DJ, kappa, stepsize):
    '''
    PDE filter as a minimization of functional:
        \int (k |Dp| + (p-p_in)**2) dx

    '''

    ALPHA = 0.5 # 0 <= ALPHA <= 1

    dfdp = 30*p**2*(p**2 - 2*p + 1) # TEMP
    # dfdp = 1

    if not kappa > 0.0:
        raise ValueError('Require kappa > 0')

    V = Dp.function_space()
    v = dolfin.TestFunction(V)
    Dp_ = dolfin.TrialFunction(V)

    # diffusive_part =

    lhs = Dp_ * v * dolfin.dx \
        + ALPHA * stepsize * kappa*dolfin.dot(dolfin.grad(Dp_), dolfin.grad(v)) * dolfin.dx

    rhs = - stepsize * DJ * 10 * dfdp * v * dolfin.dx \
          - stepsize * kappa*dolfin.dot(dolfin.grad(p), dolfin.grad(v)) * dolfin.dx

    dolfin.solve(lhs==rhs, Dp, bcs=[],
        solver_parameters={"linear_solver": "lu"},
        form_compiler_parameters={"optimize": True})

    Dp.vector()[:] /= (stepsize/Dp.vector().max())



def pde_filter(p_in, p_out, k):
    '''
    PDE filter as a minimization of functional:
        \int (k |Dp| + (p-p_in)**2) dx

    '''

    if not k > 0.0:
        raise ValueError('Require k > 0')

    V = p_in.function_space()
    v = dolfin.TestFunction(V)
    p = dolfin.TrialFunction(V)

    a = (k*dolfin.dot(dolfin.grad(p), dolfin.grad(v)) + p*v) * dolfin.dx

    L = p_in * v * dolfin.dx

    dolfin.solve(a==L, p_out, bcs=[],
        solver_parameters={"linear_solver": "lu"},
        form_compiler_parameters={"optimize": True})

    # Enforce lower bound since no Dirichlet BC
    p_out.vector()[p_out.vector()<0.0] = 0.0
    p_out.vector()[p_out.vector()>1.0] = 1.0


def update_parameters(lhs, rhs):
    '''Update values of keys in dict-like `lhs` with the values of those keys
    in dict-like `rhs`. `lhs` and `rhs` must both have `keys` attribute.'''

    assert hasattr(lhs, 'keys')
    assert hasattr(rhs, 'keys')

    for k in rhs:

        if k not in lhs:
            # Use traceback to query `lhs`
            raise KeyError(k)

        if hasattr(lhs[k], 'keys'):

            if not hasattr(rhs[k], 'keys'):
                raise TypeError(f'`rhs[{k}]` must be dict-like.')
            else:
                update_parameters(lhs[k], rhs[k])

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
