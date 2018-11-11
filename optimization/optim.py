''' Topology optimization based on phase-fields.

By Danas Sutula

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

        F    = dolfin.derivative(W, u)
        dFdu = dolfin.derivative(F, u)
        dFdp = dolfin.derivative(F, p)

        dJdu = dolfin.derivative(J, u)
        dJdp = dolfin.derivative(J, p)
        dCdp = dolfin.derivative(C, p)

        self._J = J
        self._C = C

        # Total energy dissipation with respect to phasefield
        self._dJdp = dJdp - dolfin.action(dolfin.adjoint(dFdp), z)
        self._dCdp = dCdp # NOTE: assuming `C` is independent of `u`

        if parameters_topology_solver['write_solutions']:
            self.iteration_outfile_u = outfile(title='u')
            self.iteration_outfile_p = outfile(title='p')

        # NOTE: `dFdu` is equivalent to `adjoint(dFdu)` because of symmetry
        adjoint_problem = dolfin.LinearVariationalProblem(dFdu, dJdu, z, bcs_z)
        nonlinear_problem = dolfin.NonlinearVariationalProblem(F, u, bcs_u, dFdu)

        self.adjoint_solver = dolfin.LinearVariationalSolver(adjoint_problem)
        update_parameters(self.adjoint_solver.parameters, parameters_adjoint_solver)
        self.parameters_adjoint_solver = self.adjoint_solver.parameters

        self.nonlinear_solver = dolfin.NonlinearVariationalSolver(nonlinear_problem)
        update_parameters(self.nonlinear_solver.parameters, parameters_nonlinear_solver)
        self.parameters_nonlinear_solver = self.nonlinear_solver.parameters

        self.parameters_topology_solver = parameters_topology_solver.copy()
        self.record_iteration_state = iteration_state_recording_function

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

        coarsening_cossim = math.cos(coarsening_delta_angle * RADIANS_PER_DEGREE)
        refinement_cossim = math.cos(refinement_delta_angle * RADIANS_PER_DEGREE)

        J = self._J # Total energy
        C = self._C # Constraint

        dJdp = self._dJdp
        dCdp = self._dCdp

        J_val_prv = np.inf
        C_val_prv = np.inf

        p_vec = self._p.vector()
        p_arr = p_vec.get_local()

        # dp_arr = np.empty_like(p_arr)
        dp_arr_prv = np.ones_like(p_arr)

        stepsize = stepsize_ini
        is_converged = False
        num_diverged = 0

        for k_itr in range(maximum_iterations):

            _, b = self.nonlinear_solver.solve()

            if not b:
                print('\nERROR: Nonlinear problem could not be solved.\n')
                is_converged = False
                break

            self.adjoint_solver.solve()
            self.record_iteration_state()

            if write_solutions:
                self.iteration_outfile_u << self._u
                self.iteration_outfile_p << self._p

            J_val = assemble(J)
            C_val = assemble(C)

            dJdp_arr = assemble(dJdp).get_local()
            dCdp_arr = assemble(dCdp).get_local()

            dp_arr = dJdp_arr # to be rescaled in-place (dJdp_arr not required)
            dp_aux = dCdp_arr.copy() # to be rescaled (dCdp_arr still required)

            # Scale dp_arr to get target energy dissipation rate
            dp_arr *= (-stepsize) * J_val / dJdp_arr.dot(dJdp_arr)

            while True:

                # Solve residual equation for correctional scale factor
                # C + \grad C \dot [\delta p + \delta p_aux scaler] = 0
                # Apply scale factor on the correctional change dp_aux

                dp_aux *= -(C_val + dCdp_arr.dot(dp_arr)) / dCdp_arr.dot(dp_aux)
                dp_aux *= min(1.0, math.sqrt(dp_arr.dot(dp_arr) / dp_aux.dot(dp_aux)))

                # Apply correction
                dp_arr += dp_aux

                # Masks for lower and upper bounds
                mask_lwr = p_arr + dp_arr < -EPS
                mask_upr = p_arr + dp_arr > 1.0+EPS

                # Enforce lower and upper bounds
                if np.any(mask_lwr) or np.any(mask_upr):

                    dp_arr[mask_lwr] = -p_arr[mask_lwr]
                    dp_arr[mask_upr] = 1.0-p_arr[mask_upr]

                    dp_aux[mask_lwr] = 0.0
                    dp_aux[mask_upr] = 0.0

                else:
                    break

            cossim = dp_arr.dot(dp_arr_prv) / math.sqrt(
                dp_arr.dot(dp_arr) * dp_arr_prv.dot(dp_arr_prv))

            is_constraint_converging = \
                abs(C_val_prv) > abs(C_val) > atol_C

            if J_val_prv > J_val or is_constraint_converging:

                if J_val_prv - J_val < atol_J and abs(C_val) < atol_C:
                    print('INFO: Negligible energy change')
                    is_converged = True
                    break

                J_val_prv = J_val
                C_val_prv = C_val

            else:

                print('\nWARNING: Solution diverged (backtrace)\n')
                if num_diverged == maximum_diverged:
                    print('\nERROR: Reached maximum times diverged\n')
                    is_converged = False
                    break

                p_arr -= dp_arr_prv
                dp_arr = dp_arr_prv

                num_diverged += 1
                cossim = -1.0

            if cossim < refinement_cossim \
                and stepsize is not stepsize_min:
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
                and stepsize is not stepsize_max:
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


            print('INFO: '
                  f'k:{k_itr:3d}, '
                  f'J:{J_val: 9.3e}, '
                  f'C:{C_val: 9.3e}, '
                  f'dp_min:{dp_arr.min(): 8.2e}, '
                  f'dp_max:{dp_arr.max(): 8.2e}, '
                  f'stepsize:{stepsize: 8.2e}, '
                  f'cosine:{cossim: 9.3e}, '
                  )

            if dp_arr.dot(dp_arr) < p_arr.dot(p_arr) * rtol_p**2:
                print('INFO: Negligible phase-field change (break)')
                is_converged = True
                break

            p_arr[:] += dp_arr
            dp_arr_prv = dp_arr
            p_vec.set_local(p_arr)

        else:

            print('\nWARNING: Reached maximum number of iterations\n')
            is_converged = False

        if not is_converged:
            if prm['error_on_nonconvergence']:
                raise RuntimeError('Iterations did not converge')
            else:
                print('\nERROR: Iterations did not converge\n')

        return is_converged

    def apply_pde_filter(self, k, degree=1):

        if not isinstance(degree, int) or degree < 1:
            raise ValueError('Require integer degree >= 0')

        V = self._p.function_space()

        if V.ufl_element().degree() != degree:
            V = dolfin.FunctionSpace(V.mesh(), 'CG', degree)

        p = dolfin.Function(V)
        pde_filter(self._p, p, k)
        self._p.interpolate(p)


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


def update_phasefield(p_in, p_out, k):
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


def outfile(title, ext='pvd'):

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

    outfile = os.path.join(outdir, title)
    outfile += os.extsep + ext

    return dolfin.File(outfile)
