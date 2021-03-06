
import logging

logger = logging.getLogger()


parameters_topology_solver = {
    'convergence_tolerance': 1e-6,
    'maximum_iterations': 15000,
    'minimum_convergences': 15,
    'distances_update_frequency': 0.25, # from 0.0 to 1.0
    }

parameters_distance_solver = {
    'method': 'variational',
    # 'method': 'algebraic',
    'variational_solver': {
        'threshold': 1/3, # Threshold marking the zero-distance subdomain
        'viscosity': 5e-3, # Diffusive stabilization for solution uniqueness
        'penalty': 1e5, # Penalty for weakly enforcing the zero-distance BC's
        },
    'algebraic_solver': {
        'threshold': 1/4, # Threshold marking the zero-distance subdomain
        },
    }

parameters_nonlinear_solver = {
    'nonlinear_solver': 'newton', # Faster than "snes" but can not impose displacement bounds
    # 'nonlinear_solver': 'snes', # Can impose displacement bounds if using method "vinewtonrsls"
    'symmetric': True,
    'print_matrix': False,
    'print_rhs': False,
    'newton_solver': {
        'absolute_tolerance': 1e-9,
        'convergence_criterion': 'residual',
        'error_on_nonconvergence': True,
        'linear_solver': 'default',
        'maximum_iterations': 10, # 25
        'preconditioner': 'default',
        'relative_tolerance': 1e-12,
        'relaxation_parameter': 1.0,
        'report': False,
        },
    'snes_solver' : {
        'absolute_tolerance': 1e-12,
        'error_on_nonconvergence': True,
        'line_search': 'bt', # 'basic' | 'bt'
        'linear_solver': 'lu',
        'maximum_iterations': 20, # 50
        'maximum_residual_evaluations': 2000,
        # 'method': 'default',
        'method': 'vinewtonrsls',
        # 'preconditioner': 'default',
        'preconditioner': 'ilu',
        # 'preconditioner': 'hypre_amg',
        'relative_tolerance': 1e-9,
        'report': False,
        'sign': 'default',
        'solution_tolerance': 1e-9,
        },
    }

parameters_linear_solver = {
    'linear_solver': 'default',
    'preconditioner': 'default',
    'symmetric': True,
    'print_matrix': False,
    'print_rhs': False,
    }
