'''

optimization/config.py

'''

import logging 

logger = logging.getLogger()

parameters_topology_solver = {
    'maximum_iterations': 500,
    'maximum_divergences': 3,
    'influence_threshold': 0.05,
    'phasefield_tolerance': 1e-2,
    'constraint_tolerance': 1e-6,
    'error_on_nonconvergence': False,
    }

parameters_nonlinear_solver = {
    'nonlinear_solver': 'snes',
    'symmetric': True,
    'print_matrix': False,
    'print_rhs': False,
    'newton_solver': {
        'absolute_tolerance': 1e-9,
        'convergence_criterion': 'residual',
        'error_on_nonconvergence': True,
        'linear_solver': 'default',
        'maximum_iterations': 25,
        'preconditioner': 'default',
        'relative_tolerance': 1e-12,
        'relaxation_parameter': 1.0,
        'report': False,
        },
    'snes_solver' : {
        'absolute_tolerance': 1e-9,
        'error_on_nonconvergence': True,
        'line_search': 'bt', # 'basic' | 'bt'
        'linear_solver': 'lu',
        'maximum_iterations': 25,
        'maximum_residual_evaluations': 2000,
        'method': 'default',
        'preconditioner': 'default',
        'relative_tolerance': 1e-12,
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
