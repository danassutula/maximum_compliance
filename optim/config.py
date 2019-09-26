
import logging

logger = logging.getLogger()


parameters_topology_solver = {
    'convergence_tolerance': 1e-4,
    'minimum_convergences': 5,
    'maximum_iterations': 10000,
    }

parameters_distance_solver = {
    'threshold': 1/3, # Lower bound value defining the phasefield boundary
    'viscosity': 1e-2, # Stabilization for the boundary distance solution
    'penalty': 1e5, # Penalty that weakly enforces zero-distance BC's
    }

parameters_nonlinear_solver = {
    'nonlinear_solver': 'newton',
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
        'absolute_tolerance': 1e-12,
        'error_on_nonconvergence': True,
        'line_search': 'bt', # 'basic' | 'bt'
        'linear_solver': 'lu',
        'maximum_iterations': 50,
        'maximum_residual_evaluations': 2000,
        'method': 'default',
        'preconditioner': 'default',
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
