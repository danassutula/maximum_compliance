'''

optim/config.py

Notes
-----

Some explanations of parameters.

parameters_distance_solver['alpha']:
    Threshold-like parameter. Phasefield value greater than `phasefield_alpha`
    marks the subdomain whose boundary is considered to be the "phasefield
    boundary". This boundary (and the interior domain) is marked as zero-
    distance to the phasefield.

parameters_distance_solver['kappa']:
    Diffusion-like parameter (>0.0) used to regularize the distance equation
    so that the phasefield distance problem can be solved uniquely.

parameters_distance_solver['gamma']:
    Penalty-like parameter that serves to weaky impose the Dirichlet BC's
    in the phasefield distance problem.

'''

import logging

logger = logging.getLogger()


parameters_topology_solver = {
    'convergence_tolerance': 1e-3,
    'minimum_convergences': 3,
    'maximum_divergences': 2,
    'maximum_iterations': 2000,
    }

parameters_distance_solver = {
    'alpha': 1/3, # Lower bound value defining the phasefield boundary
    'kappa': 2e-3, # Stabilization for the boundary distance solution
    'gamma': 1e5,  # Penalty that weakly enforces zero-distance BC's
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
