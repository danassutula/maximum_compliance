''' Configuration file.

By Danas Sutula

'''

parameters_topology_solver = {
    'absolute_tolerance_energy': 1e-5,
    'absolute_tolerance_constraint': 1e-14,
    'relative_tolerance_phasefield': 1e-5,
    'maximum_iterations': 500,
    'maximum_diverged': 3,
    'maximum_stepsize': 0.10,
    'minimum_stepsize': 0.00,
    'initial_stepsize': 0.10,
    'error_on_nonconvergence': False,
    'stepsize_increase_factor': 2.0,
    'stepsize_decrease_factor': 0.5,
    'coarsening_delta_angle': 15.0,
    'refinement_delta_angle': 75.0,
    'write_solutions': True,
    }

parameters_nonlinear_solver = {
    'nonlinear_solver': 'snes',
    'symmetric': True,
    'print_matrix': False,
    'print_rhs': False,
    'newton_solver': {
        'absolute_tolerance': 1e-7,
        'convergence_criterion': 'residual',
        'error_on_nonconvergence': False,
        'linear_solver': 'default',
        'maximum_iterations': 25,
        'preconditioner': 'default',
        'relative_tolerance': 1e-12,
        'relaxation_parameter': 1.0,
        'report': False,
        },
    'snes_solver' : {
        'absolute_tolerance': 1e-7,
        'error_on_nonconvergence': False,
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

parameters_adjoint_solver = {
    'linear_solver': 'default',
    'preconditioner': 'default',
    'symmetric': True,
    'print_matrix': False,
    'print_rhs': False,
    }
