# -*- coding: utf-8 -*-
"""
Created on 01/10/2018

@author: Danas Sutula

Todos
-----
Need to come up with a criterion for stopping the incrementation of phasefield
volume fraction. The phasefield is always successful in finding a more optimal
phasefield for a given phasefield volume fraction; however, after some phasefield
volume fraction value, it does not make sense to continue trying to optimize.


"""

import config

import os
import time
import math
import scipy
import dolfin
import logging
import numpy as np
import scipy.linalg as linalg
import matplotlib.pyplot as plt

# from scipy.io import loadmat
# from scipy.io import savemat

from dolfin import *

import optim
import material
import utility

logger = logging.getLogger()
logger.setLevel(logging.INFO)

EPS = 1e-12


### Results output

PROBLEM_NAME = os.path.splitext(os.path.basename(__file__))[0]
RESULTS_OUTDIR_PARENT = os.path.join("results", PROBLEM_NAME)

if not os.path.isdir(RESULTS_OUTDIR_PARENT):
    os.makedirs(RESULTS_OUTDIR_PARENT)

SAFE_TO_REMOVE_FILE_TYPES = \
    ('.out', '.npy', '.pvd', '.vtu', '.png', '.svg', '.eps', '.pdf')


### Problem definition

def phasefield_regularization(p):
    '''Penalty for large phasefield gradients.'''
    return dolfin.grad(p)**2


def material_integrity_model(p):
    '''Material integrity given the value of phasefield.

    Notes
    -----
    The returned value should be in range [rho_min, 1].

    '''

    # Minimum (residual) material integrity
    rho_min = Constant(1e-5)

    # Material degradation exponent (`>1`)
    beta = 2

    return rho_min + (1.0-rho_min) * ((1.0+EPS)-p) ** beta


if __name__ == "__main__":

    ### Problem parameters

    load_type = 'biaxial'
    # load_type = 'uniaxial'

    boundary_displacement_value = 0.01

    material_model_name = "LinearElasticModel"
    # material_model_name = "NeoHookeanModel"

    defect_nucleation_pattern = "uniform_wout_margin"
    # defect_nucleation_pattern = "uniform_with_margin"

    numbers_of_defects_per_dimension = [6,]

    phasefield_volume_fraction_stepsize = 0.1
    phasefield_volume_fraction_maximum = 1.0

    save_results = True
    plot_results = True

    # Write solutions every number of solver iterations
    function_writing_period = 25


    ### Discretization parameters

    # Displacement function degree
    displacement_degree = 1

    # Phasefield function degree
    phasefield_degree = 1

    # mesh_pattern = "left/right"
    mesh_pattern = "crossed"
    # mesh_pattern = "left"

    # number_of_cells_along_edge = 41
    # number_of_cells_along_edge = 60
    # number_of_cells_along_edge = 80
    # number_of_cells_along_edge = 100
    number_of_cells_along_edge = 160
    # number_of_cells_along_edge = 320


    ### Solver parameters

    # Maximum phasefield increment size (L_inf-norm)
    phasefield_stepsize = 0.01

    # Phasefield convergence tolerance (L_1-norm)
    convergence_tolerance = 1e-3

    # Minimum distance between local phasefields
    collision_distance = 4 / number_of_cells_along_edge
    # collision_distance_hard = collision_distance  * 0
    # NOTE: Generally, should not be mesh dependent,
    # but we would like as close proximity as possible.

    defect_nucleation_radius = 3 / number_of_cells_along_edge
    # NOTE: Generally, should not be mesh dependent.

    # Phasefield penalty (regularization) weight
    # penalty_weight = 0.350
    # penalty_weight = 0.400
    penalty_weight = 0.425
    # penalty_weight = 0.450
    # penalty_weight = 0.475


    ### Discretization

    if number_of_cells_along_edge % 2:
        if not all(n_i % 2 for n_i in numbers_of_defects_per_dimension):
            logger.warning('For an ODD number of elements, the number of '
                           'defects should be ODD. This improves the '
                           'stability of the phasefield evolution.')
    else:
        if any(n_i % 2 for n_i in numbers_of_defects_per_dimension):
            logger.warning('For an EVEN number of elements, the number of '
                           'defects should be EVEN. This improves the '
                           'stability of the phasefield evolution.')

    mesh = utility.unit_square_mesh(number_of_cells_along_edge, mesh_pattern)


    ### Integration measure

    dx = dolfin.dx(domain=mesh)
    ds = dolfin.ds(domain=mesh)


    ### Function spaces

    V_u = VectorFunctionSpace(mesh, 'CG', displacement_degree)
    V_p = FunctionSpace(mesh, 'CG', phasefield_degree)

    u = Function(V_u, name="displacement")
    p = Function(V_p, name="phasefield")


    ### Dirichlet boundary conditions

    if load_type == "biaxial":
        bcs, bcs_set_value = utility.uniform_biaxial_extension_bcs(V_u)
    elif load_type == "uniaxial":
        bcs, bcs_set_value = utility.uniform_uniaxial_extension_bcs(V_u)
    else:
        raise ValueError('`load_type`?')

    bcs_set_value(boundary_displacement_value)


    ### Material model

    if material_model_name == "LinearElasticModel":

        material_parameters = {'E': Constant(1.0), 'nu': Constant(0.4)}
        material_model = material.LinearElasticModel(material_parameters, u)

    elif material_model_name == "NeoHookeanModel":

        material_parameters = {'E': Constant(1.0), 'nu': Constant(0.4)}
        material_model = material.NeoHookeanModel(material_parameters, u)

    else:
        raise ValueError('Parameter `material_model_name`?')

    psi = material_model.strain_energy_density()
    pk1 = material_model.stress_measure_pk1()
    pk2 = material_model.stress_measure_pk2()

    rho = material_integrity_model(p)

    # Potential energy (only strain energy)
    W = rho * psi * dx

    # Variational form
    F = dolfin.derivative(W, u)


    ### Phasefield constraints

    phi = phasefield_regularization(p)

    # Phasefield penalty (for regularizatio)
    P = phi * dx

    # Target mean phasefield
    p_mean = Constant(0.0)

    # Phasefield fraction constraint
    C = (p - p_mean) * dx


    ### Generate phasefield nucleation coordinates

    list_of_defect_coordinates = []

    if defect_nucleation_pattern == "uniform_wout_margin":
        meshgrid = optim.helper.meshgrid_uniform
    elif defect_nucleation_pattern == "uniform_with_margin":
        meshgrid = optim.helper.meshgrid_uniform_with_margin
    elif defect_nucleation_pattern == "checker":
        meshgrid = optim.helper.meshgrid_checker
    else:
        raise ValueError('`defect_nucleation_pattern`?')

    x0, y0 = mesh.coordinates().min(axis=0)
    x1, y1 = mesh.coordinates().max(axis=0)

    for n_i in numbers_of_defects_per_dimension:

        list_of_defect_coordinates.append(
            meshgrid((x0, x1), (y0, y1), n_i, n_i)
            # meshgrid([-0.5,0.5], [0.0,0.0], n_i, 1)
            )


    ### Main loop for parametric study

    # NOTE: The displacement solution will be used for reinitialization.
    # NOTE: The undamged potential energy will serve as a reference.

    assert p.vector().get_local().any() == False

    # Solve hyperelastic problem without any damage
    dolfin.solve(F==0, u, bcs)

    undamaged_solution_vector = u.vector().get_local()
    undamaged_potential_energy = dolfin.assemble(W)

    for defect_coordinates_i in list_of_defect_coordinates:

        problem_start_time = time.time()

        problem_title = (
            f"load({load_type})-"
            f"periodicBC({False})-"
            f"displacement({boundary_displacement_value})-"
            f"defectPattern({defect_nucleation_pattern})-"
            f"defectCount({len(defect_coordinates_i)})-"
            f"penaltyWeight({penalty_weight})-"
            f"material({material_model_name})-"
            f"mesh({mesh.num_vertices()})-"
            f"date({time.strftime('%m%d')})-"
            f"hour({time.strftime('%H')})"
            )

        logger.info("BEGIN PROBLEM:\n\n\t"
            + problem_title.replace('-','\n\t')+"\n")


        ### Results output

        if save_results:

            results_outdir_arrays = os.path.join(
                RESULTS_OUTDIR_PARENT, problem_title, "arrays")

            results_outdir_figures = os.path.join(
                RESULTS_OUTDIR_PARENT, problem_title, "figures")

            results_outdir_functions = os.path.join(
                RESULTS_OUTDIR_PARENT, problem_title, "functions")

            if not os.path.isdir(results_outdir_arrays):
                os.makedirs(results_outdir_arrays)

            if not os.path.isdir(results_outdir_figures):
                os.makedirs(results_outdir_figures)

            if not os.path.isdir(results_outdir_functions):
                os.makedirs(results_outdir_functions)

            utility.remove_outfiles(results_outdir_arrays, SAFE_TO_REMOVE_FILE_TYPES)
            utility.remove_outfiles(results_outdir_figures, SAFE_TO_REMOVE_FILE_TYPES)
            utility.remove_outfiles(results_outdir_functions, SAFE_TO_REMOVE_FILE_TYPES)

            solution_writer = utility.SolutionWriter(
                results_outdir_functions, u, p, writing_period=function_writing_period)

        else:
            solution_writer = type("DummySolutionWriter", (),
                dict(write=lambda:None, periodic_write=lambda:None))


        ### Initialize local phasefields

        defect_nucleation_radius += mesh.hmax() * 1e-4

        p_locals = optim.helper.make_defect_like_phasefield_array(
            V_p, defect_coordinates_i, defect_nucleation_radius, kappa=1e-4)

        phasefield_volume_fraction_initial = \
            assemble(sum(p_locals)*dx) / assemble(1*dx)


        ### Optimization problem

        optimizer = optim.TopologyOptimizer(F, W, P, C, p, p_locals, u,
            bcs, external_callable=solution_writer.periodic_write)

        # Reinitialize the undamged displacement solution
        u.vector()[:] = undamaged_solution_vector


        ### Solve case

        potentials_vs_iterations = []
        potentials_vs_phasefield = []
        phasefield_volume_fractions = []

        solver_iteration_failed = False
        minimum_iters_for_early_stop = 3

        phasefield_volume_fraction_i = phasefield_volume_fraction_initial
        while phasefield_volume_fraction_i < phasefield_volume_fraction_maximum:

            phasefield_volume_fraction_i += \
                phasefield_volume_fraction_stepsize

            phasefield_volume_fractions.append(
                phasefield_volume_fraction_i)

            p_mean.assign(phasefield_volume_fraction_i)

            try:

                logger.info('Solving for phasefield volume fraction '
                            f'{phasefield_volume_fraction_i:.3f}')

                iterations_i, converged_i, potentials_vs_iterations_i = \
                    optimizer.optimize(phasefield_stepsize, penalty_weight,
                        collision_distance, convergence_tolerance)

            except RuntimeError:

                logger.error('Solver failed for volume fraction '
                             f'{phasefield_volume_fraction_i:.3f}')

                phasefield_volume_fractions.pop()
                solver_iteration_failed = True
                break

            potentials_vs_iterations.extend(potentials_vs_iterations_i)
            potentials_vs_phasefield.append(potentials_vs_iterations_i[-1])

            if potentials_vs_iterations_i[0] < potentials_vs_iterations_i[-1]:
                break

            if iterations_i <= minimum_iters_for_early_stop:
                logger.info('Early stop triggered [STOP]')
                break


        ### Save results

        potentials_vs_iterations = [W_i / undamaged_potential_energy
                                    for W_i in potentials_vs_iterations]

        potentials_vs_phasefield = [W_i / undamaged_potential_energy
                                    for W_i in potentials_vs_phasefield]

        if save_results:

            np.savetxt(os.path.join(results_outdir_arrays,
                "potentials_vs_iterations.out"), potentials_vs_iterations)

            np.savetxt(os.path.join(results_outdir_arrays,
                "potentials_vs_phasefield.out"), potentials_vs_phasefield)

            np.savetxt(os.path.join(results_outdir_arrays,
                "phasefield_volume_fractions.out"), phasefield_volume_fractions)

            solution_writer.write(forcewrite_all=True)

        if save_results or plot_results:

            figure_handles = []

            figure_handles.append(
                utility.plot_energy_vs_iterations(
                    potentials_vs_iterations))

            figure_handles.append(
                utility.plot_energy_vs_phasefields(
                    potentials_vs_phasefield, phasefield_volume_fractions))

            figure_handles.append(
                utility.plot_phasefiled(p))

            if save_results:

                fig_handles = [f[0] for f in figure_handles]
                fig_names = [f[1] for f in figure_handles]

                for handle_i, name_i in zip(fig_handles, fig_names):
                    name_i = os.path.join(results_outdir_figures, name_i)

                    handle_i.savefig(name_i+'.png')
                    handle_i.savefig(name_i+'.svg')
                    handle_i.savefig(name_i+'.pdf')

            if not plot_results:
                plt.close('all')


        ### Finish

        logger.info("END PROBLEM:\n\n\t"
            + problem_title.replace('-','\n\t')+"\n")

        problem_finish_time = time.time()
        problem_elapsed_time = problem_finish_time - problem_start_time # (sec)

        # problem_elapsed_time_readable = {
        #     "Days": (problem_elapsed_time / 3600) // 24,
        #     "Hours": (problem_elapsed_time / 3600) % (24 * 3600)) * 24 // 3600,
        #     "Minuts": (problem_elapsed_time % (24 * 3600)) * 24 // 3600,}

        logger.info(f'Elapsed time (s): {problem_elapsed_time:g}')