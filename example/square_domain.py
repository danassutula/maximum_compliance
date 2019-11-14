# -*- coding: utf-8 -*-
'''

'''

import config

import os
import sys
import time
import math
import scipy
import dolfin
import logging
import numpy as np
import scipy.linalg as linalg
import matplotlib.pyplot as plt

from dolfin import *

import optim
import material
import example.utility

logger = logging.getLogger()
logger.setLevel(logging.INFO)

EPS = 1e-12

PROBLEM_NAME = os.path.splitext(os.path.basename(__file__))[0]
RESULTS_OUTDIR_PARENT = os.path.join("results", PROBLEM_NAME)

if not os.path.isdir(RESULTS_OUTDIR_PARENT):
    os.makedirs(RESULTS_OUTDIR_PARENT)

SAFE_TO_REMOVE_FILE_TYPES = \
    ('.log', '.out', '.npy', '.pvd', '.vtu', '.png', '.svg', '.eps', '.pdf')


def phasefield_penalty(p):
    '''For limiting sharp gradients.'''
    return dolfin.grad(p)**2


def material_integrity(p, minimum_value=1e-4, beta=3):
    '''Material integrity from phasefield `p`.'''
    return minimum_value + (1.0-minimum_value) * ((1.0+EPS)-p) ** beta


if __name__ == "__main__":

    plot_results = False
    write_results = True

    # function_writing_period = 100
    function_writing_period = 100

    ### Problem parameters

    domain_p0 = np.array([0.0, 0.0])
    domain_p1 = np.array([1.0, 1.0])

    # domain_p0 = np.array([0.0, 0.0])
    # domain_p1 = np.array([1.0, 0.5])
    # domain_p1 = np.array([1.0, 1/4])
    # domain_p1 = np.array([1.0, 1/8])

    domain_x0, domain_y0 = domain_p0
    domain_x1, domain_y1 = domain_p1

    domain_L = domain_x1 - domain_x0
    domain_H = domain_y1 - domain_y0

    material_model_name = [
        # "LinearElasticModel",
        "NeoHookeanModel",
        ]

    load_type = "biaxial"
    # load_type = "vertical"

    incremental_loading = False
    num_load_increments = 50

    mean_axial_strains = [
        # [4.0, 4.0],
        # [3.0, 3.0],
        [2.0, 2.0],
        # [1.0, 1.0],
        # [0.01, 0.01],
        # [0.000, 2.000],
        # [0.000, 1.000],
        # [0.000, 2.000],
        # [0.000, 3.000],
        # [0.000, 6.000],
        # [0.000, 0.010],
        ]

    delta = 1e-3
    defect_nucleation_centers = [
        #
        np.array([[domain_x0, domain_y0],
                  [domain_x1, domain_y1]]),
        #
        # np.array([[(domain_x0+domain_x1)*0.5, domain_y0],
        #           [domain_x1, (domain_y0+domain_y1)*0.5],
        #           [domain_x0, (domain_y0+domain_y1)*0.5],
        #           [(domain_x0+domain_x1)*0.5, domain_y1]]), # Benchmark
        # np.array([[domain_x0, domain_y0],
        #           [domain_x1, domain_y0],
        #           [domain_x1, domain_y1],
        #           [domain_x0, domain_y1],
        #           [(domain_x0+domain_x1)*0.5, (domain_y0+domain_y1)*0.5]]), # Benchmark
        #
        # np.array([[domain_x0+delta*domain_L, domain_y0],
        #           [domain_x1-delta*domain_L, domain_y1]]),
        # np.array([[domain_x0+1/5*domain_L, domain_y0],
        #           [domain_x1-1/5*domain_L, domain_y1]]),
        # np.array([[domain_x0+1/4*domain_L, domain_y0],
        #           [domain_x1-1/4*domain_L, domain_y1]]),
        # np.array([[domain_x0+1/3*domain_L, domain_y0],
        #           [domain_x1-1/3*domain_L, domain_y1]]),
        #
        # np.array([[domain_x0+delta*domain_L, domain_y0],
        #           [domain_x1-delta*domain_L, domain_y1],
        #           [domain_x1, domain_y0+delta*domain_H],
        #           [domain_x0, domain_y1-delta*domain_H]]),
        #
        # example.utility.perturbed_gridcols(
        # example.utility.perturbed_gridcols(
        #     example.utility.meshgrid_uniform(domain_p0, domain_p1, 6, 6),
        #     6, 6, delta), 6, 6, delta)
        #
        # example.utility.meshgrid_uniform_with_margin(domain_p0, domain_p1, 4, 4),
        # example.utility.meshgrid_checker_symmetric(domain_p0, domain_p1, 7, 7)
        #
        ]

    elliptical_defect_pnorm = 2

    def compute_elliptical_defect_radii(mesh_element_size):
        '''Compute the radii using the mesh element size.

        Note, do not pressume symmetry of the local phasefields. (Check it!)

        '''
        # rx = ry = mesh_element_size * (1+EPS) * 5

        rx = domain_L * (9/10/2)
        ry = mesh_element_size * (1+EPS) * 4

        return rx, ry

    phasefield_penalty_weight = [
        # 0.48,
        0.49,
        # 0.50,
        ]

    phasefield_collision_distance = [
        # 0.050,
        # 0.100,
        0.150,
        # 0.200,
        # 0.250,
        ]

    # Phasefield mean-value stepsize
    phasefield_meanvalue_stepsize = [
        # 0.10,
        0.01,
        ]

    # Phasefield iteration stepsize (L_inf-norm)
    phasefield_iteration_stepsize = [
        0.05
        # 0.02,
        # 0.01,
        ]

    minimum_phasefield_meanvalue = 0.150
    maximum_phasefield_meanvalue = 0.300
    minimum_energy_fraction = 1e-4

    ### Discretization parameters

    num_elements_on_edges = [
        # 61,
        # 80,
        100,
        # 150,
        # 180,
        # 200,
        # 240,
        # 244,
        # 320,
        ] # NOTE: Even/odd numbers of elements may reveal mesh dependence

    # mesh_diagonal = "left/right"
    mesh_diagonal = "crossed"
    # mesh_diagonal = "left"
    # mesh_diagonal = "right"

    displacement_degree = 1
    phasefield_degree = 1

    ### Control parameter grid

    outer_loop_parameters = example.utility.make_parameter_combinations(
        mean_axial_strains,
        material_model_name,
        num_elements_on_edges,
        )

    inner_loop_parameters = example.utility.make_parameter_combinations(
        defect_nucleation_centers,
        phasefield_penalty_weight,
        phasefield_collision_distance,
        phasefield_meanvalue_stepsize,
        phasefield_iteration_stepsize,
        )

    for (
        mean_axial_strains_i,
        material_model_name_i,
        num_elements_on_edges_i,
        ) in outer_loop_parameters:

        if not isinstance(num_elements_on_edges_i, (tuple, list, np.ndarray)):
            nx = num_elements_on_edges_i
            ny = int(round(domain_H/domain_L*nx))
            num_elements_on_edges_i = (nx, ny)
        elif len(num_elements_on_edges_i) != 2:
            raise TypeError

        if not isinstance(mean_axial_strains_i, (tuple, list, np.ndarray)):
            mean_axial_strains_i = (mean_axial_strains_i,)*2
        elif len(mean_axial_strains_i) != 2:
            raise TypeError

        boundary_displacement_values_i = np.stack((
            np.linspace(0.0, mean_axial_strains_i[0] * domain_L, num_load_increments+1),
            np.linspace(0.0, mean_axial_strains_i[1] * domain_H, num_load_increments+1)
            ), axis=1)[1:,:]

        mesh = example.utility.rectangle_mesh(domain_p0, domain_p1,
            num_elements_on_edges_i[0], num_elements_on_edges_i[1],
            mesh_diagonal)

        boundary_bot, boundary_rhs, boundary_top, boundary_lhs = \
            example.utility.boundaries_of_rectangle_mesh(mesh)

        boundary_markers = dolfin.MeshFunction(
            'size_t', mesh, mesh.geometry().dim()-1)

        boundary_markers.set_all(0)

        subdomain_id_bot = 1
        subdomain_id_rhs = 2
        subdomain_id_top = 3
        subdomain_id_lhs = 4

        boundary_bot.mark(boundary_markers, subdomain_id_bot)
        boundary_rhs.mark(boundary_markers, subdomain_id_rhs)
        boundary_top.mark(boundary_markers, subdomain_id_top)
        boundary_lhs.mark(boundary_markers, subdomain_id_lhs)

        ### Integration measures

        dx = dolfin.dx(domain=mesh)
        ds = dolfin.ds(domain=mesh, subdomain_data=boundary_markers)

        ### Function spaces

        fe_u = VectorElement('CG', mesh.cell_name(), displacement_degree)
        fe_p = FiniteElement('CG', mesh.cell_name(), phasefield_degree)

        V_u = FunctionSpace(mesh, fe_u)
        V_p = FunctionSpace(mesh, fe_p)

        u = Function(V_u, name="displacement")
        p = Function(V_p, name="phasefield")

        ### Dirichlet boundary conditions

        bcs, bcs_set_values = \
            example.utility.uniform_extension_bcs(V_u, load_type)

        ### Material model

        if material_model_name_i == "LinearElasticModel":

            material_parameters = {'E': Constant(1.0), 'nu': Constant(0.0)}
            material_model = material.LinearElasticModel(material_parameters, u)

        elif material_model_name_i == "NeoHookeanModel":

            material_parameters = {'E': Constant(1.0), 'nu': Constant(0.0)}
            material_model = material.NeoHookeanModel(material_parameters, u)

        else:
            raise ValueError('Parameter `material_model_name_i`?')

        psi_0 = material_model.strain_energy_density()
        pk1_0 = material_model.stress_measure_pk1()
        pk2_0 = material_model.stress_measure_pk2()

        rho = material_integrity(p)
        phi = phasefield_penalty(p)

        psi = rho * psi_0
        pk1 = rho * pk1_0
        pk2 = rho * pk2_0

        ### Cost functionals

        # Potential energy (strain energy only)
        W = psi * dx

        # Phasefield regularization
        P = phi * dx

        ### Solving for the undamaged material (reference solution)

        # Equilibrium variational form
        F = dolfin.derivative(W, u)

        equilibrium_solve = example.utility.equilibrium_solver(
            F, u, bcs, bcs_set_values, boundary_displacement_values_i)

        # Solve for undamaged material
        equilibrium_solve(incremental_loading)

        W_undamaged = dolfin.assemble(W)
        u_arr_undamaged = u.vector().get_local()

        minimum_energy_for_stopping = minimum_energy_fraction * W_undamaged \
                                      if minimum_energy_fraction is not None \
                                      else None

        for (
            defect_nucleation_centers_i,
            phasefield_penalty_weight_i,
            phasefield_collision_distance_i,
            phasefield_meanvalue_stepsize_i,
            phasefield_iteration_stepsize_i,
            ) in inner_loop_parameters:

            problem_start_time = time.perf_counter()

            problem_title = (
                f"date({time.strftime('%m%d_%H%M')})-"
                f"model({material_model_name_i})-"
                f"load({load_type})-"
                f"mesh({num_elements_on_edges_i[0]:d}x"
                     f"{num_elements_on_edges_i[1]:d})-"
                f"dims({domain_L:.3g}x{domain_H:.3g})-"
                f"flaws({len(defect_nucleation_centers_i)})-"
                f"exx({mean_axial_strains_i[0]:.3g})-"
                f"eyy({mean_axial_strains_i[1]:.3g})-"
                f"reg({phasefield_penalty_weight_i:.3g})-"
                f"dist({phasefield_collision_distance_i:.3g})-"
                f"inc({phasefield_meanvalue_stepsize_i:.3g})-"
                f"step({phasefield_iteration_stepsize_i:.3g})"
                )

            logger.info("Begin solving problem:\n\t"
                + problem_title.replace('-','\n\t'))

            if write_results:

                results_outdir = os.path.join(RESULTS_OUTDIR_PARENT, problem_title)
                results_outdir_arrays = os.path.join(results_outdir, "arrays")
                results_outdir_figures = os.path.join(results_outdir, "figures")
                results_outdir_functions = os.path.join(results_outdir, "functions")

                if not os.path.isdir(results_outdir_arrays): os.makedirs(results_outdir_arrays)
                if not os.path.isdir(results_outdir_figures): os.makedirs(results_outdir_figures)
                if not os.path.isdir(results_outdir_functions): os.makedirs(results_outdir_functions)

                example.utility.remove_outfiles(results_outdir, SAFE_TO_REMOVE_FILE_TYPES)
                example.utility.remove_outfiles(results_outdir_arrays, SAFE_TO_REMOVE_FILE_TYPES)
                example.utility.remove_outfiles(results_outdir_figures, SAFE_TO_REMOVE_FILE_TYPES)
                example.utility.remove_outfiles(results_outdir_functions, SAFE_TO_REMOVE_FILE_TYPES)

                solution_writer_p = example.utility.FunctionWriter(
                    results_outdir_functions, p, "p", function_writing_period)

                write_solution_p = solution_writer_p.write
                write_solution_p_periodic = solution_writer_p.periodic_write

            else:
                write_solution_p = None
                write_solution_p_periodic = None

            u.vector()[:] = u_arr_undamaged

            rx, ry = compute_elliptical_defect_radii(mesh.hmax())

            p_locals = example.utility.make_defect_like_phasefield_array(
                V_p, defect_nucleation_centers_i, rx, ry, elliptical_defect_pnorm)

            optim.filter.apply_diffusive_smoothing(p_locals, kappa=1e-4)

            solver_iterations_failed, energy_vs_iterations, energy_vs_phasefield, \
            phasefield_meanvalues, phasefield_iterations, topology_optimizer, \
            p_mean_target = example.utility.solve_compliance_maximization_problem(
                    W, P, p, p_locals, equilibrium_solve,
                    phasefield_penalty_weight_i,
                    phasefield_collision_distance_i,
                    phasefield_iteration_stepsize_i,
                    phasefield_meanvalue_stepsize_i,
                    minimum_phasefield_meanvalue,
                    maximum_phasefield_meanvalue,
                    minimum_energy_for_stopping,
                    write_solution_p_periodic,
                    )

            if write_solution_p:
                write_solution_p()

            # energy_vs_iterations = energy_vs_iterations[
            #     ::max(1, int(len(energy_vs_iterations)/1000))]

            normalized_energy_vs_iterations = \
                [W_j / W_undamaged for W_j in energy_vs_iterations]

            normalized_energy_vs_phasefield = \
                [W_j / W_undamaged for W_j in energy_vs_phasefield]

            if write_results:

                # Create a new file with solver status in the title
                open(os.path.join(results_outdir,
                    f'finished_normally({solver_iterations_failed==False}).txt'),
                    mode='w').close()

                np.savetxt(os.path.join(results_outdir_arrays,
                    "normalized_energy_vs_iterations.txt"),
                    normalized_energy_vs_iterations)

                np.savetxt(os.path.join(results_outdir_arrays,
                    "normalized_energy_vs_phasefield.txt"),
                    normalized_energy_vs_phasefield)

                np.savetxt(os.path.join(results_outdir_arrays,
                    "phasefield_meanvalues.txt"), phasefield_meanvalues)

                np.savetxt(os.path.join(results_outdir_arrays,
                    "phasefield_iterations.txt"), phasefield_iterations, fmt='%d')

            if plot_results or write_results:

                figure_handles = []

                figure_handles.append(
                    example.utility.plot_energy_vs_iterations(
                        normalized_energy_vs_iterations,
                        figname="potential_energy_vs_iterations",
                        ylabel="Normalized strain energy", fontsize="xx-large"))

                figure_handles.append(
                    example.utility.plot_energy_vs_phasefields(
                        normalized_energy_vs_phasefield, phasefield_meanvalues,
                        figname="potential_energy_vs_phasefield",
                        ylabel="Normalized strain energy", fontsize="xx-large"))

                figure_handles.append(
                    example.utility.plot_phasefiled_vs_iterations(
                        phasefield_meanvalues, phasefield_iterations,
                        figname="phasefield_vs_iterations", fontsize="xx-large"))

                if write_results:

                    fig_handles = [f[0] for f in figure_handles]
                    fig_names = [f[1] for f in figure_handles]

                    for handle_i, name_i in zip(fig_handles, fig_names):
                        name_i = os.path.join(results_outdir_figures, name_i)

                        handle_i.savefig(name_i+'.png')
                        handle_i.savefig(name_i+'.pdf')

                if not plot_results:
                    plt.close('all')

            logger.info("Finished solving problem\n\t"
                + problem_title.replace('-','\n\t'))

            problem_finish_time = time.perf_counter()
            problem_elapsed_time = problem_finish_time - problem_start_time

            problem_elapsed_time_readable = {
                "Days":    int(problem_elapsed_time // (24*3600)),
                "Hours":   int((problem_elapsed_time % (24*3600)) // 3600),
                "Minutes": int((problem_elapsed_time % 3600) // 60),
                "Seconds": int(problem_elapsed_time % 60)}

            if logger.getEffectiveLevel() <= logging.INFO:
                logger.info("Elapsed times:")

                tmp = ((v, k) for k, v in problem_elapsed_time_readable.items())
                print('\t'+', '.join(("{} {}",)*4).format(*sum(tmp, ())) + '\n')
