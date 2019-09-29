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
    ('.out', '.npy', '.pvd', '.vtu', '.png', '.svg', '.eps', '.pdf')

optim.config.parameters_nonlinear_solver['nonlinear_solver'] = 'newton'
# optim.config.parameters_nonlinear_solver['nonlinear_solver'] = 'snes'


def phasefield_penalty(p):
    return dolfin.grad(p)**2

def material_integrity(p, rho_min=1e-5):
    '''Material integrity given the value of phasefield.

    Notes
    -----
    The returned value should be in range [rho_min, 1].

    '''

    # Material degradation exponent (`>=2`)
    # beta = 2 # Iteration progress is slow, reaches "fully-dagmed" level easily
    beta = 3
    # beta = 4 # Faster iteration progress but can not quite reach "fully-damage" level easily

    return rho_min + (1.0-rho_min) * ((1.0+EPS)-p) ** beta

    # return 1.0 - (3.0*p**2 - 2.0*p**3)


if __name__ == "__main__":

    plot_results = False
    write_results = True

    # Write solutions every number of solver iterations
    # (All last solutions will be written automatically)

    results_writing_period = 200
    write_phasefield_pvd = True
    write_phasefield_npy = True
    write_displacements_pvd = False
    write_displacements_npy = False

    ### Problem parameters

    domain_p0 = np.array([0.0,0.0])
    domain_p1 = np.array([1.0,1.0])

    domain_x0, domain_y0 = domain_p0
    domain_x1, domain_y1 = domain_p1

    domain_L = domain_x1 - domain_x0
    domain_H = domain_y1 - domain_y0

    material_model_name = [
        "LinearElasticModel",
        # "NeoHookeanModel",
        ]

    load_type = "biaxial"
    mean_axial_strains = [
        # [0.000, 0.100],
        [0.100, 0.100],
        ]

    delta = 1e-8
    defect_nucleation_centers = [
        # np.array([[domain_x0, domain_y0]]), # Benchmark
        # np.array([[(domain_x0+domain_x1)/2, (domain_y0+domain_y1)/2]]), # Benchmark
        # np.array([[domain_x0, domain_y0],
        #           [domain_x1, domain_y1]]),
        np.array([[domain_x0+delta, domain_y0],
                  [domain_x1, domain_y0+delta],
                  [domain_x1-delta, domain_y1],
                  [domain_x0, domain_y1-delta]]),
        # np.array([[domain_x0+0.25*domain_L, domain_y0],
        #           [domain_x0+0.75*domain_L, domain_y1]]), # Interesting
        # np.array([[domain_x0+0.25*domain_L, domain_y0],
        #           [domain_x1-0.25*domain_L, domain_y1],
        #           [domain_x1, domain_y0+0.25*domain_H],
        #           [domain_x0, domain_y1-0.25*domain_H]]), # Interesting
        # np.array([[domain_x0+0.25*domain_L, domain_y0],
        #           [domain_x1-0.25*domain_L, domain_y1],
        #           [domain_x1, domain_y0+0.25*domain_H],
        #           [domain_x0, domain_y1-0.25*domain_H],
        #           [0.5*(domain_x0+domain_x1), 0.5*(domain_y0+domain_y1)]]), # Interesting
        # np.array([[domain_x0, domain_y0],
        #           [domain_x1, 0.5*(domain_y0+domain_y1)],
        #           [domain_x0, domain_y1]]),
        ]

    constrained_subdomain_functions = False
    if constrained_subdomain_functions:
        def constrained_subdomain_functions():
            '''Returns a predicate function that evaluates to `True` if a point is
            inside the subdomain where the phasefield fraction will be constrained.
            '''

            # eps = max(domain_L, domain_H) * EPS

            inside_functions = [
                lambda x: True,
                ]

            return inside_functions

    def compute_defect_nucleation_diameter(mesh_element_size):
        "Compute the diameter using the mesh element size."
        return mesh_element_size * (1+EPS) * 10

    phasefield_penalty_weight = [
        # 0.400,
        # 0.450,
        0.470,
        # 0.480,
        # 0.490,
        # 0.500,
        ]

    phasefield_collision_distance = [
        0.100,
        # 0.050,
        ]

    # Phasefield domain fraction increment
    phasefield_fraction_increment = [
        # 0.200,
        # 0.100,
        0.050,
        # 0.025,
        # 0.010,
        ]

    # Phasefield iteration stepsize (L_inf-norm)
    phasefield_iteration_stepsize = [
        # 0.050,
        # 0.025,
        0.010,
        ]

    maximum_phasefield_fraction = 1/3
    minimum_energy_fraction = 1e-3

    ### Discretization parameters

    num_elements_on_edges = [
        # 40,
        # 41,
        # 80,
        81,
        # 121,
        # 160,
        # 161,
        # 320,
        # 321,
        ] # NOTE: Even/odd numbers of elements may reveal mesh dependence

    # mesh_diagonal = "left/right"
    mesh_diagonal = "crossed"
    # mesh_diagonal = "left"

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
        phasefield_fraction_increment,
        phasefield_iteration_stepsize,
        )

    for (
        mean_axial_strains_i,
        material_model_name_i,
        num_elements_on_edges_i,
        ) in outer_loop_parameters:

        if not isinstance(num_elements_on_edges_i, (tuple, list, np.ndarray)):
            num_elements_on_edges_i = (num_elements_on_edges_i,
                max(int(round(num_elements_on_edges_i*domain_H/domain_L)), 1))
        elif len(num_elements_on_edges_i) != 2:
            raise TypeError

        if not isinstance(mean_axial_strains_i, (tuple, list, np.ndarray)):
            mean_axial_strains_i = (mean_axial_strains_i,)*2
        elif len(mean_axial_strains_i) != 2:
            raise TypeError

        boundary_displacements_i = [
            mean_axial_strains_i[0] * domain_L/2,
            mean_axial_strains_i[1] * domain_H/2,
            ]

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

        if displacement_degree > 1:
            fe_T = TensorElement('CG', mesh.cell_name(), displacement_degree-1)
        else:
            fe_T = TensorElement('DG', mesh.cell_name(), 0)

        V_u = FunctionSpace(mesh, fe_u)
        V_p = FunctionSpace(mesh, fe_p)
        V_T = FunctionSpace(mesh, fe_T)

        u = Function(V_u, name="displacement")
        p = Function(V_p, name="phasefield")

        ### Dirichlet boundary conditions

        bcs, bcs_set_values = example.utility.uniform_extension_bcs(V_u, load_type)

        bcs_set_values(ux=boundary_displacements_i[0],
                       uy=boundary_displacements_i[1])

        ### Material model

        if material_model_name_i == "LinearElasticModel":

            material_parameters = {'E': Constant(1.0), 'nu': Constant(0.4)}
            material_model = material.LinearElasticModel(material_parameters, u)

        elif material_model_name_i == "NeoHookeanModel":

            material_parameters = {'E': Constant(1.0), 'nu': Constant(0.4)}
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

        # Variational form of equilibrium
        F = derivative(W, u)

        ### Solving for the undamaged material (reference solution)

        dolfin.solve(F==0, u, bcs, solver_parameters={"nonlinear_solver": "snes"})
        W_undamaged, u_arr_undamaged = dolfin.assemble(W), u.vector().get_local()
        minimum_energy_threshold = minimum_energy_fraction * W_undamaged

        for (
            defect_nucleation_centers_i,
            phasefield_penalty_weight_i,
            phasefield_collision_distance_i,
            phasefield_fraction_increment_i,
            phasefield_iteration_stepsize_i,
            ) in inner_loop_parameters:

            problem_start_time = time.perf_counter()

            defect_nucleation_diameter_i = \
                compute_defect_nucleation_diameter(mesh.hmax())

            problem_title = (
                f"date({time.strftime('%m%d_%H%M')})-"
                f"model({material_model_name_i})-"
                f"mesh({num_elements_on_edges_i[0]:d}x"
                     f"{num_elements_on_edges_i[1]:d})-"
                f"dims({domain_L:.3g}x{domain_H:.3g})-"
                f"flaws({len(defect_nucleation_centers_i)})-"
                f"exx({mean_axial_strains_i[0]:.3g})-"
                f"eyy({mean_axial_strains_i[1]:.3g})-"
                f"reg({phasefield_penalty_weight_i:.3g})-"
                f"inc({phasefield_fraction_increment_i:.3g})-"
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

                if write_displacements_pvd or write_displacements_npy:

                    solution_writer_u = example.utility.FunctionWriter(
                        results_outdir_functions, u, "u", results_writing_period,
                        write_displacements_pvd, write_displacements_npy)

                    write_solution_u = solution_writer_u.write
                    write_solution_u_periodic = solution_writer_u.periodic_write

                else:
                    write_solution_u = lambda : None
                    write_solution_u_periodic = lambda : None

                if write_phasefield_pvd or write_phasefield_npy:

                    solution_writer_p = example.utility.FunctionWriter(
                        results_outdir_functions, p, "p", results_writing_period,
                        write_phasefield_pvd, write_phasefield_npy)

                    write_solution_p = solution_writer_p.write
                    write_solution_p_periodic = solution_writer_p.periodic_write

                else:
                    write_solution_p = lambda : None
                    write_solution_p_periodic = lambda : None

                def write_solutions():
                    write_solution_u()
                    write_solution_p()

                def write_solutions_periodic():
                    write_solution_u_periodic()
                    write_solution_p_periodic()

            else:
                write_solutions = lambda : None
                write_solutions_periodic = lambda : None

            u.vector()[:] = u_arr_undamaged

            _solution_writer_function_for_each_phasefield_fraction = None # write_solutions
            _solution_writer_function_for_each_phasefield_iteration = write_solutions_periodic

            if constrained_subdomain_functions:
                constrained_subdomain_functions_i = constrained_subdomain_functions()
            else:
                constrained_subdomain_functions_i = None

            solver_iterations_failed, energy_vs_iterations, energy_vs_phasefield, \
            phasefield_fractions, topology_optimizer, p_locals, p_mean_target = \
                example.utility.solve_compliance_maximization_problem(
                    W, P, u, p, bcs,
                    defect_nucleation_centers_i,
                    defect_nucleation_diameter_i,
                    phasefield_penalty_weight_i,
                    phasefield_collision_distance_i,
                    phasefield_iteration_stepsize_i,
                    phasefield_fraction_increment_i,
                    maximum_phasefield_fraction,
                    minimum_energy_threshold,
                    constrained_subdomain_functions_i,
                    _solution_writer_function_for_each_phasefield_fraction,
                    _solution_writer_function_for_each_phasefield_iteration
                    )

            if _solution_writer_function_for_each_phasefield_fraction is None:
                write_solutions()

            energy_vs_iterations = energy_vs_iterations[
                ::max(1, int(len(energy_vs_iterations)/1000))]

            normalized_energy_vs_iterations = \
                [W_j / W_undamaged for W_j in energy_vs_iterations]

            normalized_energy_vs_phasefield = \
                [W_j / W_undamaged for W_j in energy_vs_phasefield]

            material_fraction = dolfin.project(rho, V_p)
            optim.filter.apply_diffusive_smoothing(material_fraction, 1e-5)
            optim.filter.apply_interval_bounds(material_fraction, 0.0, 1.0)

            stress_field = dolfin.project(pk2, V_T)

            maximal_compressive_stress_field = example.utility \
                .compute_maximal_compressive_stress_field(stress_field)

            fraction_compressive_stress_field = example.utility \
                .compute_fraction_compressive_stress_field(stress_field)

            stress_field.rename('stress_tensor', '')
            material_fraction.rename('material_fraction', '')
            maximal_compressive_stress_field.rename('maximal_compression', '')
            fraction_compressive_stress_field.rename('fraction_compression', '')

            if write_results:

                # Create a new file with solver status in the title
                open(os.path.join(RESULTS_OUTDIR_PARENT, problem_title,
                    f'finished_normally({solver_iterations_failed==False}).out'),
                    mode='w').close()

                dolfin.File(os.path.join(results_outdir_functions,
                    "material_fraction.pvd")) << material_fraction

                dolfin.File(os.path.join(results_outdir_functions,
                    "stress_field.pvd")) << stress_field

                dolfin.File(os.path.join(results_outdir_functions,
                    "maximal_compressive_stress_field.pvd")) \
                    << maximal_compressive_stress_field

                dolfin.File(os.path.join(results_outdir_functions,
                    "fraction_compressive_stress_field.pvd")) \
                    << fraction_compressive_stress_field

                np.savetxt(os.path.join(results_outdir_arrays,
                    "normalized_energy_vs_iterations.out"),
                    normalized_energy_vs_iterations)

                np.savetxt(os.path.join(results_outdir_arrays,
                    "normalized_energy_vs_phasefield.out"),
                    normalized_energy_vs_phasefield)

                np.savetxt(os.path.join(results_outdir_arrays,
                    "phasefield_fractions.out"), phasefield_fractions)

            if plot_results or write_results:

                figure_handles = []

                figure_handles.append(
                    example.utility.plot_energy_vs_iterations(
                        normalized_energy_vs_iterations,
                        figname="potential_energy_vs_iterations",
                        ylabel="Normalized potential energy"))

                figure_handles.append(
                    example.utility.plot_energy_vs_iterations(
                        normalized_energy_vs_iterations,
                        figname="potential_energy_vs_iterations_semilogy",
                        ylabel="Normalized potential energy", semilogy=True))

                figure_handles.append(
                    example.utility.plot_energy_vs_phasefields(
                        normalized_energy_vs_phasefield, phasefield_fractions,
                        figname="potential_energy_vs_phasefield",
                        ylabel="Normalized potential energy"))

                figure_handles.append(
                    example.utility.plot_energy_vs_phasefields(
                        normalized_energy_vs_phasefield, phasefield_fractions,
                        figname="potential_energy_vs_phasefield_semilogy",
                        ylabel="Normalized potential energy", semilogy=True))

                figure_handles.append(example.utility.plot_phasefiled(p))
                figure_handles.append(example.utility.plot_material_fraction(material_fraction))

                if write_results:

                    fig_handles = [f[0] for f in figure_handles]
                    fig_names = [f[1] for f in figure_handles]

                    for handle_i, name_i in zip(fig_handles, fig_names):
                        name_i = os.path.join(results_outdir_figures, name_i)

                        handle_i.savefig(name_i+'.png')
                        handle_i.savefig(name_i+'.svg')
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
