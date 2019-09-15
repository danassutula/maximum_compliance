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
import utility

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


def phasefield_regularization(p):
    '''Phasefield regularization.'''
    return dolfin.grad(p)**2


def material_integrity_model(p):
    '''Material integrity given the value of phasefield.

    Notes
    -----
    The returned value should be in range [rho_min, 1].

    '''

    # Minimum (residual) material integrity
    rho_min = Constant(1e-4)

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

    writing_period = 100
    write_phasefield_pvd = True
    write_displacements_pvd = False

    ### Problem parameters

    load_type = [
        "biaxial_uniform",
        # "uniaxial_uniform",
        ]

    material_model_name = [
        "LinearElasticModel",
        # "NeoHookeanModel",
        ]

    boundary_displacement_value = [
        0.1,
        # [0.1, 0.2],
        ] # NOTE: If an element is a sequence, the end-value of the sequence
          #       will be used but the value will be attained incrementally.

    slack = 0.0
    defect_nucleation_centers = [
        np.array([[0,0.5-slack],[1,0.5+slack],[0.5+slack,0],[0.5-slack,1]]),
        # np.array([[slack,0],[1,slack],[1-slack,1],[0,1-slack]]), # Good
        # optim.helper.meshgrid_uniform([0,1], [0,1], 4, 4),
        # optim.helper.pertub_gridrows(optim.helper.meshgrid_uniform(
        #     [0,1], [0,1], nrow=6, ncol=6), nrow=6, ncol=6, dx=1e-2),
        ]

    phasefield_regularization_weight = [
        # 0.300,
        # 0.350,
        # 0.400,
        # 0.425,
        # 0.450,
        # 0.460,
        0.475,
        # 0.5,
        ]

    defect_nucleation_diameter = "default" # or `None`, or "default"
    defect_nucleation_elemental_diameter = 8.0 # Default fallback

    phasefield_collision_distance = 0.25 # `None`, or "default"
    phasefield_collision_elemental_distance = 12.0 # Default fallback

    # Phasefield domain fraction increment
    phasefield_fraction_increment = [
        # 0.02000,
        # 0.01000,
        0.00500,
        # 0.00250,
        ]

    # Phasefield iteration stepsize (L_inf-norm)
    phasefield_iteration_stepsize = [
        # 0.050,
        0.025,
        # 0.010,
        ]

    # Phasefield convergence tolerance (L_inf-norm)
    phasefield_convergence_stepsize_fraction = 1/3

    phasefield_maximum_domain_fraction = 1.0

    if phasefield_maximum_domain_fraction != 1.0:
        logger.warning('phasefield_maximum_domain_fraction != 1.0')

    ### Discretization parameters

    # Displacement function degree
    displacement_degree = 1 # 1, 2

    # Phasefield function degree
    phasefield_degree = 1

    # mesh_diagonal = "left/right"
    mesh_diagonal = "crossed"
    # mesh_diagonal = "left"

    number_of_elements_along_edge = [
        # 30,
        # 40,
        # 41,
        # 60,
        # 61,
        # 80,
        # 81,
        # 121,
        # 160,
        161,
        # 320,
        # 321,
        ] # NOTE: Should try both even and odd numbers of elements

    ### Input control parameters (if not defined above)

    if boundary_displacement_value is None or not boundary_displacement_value:
        boundary_displacement_value = eval(input('\nboundary_displacement_value:\n'))

    if phasefield_regularization_weight is None or not phasefield_regularization_weight:
        phasefield_regularization_weight = eval(input('\nphasefield_regularization_weight:\n'))

    if phasefield_fraction_increment is None or not phasefield_fraction_increment:
        phasefield_fraction_increment = eval(input('\nphasefield_fraction_increment:\n'))

    if phasefield_iteration_stepsize is None or not phasefield_iteration_stepsize:
        phasefield_iteration_stepsize = eval(input('\nphasefield_iteration_stepsize:\n'))

    if number_of_elements_along_edge is None or not number_of_elements_along_edge:
        number_of_elements_along_edge = eval(input('\nnumber_of_elements_along_edge:\n'))

    ### Input fixed parameters (if not defined above)

    if defect_nucleation_diameter is None:
        defect_nucleation_diameter = eval(input('\ndefect_nucleation_diameter:\n'))

    if phasefield_collision_distance is None:
        phasefield_collision_distance = eval(input('\nphasefield_collision_distance:\n'))

    ### Control parameter grid

    outer_loop_parameters = optim.helper.make_parameter_combinations(
        load_type,
        material_model_name,
        number_of_elements_along_edge,
        boundary_displacement_value,
        )

    inner_loop_parameters = optim.helper.make_parameter_combinations(
        defect_nucleation_centers,
        phasefield_regularization_weight,
        phasefield_fraction_increment,
        phasefield_iteration_stepsize,
        )

    for (
        load_type_i,
        material_model_name_i,
        number_of_elements_along_edge_i,
        boundary_displacement_value_i,
        ) in outer_loop_parameters:

        ### Discretization

        mesh = utility.unit_square_mesh(
            number_of_elements_along_edge_i, mesh_diagonal)

        mesh_x0, mesh_y0 = mesh.coordinates().min(axis=0)
        mesh_x1, mesh_y1 = mesh.coordinates().max(axis=0)

        defect_xlim = mesh_x0, mesh_x1
        defect_ylim = mesh_y0, mesh_y1

        ### Boundary subdomains

        boundary_bot, boundary_rhs, boundary_top, boundary_lhs = \
            utility.boundaries_of_rectangular_domain(mesh)

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

        if load_type_i == "biaxial_uniform":
            bcs, set_value_bcs = utility.uniform_biaxial_extension_bcs(V_u)

        elif load_type_i == "uniaxial_uniform":
            bcs, set_value_bcs = utility.uniform_uniaxial_extension_bcs(V_u)

        else:
            raise ValueError('Parameter `load_type_i`?')

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

        rho = material_integrity_model(p)
        phi = phasefield_regularization(p)

        psi = rho * psi_0
        pk1 = rho * pk1_0
        pk2 = rho * pk2_0

        ### Objective functionals

        # Potential energy (strain energy only)
        W = psi * dx

        # Phasefield regularization
        R = phi * dx

        # Variational form of equilibrium
        F = derivative(W, u)

        ### Solve for undamaged material

        p.vector()[:] = 0.0
        u.vector()[:] = 0.0

        if hasattr(boundary_displacement_value_i, '__len__'):
            for boundary_displacement_value_i \
                in boundary_displacement_value_i:
                set_value_bcs(boundary_displacement_value_i)
                dolfin.solve(F==0, u, bcs, solver_parameters={"nonlinear_solver": "snes"})
        else:
            set_value_bcs(boundary_displacement_value_i)
            dolfin.solve(F==0, u, bcs, solver_parameters={"nonlinear_solver": "snes"})

        undamaged_solution_vector = u.vector().get_local()
        undamaged_potential_energy = dolfin.assemble(W)

        for (
            defect_nucleation_centers_i,
            phasefield_regularization_weight_i,
            phasefield_fraction_increment_i,
            phasefield_iteration_stepsize_i,
            ) in inner_loop_parameters:

            problem_start_time = time.time()

            if not (defect_nucleation_diameter is None or \
                    defect_nucleation_diameter is "default"):
                defect_nucleation_diameter_i = defect_nucleation_diameter
            else:
                defect_nucleation_diameter_i = \
                    defect_nucleation_elemental_diameter * mesh.hmax() * (1+EPS)

            if not (phasefield_collision_distance is None or \
                    phasefield_collision_distance is "default"):
                phasefield_collision_distance_i = phasefield_collision_distance
            else:
                phasefield_collision_distance_i = \
                    phasefield_collision_elemental_distance * mesh.hmax() * (1+EPS)

            phasefield_convergence_tolerance_i = \
                phasefield_iteration_stepsize_i * phasefield_convergence_stepsize_fraction

            problem_title = (
                f"date({time.strftime('%m%d_%H%M')})-"
                f"load({load_type_i})-"
                f"model({material_model_name_i})-"
                f"mesh({mesh.num_vertices()})-"
                f"count({len(defect_nucleation_centers_i)})-"
                f"disp({str(f'{boundary_displacement_value_i:.3f}').replace('.','d')})-"
                f"regul({str(f'{phasefield_regularization_weight_i:.3f}').replace('.','d')})-"
                f"inc({str(f'{phasefield_fraction_increment_i:.3f}').replace('.','d')})-"
                f"step({str(f'{phasefield_iteration_stepsize_i:.3f}').replace('.','d')})"
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

                utility.remove_outfiles(results_outdir, SAFE_TO_REMOVE_FILE_TYPES)
                utility.remove_outfiles(results_outdir_arrays, SAFE_TO_REMOVE_FILE_TYPES)
                utility.remove_outfiles(results_outdir_figures, SAFE_TO_REMOVE_FILE_TYPES)
                utility.remove_outfiles(results_outdir_functions, SAFE_TO_REMOVE_FILE_TYPES)

                solution_writer = utility.PeriodicSolutionWriter(
                    results_outdir_functions, u, p, writing_period,
                    write_phasefield_pvd, write_displacements_pvd)

            else:
                solution_writer = type("DummySolutionWriter", (),
                    dict(write=lambda calling_object : None,
                         periodic_write=lambda calling_object : None))

            u.vector()[:] = undamaged_solution_vector

            solver_iterations_failed, energy_vs_iterations, energy_vs_phasefield, \
            phasefield_fractions, topology_optimizer, p_locals, p_mean_target = \
                optim.helper.solve_compliance_maximization_problem(
                    W, R, u, p, bcs,
                    defect_nucleation_centers_i,
                    defect_nucleation_diameter_i,
                    phasefield_collision_distance_i,
                    phasefield_iteration_stepsize_i,
                    phasefield_fraction_increment_i,
                    phasefield_regularization_weight_i,
                    phasefield_convergence_tolerance_i,
                    phasefield_maximum_domain_fraction,
                    solution_writer.periodic_write,
                    )

            energy_vs_iterations = energy_vs_iterations[
                ::max(1, int(len(energy_vs_iterations)/500))]

            normalized_energy_vs_iterations = [W_i / undamaged_potential_energy
                                               for W_i in energy_vs_iterations]

            normalized_energy_vs_phasefield = [W_i / undamaged_potential_energy
                                               for W_i in energy_vs_phasefield]

            material_fraction = dolfin.project(rho, V_p)
            optim.helper.apply_diffusive_smoothing(material_fraction, 1e-5)
            optim.helper.apply_interval_bounds(material_fraction, 0.0, 1.0)

            stress_field = dolfin.project(pk2, V_T)

            maximal_compressive_stress_field = optim.helper \
                .compute_maximal_compressive_stress_field(stress_field)

            fraction_compressive_stress_field = optim.helper \
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

                np.save(os.path.join(results_outdir_functions,
                    "material_fraction_dofs.npy"),
                    material_fraction.vector().get_local())

                dolfin.File(os.path.join(results_outdir_functions,
                    "material_fraction.pvd")) << material_fraction

                dolfin.File(os.path.join(results_outdir_functions,
                    "stress_field.pvd")) << stress_field

                dolfin.File(os.path.join(results_outdir_functions,
                    "maximal_compressive_stress_field.pvd")) << \
                    maximal_compressive_stress_field

                dolfin.File(os.path.join(results_outdir_functions,
                    "fraction_compressive_stress_field.pvd")) << \
                    fraction_compressive_stress_field

                np.savetxt(os.path.join(results_outdir_arrays,
                    "normalized_energy_vs_iterations.out"),
                    normalized_energy_vs_iterations)

                np.savetxt(os.path.join(results_outdir_arrays,
                    "normalized_energy_vs_phasefield.out"),
                    normalized_energy_vs_phasefield)

                np.savetxt(os.path.join(results_outdir_arrays,
                    "phasefield_fractions.out"), phasefield_fractions)

                solution_writer.write(forcewrite_all=True)

            if plot_results or write_results:

                figure_handles = []

                figure_handles.append(utility.plot_energy_vs_iterations(
                    normalized_energy_vs_iterations,
                    figname="potential_energy_vs_iterations",
                    ylabel="Normalized potential energy"))

                # figure_handles.append(utility.plot_energy_vs_iterations(
                #     normalized_energy_vs_iterations,
                #     figname="potential_energy_vs_iterations_semilogy",
                #     ylabel="Normalized potential energy", semilogy=True))

                figure_handles.append(utility.plot_energy_vs_phasefields(
                    normalized_energy_vs_phasefield, phasefield_fractions,
                    figname="potential_energy_vs_phasefield",
                    ylabel="Normalized potential energy"))

                # figure_handles.append(utility.plot_energy_vs_phasefields(
                #     normalized_energy_vs_phasefield, phasefield_fractions,
                #     figname="potential_energy_vs_phasefield_semilogy",
                #     ylabel="Normalized potential energy", semilogy=True))

                figure_handles.append(utility.plot_phasefiled(p))

                figure_handles.append(
                    utility.plot_material_fraction(material_fraction))

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

            logger.info("Done solving problem\n\t"
                + problem_title.replace('-','\n\t'))

            problem_finish_time = time.time()
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
