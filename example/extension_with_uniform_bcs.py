# -*- coding: utf-8 -*-
"""
Created on 01/10/2018

@author: Danas Sutula

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


# TEMP
def plot(*args, **kwargs):
    '''Plot either `np.ndarray`s or something plottable from `dolfin`.'''
    plt.figure(kwargs.pop('name', None))
    if isinstance(args[0], (np.ndarray, list, tuple)):
        plt.plot(*args, **kwargs)
    else: # just try it anyway
        dolfin.plot(*args, **kwargs)
    plt.show()


### Problem parameters

EPS = 1e-12

# Write solutions every number of iterations
SOLUTION_WRITING_PERIOD = 25

# Residual material integrity when phasefield value is `1`
MINIMUM_MATERIAL_INTEGRITY = Constant(1e-5)

# Exponent in the material degradation law
MATERIAL_DEGRADATION_EXPONENT = 2

# When meshing the domain using `dolfin.UnitSquareMesh`
MESH_PATTERN = "left/right"
# MESH_PATTERN = "crossed"
# MESH_PATTERN = "left"

PLOT_RESULTS = False
SAVE_FIGURES = True

### Results output

PROBLEM_NAME = os.path.splitext(os.path.basename(__file__))[0]
RESULTS_OUTDIR_PARENT = os.path.join("results", PROBLEM_NAME)

if not os.path.isdir(RESULTS_OUTDIR_PARENT):
    os.makedirs(RESULTS_OUTDIR_PARENT)

SAFE_TO_REMOVE_FILE_TYPES = \
    ('.out', '.npy', '.pvd', '.vtu', '.png', '.svg', '.eps', '.pdf')


def phasefield_regularization(p):
    '''Penalty for large phasefield gradients.'''
    return 0.5*dot(grad(p), grad(p))


def material_integrity_law(p):

    m_inf = MINIMUM_MATERIAL_INTEGRITY
    b = MATERIAL_DEGRADATION_EXPONENT

    return m_inf + (1.0-m_inf) * ((1.0+EPS)-p) ** b


def unit_square_mesh(number_of_cells_along_edge, mesh_pattern):

    nx = ny = number_of_cells_along_edge
    mesh = UnitSquareMesh(nx, ny, diagonal=mesh_pattern)

    # mesh.translate(Point(-0.5,-0.5))
    # mesh.scale(GEOMETRIC_SCALING)

    return mesh


def displacement_boundary_conditions(V, load_type):
    '''Displacement boundary conditions concerning a rectangular 2D domain.'''

    mesh = V.mesh()

    x = mesh.coordinates()

    x_min = x.min(axis=0)
    x_max = x.max(axis=0)

    x0, y0 = x_min
    x1, y1 = x_max

    L = x1-x0
    H = y1-y0

    rtol = EPS
    atol = rtol*min(L,H)

    ### Boundary subdomains

    def boundary_bot(x, on_boundary):
        return on_boundary and x[1] < y0+atol

    def boundary_top(x, on_boundary):
        return on_boundary and x[1] > y1-atol

    def boundary_lhs(x, on_boundary):
        return on_boundary and x[0] < x0+atol

    def boundary_rhs(x, on_boundary):
        return on_boundary and x[0] > x1-atol

    ### Boundary conditions

    uyD_top = Expression(' s', s=0.0, degree=0)
    uyD_bot = Expression('-s', s=0.0, degree=0)
    # uxD_lbc = Expression('-s', s=0.0, degree=0)
    # uyD_lbc = Expression('-s', s=0.0, degree=0)

    if load_type == 'biaxial':

        uxD_lhs = Expression('-s', s=0.0, degree=0)
        uxD_rhs = Expression(' s', s=0.0, degree=0)

        bcs = [
            DirichletBC(V.sub(0), uxD_lhs, boundary_lhs),
            DirichletBC(V.sub(0), uxD_rhs, boundary_rhs),
            DirichletBC(V.sub(1), uyD_top, boundary_top),
            DirichletBC(V.sub(1), uyD_bot, boundary_bot),
            ]

        def set_boundary_displacement_values(s):
            uxD_lhs.s = s
            uxD_rhs.s = s
            uyD_top.s = s
            uyD_bot.s = s

    elif load_type == 'uniaxial':

        bcs = [
            DirichletBC(V.sub(1), uyD_top, boundary_top),
            DirichletBC(V.sub(1), uyD_bot, boundary_bot),
            DirichletBC(V.sub(0), Constant(0.0), boundary_top),
            DirichletBC(V.sub(0), Constant(0.0), boundary_bot),
            ]

        def set_boundary_displacement_values(s):
            uyD_top.s = s
            uyD_bot.s = s

    else:
        raise ValueError('Parameter `load_type`?')

    return bcs, set_boundary_displacement_values


def simulate(optimizer, target_phasefield_means, stepsize,
             tolerance, smoothing_weight, minimum_distance,
             stepsize_final=None, tolerance_final=None):

    if stepsize_final is None:
        stepsize_final = stepsize

    if tolerance_final is None:
        tolerance_final = tolerance

    potential_vs_iteration = []
    potential_vs_phasefield = []

    for p_mean_i in target_phasefield_means:
        logger.info(f'Solving for p_mean: {p_mean_i:4.3}')

        p_mean.assign(p_mean_i)

        n, b, potentials = optimizer.optimize(stepsize,
            smoothing_weight, minimum_distance, tolerance)

        potential_vs_iteration.extend(potentials)
        potential_vs_phasefield.append(potentials[-1])

    else:
        logger.info(f'Solving for p_mean: {p_mean_i:4.3} [final]')

        n, b, potentials = optimizer.optimize(stepsize_final,
            smoothing_weight, minimum_distance, tolerance_final)

        potential_vs_iteration.extend(potentials)
        potential_vs_phasefield[-1] = potentials[-1]

    return potential_vs_iteration, potential_vs_phasefield


if __name__ == "__main__":

    ### Discretization parameters

    # Displacement function degree
    displacement_degree = 1

    # Phasefield function degree
    phasefield_degree = 1

    # number_of_cells_along_edge = 61
    # number_of_cells_along_edge = 81
    number_of_cells_along_edge = 161
    # number_of_cells_along_edge = 321


    ### Solver parameters

    # Upperbound on phasefield nodal change per iteration
    stepsize = 0.01

    # Phasefield convergence tolerance (L1-norm)
    tolerance = 1e-3

    # Concerns the final optimization attempt
    stepsize_final = stepsize * 0.1
    tolerance_final = tolerance * 0.1

    # Phasefield smoothing weight
    smoothing_weight = 0.425
    # smoothing_weight = 0.450
    # smoothing_weight = 0.475
    # smoothing_weight = 0.500

    # Minimum distance between local phasefields
    minimum_distance = 0.05

    initial_defect_radius = 3e-2 + 1e-4


    ### Study parameters

    load_type = 'biaxial'
    # load_type = 'uniaxial'

    material_model_name = "LinearElasticModel"
    # material_model_name = "NeoHookeanModel"

    initial_defect_pattern = "uniform"
    # initial_defect_pattern = "checker"

    numbers_of_defects_per_dimension = [3,4,5] # ,6,7,8

    maximum_boundary_displacements = np.linspace(0.01, 0.02, 2)

    target_phasefield_means = np.linspace(0.100, 0.125, 2)

    initial_phasefield_mean = target_phasefield_means[0]
    final_phasefield_mean = target_phasefield_means[-1]


    ### Discretization

    mesh = unit_square_mesh(number_of_cells_along_edge, MESH_PATTERN)

    V_u = VectorFunctionSpace(mesh, 'CG', displacement_degree)
    V_p = FunctionSpace(mesh, 'CG', phasefield_degree)

    u = Function(V_u, name="displacement")
    p = Function(V_p, name="phasefield")

    phasefield_smoother = optim.DiffusionFilter(V_p, kappa=1e-4)


    ### Boundary conditions

    bcs, set_boundary_displacement_values = \
        displacement_boundary_conditions(V_u, load_type)


    ### Material model


    if material_model_name == "LinearElasticModel":

        material_parameters = {'E': Constant(1.0), 'nu': Constant(0.3)}
        material_model = material.LinearElasticModel(material_parameters, u)

    elif material_model_name == "NeoHookeanModel":

        material_parameters = {'E': Constant(1.0), 'nu': Constant(0.3)}
        material_model = material.NeoHookeanModel(material_parameters, u)

    else:
        raise ValueErrpr('Parameter `material_model_name`?')

    psi = material_model.strain_energy_density()
    pk1 = material_model.stress_measure_pk1()
    pk2 = material_model.stress_measure_pk2()

    m = material_integrity_law(p)
    phi = phasefield_regularization(p)

    # Potential/strain energy
    W = m * psi * dx


    ### Phasefield constraints

    # Penalty-like phasefield regularization
    P = phi * dx

    # Target mean phasefield
    p_mean = Constant(0.0)

    # Phasefield fraction constraint(s)
    C = (p - p_mean) * dx


    ### Compute defect/phasefield nucleation sites

    x0, y0 = mesh.coordinates().min(axis=0)
    x1, y1 = mesh.coordinates().max(axis=0)

    defect_xlim = (x0, x1)
    defect_ylim = (y0, y1)

    list_of_defect_coordinates = []

    if initial_defect_pattern == "uniform":
        meshgrid_function = utility.meshgrid_uniform
    elif initial_defect_pattern == "checker":
        meshgrid_function = utility.meshgrid_checker
    else:
        raise ValueError

    for n_i in numbers_of_defects_per_dimension:
        list_of_defect_coordinates.append(meshgrid_function(
            defect_xlim, defect_ylim, n_i, n_i))


    ### Solve the undamaged problem

    # NOTE: The displacement solution will be used for reinitialization.
    # NOTE: The undamged potential energy will be used for normalization.

    assert p.vector().get_local().any() == False

    set_boundary_displacement_values(maximum_boundary_displacements[0])
    optimizer_dummy = optim.TopologyOptimizer(W, P, C, p, p, u, bcs)

    optimizer_dummy.solve_equilibrium_problem()
    undamaged_displacement_vector = u.vector().get_local()
    undamaged_potential_energy = assemble(W)


    ### Main loop for parametric study

    load_type_i = load_type
    material_model_name_i = material_model_name
    boundary_displacement_i = maximum_boundary_displacements[0]

    for defect_coordinates_i in list_of_defect_coordinates:

        defect_count_i = len(defect_coordinates_i)

        ### Results output

        iteration_name_i = (
            f"load({load_type_i})-"
            f"material({material_model_name_i})-"
            f"displacement({boundary_displacement_i})-"
            f"defectPattern({initial_defect_pattern})-"
            f"defectCount({defect_count_i})"
            )

        results_outdir_arrays = os.path.join(
            RESULTS_OUTDIR_PARENT, iteration_name_i, "arrays")

        results_outdir_figures = os.path.join(
            RESULTS_OUTDIR_PARENT, iteration_name_i, "figures")

        results_outdir_functions = os.path.join(
            RESULTS_OUTDIR_PARENT, iteration_name_i, "functions")

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
            results_outdir_functions, u, p,
            period=SOLUTION_WRITING_PERIOD)


        ### Initialize local phasefields

        p_locals = [dolfin.Function(V_p)
            for _ in range(len(defect_coordinates_i))]

        for p_i, x_i in zip(p_locals, defect_coordinates_i):
            utility.insert_defect(p_i, x_i, initial_defect_radius)


        ### Smooth the initial (sharp) phasefields

        for p_i in p_locals:
            phasefield_smoother.apply(p_i)
            # p_arr_i = p_i.vector().get_local()
            # p_arr_i[p_arr_i < 0.0] = 0.0
            # p_arr_i[p_arr_i > 1.0] = 1.0
            # p_i.vector()[:] = p_arr_i


        ### Optimization problem

        optimizer = optim.TopologyOptimizer(W, P, C, p, p_locals, u, bcs,
            external_function=solution_writer.periodic_write)

        # Reinitialize the undamged displacement solution
        u.vector()[:] = undamaged_displacement_vector


        ### Solve optimization problem

        t0 = time.time()

        potential_vs_iteration, potential_vs_phasefield = simulate(optimizer,
            target_phasefield_means, stepsize, tolerance, smoothing_weight,
            minimum_distance, stepsize_final, tolerance_final)

        print(f'\n *** CPU TIME: {time.time()-t0}\n')


        ### Save results

        potential_vs_iteration = [W_i / undamaged_potential_energy
                                  for W_i in potential_vs_iteration]

        potential_vs_phasefield = [W_i / undamaged_potential_energy
                                   for W_i in potential_vs_phasefield]

        np.savetxt(os.path.join(results_outdir_arrays,
            "potential_vs_iteration.out"),
            potential_vs_iteration)

        np.savetxt(os.path.join(results_outdir_arrays,
            "potential_vs_phasefield.out"),
            potential_vs_phasefield)

        np.savetxt(os.path.join(results_outdir_arrays,
            "target_phasefield_means.out"),
            target_phasefield_means)

        solution_writer.write(forcewrite_all=True)


        if SAVE_FIGURES:

            figure_handles = []

            figure_handles.append(
                utility.plot_energy_vs_iterations(
                    potential_vs_iteration))

            figure_handles.append(
                utility.plot_energy_vs_phasefields(
                    potential_vs_phasefield, target_phasefield_means))

            figure_handles.append(
                utility.plot_phasefiled(p))

            if SAVE_FIGURES:

                fig_handles = [f[0] for f in figure_handles]
                fig_names = [f[1] for f in figure_handles]

                for handle_i, name_i in zip(fig_handles, fig_names):
                    name_i = os.path.join(results_outdir_figures, name_i)

                    handle_i.savefig(name_i+'.png')
                    handle_i.savefig(name_i+'.svg')
                    handle_i.savefig(name_i+'.pdf')

            if not PLOT_RESULTS:
                plt.close('all')

