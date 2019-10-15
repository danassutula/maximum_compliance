
import os
import time
import dolfin
import numpy as np
import matplotlib.pyplot as plt

import material
import optim

from example import utility
from example.square_domain import material_integrity

PLOT_RESULTS = True
SAVE_RESULTS = False

RESULTS_SUBDIR_FIGURES = "figures_periodic"
RESULTS_SUBDIR_FUNCTIONS = "functions_periodic"

# filename = "results/square_domain/date(1002_1313)-model(LinearElasticModel)-mesh(81x81)-dims(1x1)-flaws(2)-exx(0.1)-eyy(0.05)-reg(0.475)-inc(0.01)-step(0.01)/functions/p000044.npy"
# filename = "results/square_domain/date(1004_2202)-model(LinearElasticModel)-mesh(81x81)-dims(1x1)-flaws(5)-exx(0.1)-eyy(0.1)-reg(0.45)-inc(0.025)-step(0.01)/functions/p000009.npy"
# filename = "results/square_domain/date(1004_2252)-model(LinearElasticModel)-mesh(81x81)-dims(1x1)-flaws(5)-exx(0.1)-eyy(0.1)-reg(0.475)-inc(0.025)-step(0.01)/functions/p000029.npy"

# filename = "results_ktt/square_domain/date(1005_0107)-model(LinearElasticModel)-mesh(161x161)-dims(1x1)-flaws(5)-exx(0.1)-eyy(0.1)-reg(0.48)-inc(0.05)-step(0.01)/functions/p000010.npy"
# num_unticells_x, num_unitcells_y, unitcell_overhang_fraction = 6, 6, 0.5

# filename = "results_ktt/square_domain/date(1005_0651)-model(LinearElasticModel)-mesh(161x161)-dims(1x1)-flaws(5)-exx(0.1)-eyy(0.1)-reg(0.48)-inc(0.05)-step(0.01)/functions/p000017.npy"
# num_unticells_x, num_unitcells_y, unitcell_overhang_fraction = 5, 5, 0.0


# mesh: 81x81
#
# filename = "results/square_domain/date(1005_2142)-model(LinearElasticModel)-mesh(81x81)-dims(1x1)-flaws(4)-exx(0.1)-eyy(0.1)-reg(0.48)-inc(0.05)-step(0.01)/functions/p000009.npy"
# num_unticells_x, num_unitcells_y, unitcell_overhang_fraction = 6, 6, 0.5
# unitcell_mirror_x, unitcell_mirror_y = False, False
#
# mesh: 161x161
#
# filename = "results_ktt/square_domain/date(1006_1044)-model(LinearElasticModel)-mesh(161x161)-dims(1x1)-flaws(4)-exx(0.1)-eyy(0.1)-reg(0.49)-inc(0.05)-step(0.01)/functions/p000014.npy"
# num_unticells_x, num_unitcells_y, unitcell_overhang_fraction = 4, 4, 0.5
# unitcell_mirror_x, unitcell_mirror_y = True, True    # dolfin.assemble(W) -> 0.015603893090432131
# # unitcell_mirror_x, unitcell_mirror_y = False, False  # dolfin.assemble(W) -> 0.01534135586913777


# filename = "results/square_domain/date(1006_2210)-model(LinearElasticModel)-mesh(81x81)-dims(1x1)-flaws(2)-exx(0.1)-eyy(0.1)-reg(0.48)-inc(0.05)-step(0.01)/functions/p000023.npy"
# num_unticells_x, num_unitcells_y, unitcell_overhang_fraction = 6, 6, 0.25
# unitcell_mirror_x, unitcell_mirror_y = True, True    # dolfin.assemble(W) -> 0.015603893090432131

# 10/15, 22h00m
filename = "results/square_domain/date(1015_2053)-model(LinearElasticModel)-mesh(81x81)-dims(1x1)-flaws(2)-exx(0.1)-eyy(0.1)-reg(0.47)-inc(0.01)-step(0.01)/functions/p000033.npy"
num_unticells_x, num_unitcells_y, unitcell_overhang_fraction = 6, 6, 0.0
unitcell_mirror_x, unitcell_mirror_y = True, True


assert os.path.isfile(filename), f'No such file: \"{filename}\"'
results_outdir = os.path.split(os.path.dirname(filename))[0]
results_outdir_figures = os.path.join(results_outdir, RESULTS_SUBDIR_FIGURES)
results_outdir_functions = os.path.join(results_outdir, RESULTS_SUBDIR_FUNCTIONS)

require_displacement_solution = True
minimum_material_integrity = 1e-5

material_parameters = {
    'E': dolfin.Constant(1.0),
    'nu': dolfin.Constant(0.4)
    }

extension_strain_final = 0.20
extension_strain_initial = 0.20
extension_strain_stepsize = 0.20
strain_stepsize_refinements = 2

element_degree = 1
element_family = "CG"
mesh_diagonal = "crossed"

# unitcell_mirror_x = True
# unitcell_mirror_y = True

# num_unticells_x = 3
# num_unitcells_y = 3
# unitcell_overhang_fraction = 0


### Load unitcell solution

unitcell_nx, unitcell_ny = \
    [int(s) for s in utility.extract_substring(
     filename, str_beg="mesh(", str_end=")").split("x")]

unitcell_L, unitcell_H = \
    [float(s) for s in utility.extract_substring(
     filename, str_beg="dims(", str_end=")").split("x")]

unitcell_exx = float(utility.extract_substring(filename, "exx(", ")"))
unitcell_eyy = float(utility.extract_substring(filename, "eyy(", ")"))

unitcell_p0, unitcell_p1 = [0,0], [unitcell_L, unitcell_H]

mesh_unitcell = utility.rectangle_mesh(
    unitcell_p0, unitcell_p1, unitcell_nx, unitcell_ny, mesh_diagonal)

V_p_unitcell = V_m_unitcell = dolfin.FunctionSpace(
    mesh_unitcell, element_family, element_degree)

p_unitcell = dolfin.Function(V_p_unitcell)
p_unitcell.vector()[:] = np.load(filename)

m_unitcell = dolfin.project(material_integrity(
    p_unitcell, minimum_material_integrity), V_m_unitcell)

optim.filter.apply_interval_bounds(
    m_unitcell, minimum_material_integrity, 1.0)

### Tile the unitcell solution to obtain the periodic solution

domain_nx = 250
domain_ny = domain_nx

domain_L = 1.0
domain_H = 1.0

domain_p0, domain_p1 = [0,0], [domain_L, domain_H]

mesh = utility.rectangle_mesh(
    domain_p0, domain_p1, domain_nx, domain_ny, mesh_diagonal)

V_p = V_m = dolfin.FunctionSpace(mesh, element_family, element_degree)

# p = utility.project_function_periodically(
#     p_unitcell, num_unticells_x, num_unitcells_y,
#     V_p, unitcell_mirror_x, unitcell_mirror_y,
#     unitcell_overhang_fraction)

m = utility.project_function_periodically(
    m_unitcell, num_unticells_x, num_unitcells_y,
    V_m, unitcell_mirror_x, unitcell_mirror_y,
    unitcell_overhang_fraction)

# optim.filter.apply_interval_bounds(p, 0, 1.0)
optim.filter.apply_interval_bounds(m, minimum_material_integrity, 1.0)


### Displacement problem

V_u = dolfin.VectorFunctionSpace(mesh, 'CG', 1)
u = dolfin.Function(V_u)

uxD = dolfin.Constant(0.0)
uyD = dolfin.Constant(0.0)

bcs = [
    dolfin.DirichletBC(V_u.sub(0), uxD, "x[0] > 1.0-DOLFIN_EPS", method="pointwise"),
    dolfin.DirichletBC(V_u.sub(1), uyD, "x[1] > 1.0-DOLFIN_EPS", method="pointwise"),
    dolfin.DirichletBC(V_u.sub(0),   0, "x[0] < DOLFIN_EPS"    , method="pointwise"),
    dolfin.DirichletBC(V_u.sub(1),   0, "x[1] < DOLFIN_EPS"    , method="pointwise"),
    ]

psi = material.LinearElasticModel(material_parameters, u).strain_energy_density()
# psi = material.NeoHookeanModel(material_parameters, u).strain_energy_density()

W = m * psi * dolfin.dx
F = dolfin.derivative(W, u)
J = dolfin.derivative(F, u)


def solve_displacement_problem(exx, eyy="auto", method="newton"):

    if eyy is "auto":
        eyy = unitcell_eyy / unitcell_exx * exx

    uxD.assign(domain_L * exx)
    uyD.assign(domain_H * eyy)

    try:
        dolfin.solve(F==0, u, bcs=bcs, J=J,
            solver_parameters={
                "nonlinear_solver": method,
                "newton_solver": {
                    "maximum_iterations": 10,
                    "error_on_nonconvergence": True,
                    },
                "snes_solver": {
                    'line_search': "bt", # "basic"
                    "maximum_iterations": 10,
                    "error_on_nonconvergence": True,
                    }
            })
    except:
        return False
    else:
        return True


def solve_displacement_problem_incrementally(
        strain_initial, strain_maximum, stepsize, maximum_refinements=0):

    refinements_count = 0
    strain_current = strain_initial
    u_arr_old = u.vector().get_local()

    while strain_current <= strain_maximum:
        print(f"INFO: Solving for {strain_current:.3f} ...")

        if solve_displacement_problem(strain_current):
            u_arr_old = u.vector().get_local()

        else:

            u.vector()[:] = u_arr_old
            strain_current -= stepsize

            if refinements_count >= maximum_refinements:
                print(f"ERROR: Could not solve.")
                break

            refinements_count += 1
            stepsize /= 2

        strain_current += stepsize


def plot_figures():

    figure_handles = []

    figure_handles.append(plt.figure("Phasefield (unitcell)"))
    dolfin.plot(m_unitcell)

    figure_handles.append(plt.figure("Phasefield (periodic)"))
    dolfin.plot(m)

    return figure_handles


def save_functions():

    if u.vector().get_local().any():
        dolfin.File(os.path.join(results_outdir_functions, "u_periodic.pvd")) << u

    dolfin.File(os.path.join(results_outdir_functions, "p_unitcell.pvd")) << p_unitcell
    dolfin.File(os.path.join(results_outdir_functions, "m_unitcell.pvd")) << m_unitcell
    dolfin.File(os.path.join(results_outdir_functions, "m_periodic.pvd")) << m


def save_figures(figure_handles):
    pass


if __name__ == "__main__":

    if require_displacement_solution:
        solve_displacement_problem_incrementally(
            extension_strain_initial, extension_strain_final,
            extension_strain_stepsize, strain_stepsize_refinements)

    if PLOT_RESULTS or SAVE_RESULTS:
        plt.close('all')
        figure_handles = plot_figures()

        if SAVE_RESULTS:
            save_figures(figure_handles)
            save_functions()

        if not PLOT_RESULTS:
            plt.close('all')
