
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

# # 10/15, 22h00m
# filename = "results/square_domain/date(1015_2053)-model(LinearElasticModel)-mesh(81x81)-dims(1x1)-flaws(2)-exx(0.1)-eyy(0.1)-reg(0.47)-inc(0.01)-step(0.01)/functions/p000033.npy"
# num_unticells_x, num_unitcells_y, unitcell_overhang_fraction = 6, 6, 0.0
# unitcell_mirror_x, unitcell_mirror_y = True, True


# filename = "results_ktt/square_domain/date(1018_1225)-model(NeoHookeanModel)-mesh(81x81)-dims(1x1)-flaws(2)-exx(1)-eyy(1)-reg(0.48)-inc(0.025)-step(0.025)/functions/p000010.npy"
# num_unticells_x, num_unitcells_y, unitcell_overhang_fraction = 6, 6, 0.0
# unitcell_mirror_x, unitcell_mirror_y = True, True


# filename = "results_ktt/square_domain/date(1018_1314)-model(NeoHookeanModel)-mesh(81x81)-dims(1x1)-flaws(2)-exx(1)-eyy(1)-reg(0.48)-inc(0.025)-step(0.025)/functions/p000039.npy"
# num_unticells_x, num_unitcells_y, unitcell_overhang_fraction = 4, 4, 0.0
# unitcell_mirror_x, unitcell_mirror_y = True, True


# filename = "results/square_domain/date(1019_0745)-model(NeoHookeanModel)-mesh(161x161)-dims(1x1)-flaws(2)-exx(2)-eyy(2)-reg(0.48)-inc(0.025)-step(0.01)/functions/p000020.npy"
# num_unticells_x, num_unitcells_y, unitcell_overhang_fraction = 4, 4, 0.0
# unitcell_mirror_x, unitcell_mirror_y = True, True


# # results_ktt
# filename = "results/square_domain/date(1019_2315)-model(NeoHookeanModel)-mesh(161x161)-dims(1x1)-flaws(2)-exx(3)-eyy(3)-reg(0.47)-inc(0.01)-step(0.01)/functions/p000007.npy"
# num_unticells_x, num_unitcells_y, unitcell_overhang_fraction = 4, 4, 0.0
# unitcell_mirror_x, unitcell_mirror_y = True, True

# results_ktt
filename = "results/square_domain/date(1020_0958)-model(NeoHookeanModel)-mesh(161x161)-dims(1x1)-flaws(2)-exx(3)-eyy(3)-reg(0.47)-inc(0.01)-step(0.01)/functions/p000020.npy"
num_unticells_x, num_unitcells_y, unitcell_overhang_fraction = 1, 1, 0.0
unitcell_mirror_x, unitcell_mirror_y = True, True



assert os.path.isfile(filename), f'No such file: \"{filename}\"'
results_outdir = os.path.split(os.path.dirname(filename))[0]
results_outdir_figures = os.path.join(results_outdir, RESULTS_SUBDIR_FIGURES)
results_outdir_functions = os.path.join(results_outdir, RESULTS_SUBDIR_FUNCTIONS)

minimum_material_integrity = 1e-4
require_displacement_solution = True

horizontal_strain_values = np.linspace(0.0, 3.0, 50)

material_parameters = {
    'E': dolfin.Constant(1.0),
    'nu': dolfin.Constant(0.0)
    }

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

domain_nx = 161
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

def bcs_set_values(values):
    uxD.assign(values[0])
    uyD.assign(values[1])

ratio_eyy_to_exx = unitcell_eyy / unitcell_exx

bcs_values = np.array([(domain_L * exx,
                        domain_H * ratio_eyy_to_exx * exx)
                       for exx in horizontal_strain_values])

# psi = material.LinearElasticModel(material_parameters, u).strain_energy_density()
psi = material.NeoHookeanModel(material_parameters, u).strain_energy_density()

W = m * psi * dolfin.dx
F = dolfin.derivative(W, u)

equilibrium_solve = utility.equilibrium_solver(
    F, u, bcs, bcs_set_values, bcs_values)


def plot_figures():

    figure_handles = []

    figure_handles.append(plt.figure("Phasefield (unitcell)"))
    dolfin.plot(m_unitcell)

    figure_handles.append(plt.figure("Phasefield (periodic)"))
    dolfin.plot(m)

    return figure_handles


def save_functions():

    dolfin.File(os.path.join(results_outdir_functions, "periodic_u.pvd")) << u
    dolfin.File(os.path.join(results_outdir_functions, "periodic_m.pvd")) << m

    dolfin.File(os.path.join(results_outdir_functions, "unitcell_m.pvd")) << m_unitcell
    dolfin.File(os.path.join(results_outdir_functions, "unitcell_p.pvd")) << p_unitcell


if __name__ == "__main__":

    equilibrium_solve(incremental=True)

    if PLOT_RESULTS or SAVE_RESULTS:

        plt.close('all')
        figure_handles = plot_figures()

        if SAVE_RESULTS:
            save_functions()

        if not PLOT_RESULTS:
            plt.close('all')
