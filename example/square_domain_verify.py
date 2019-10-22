
import os
import time
import dolfin
import numpy as np
import matplotlib.pyplot as plt

import material
import optim

from example import utility
from example.square_domain import material_integrity


SAVE_RESULTS = True

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

# # results_ktt
# filename = "results/square_domain/date(1020_0958)-model(NeoHookeanModel)-mesh(161x161)-dims(1x1)-flaws(2)-exx(3)-eyy(3)-reg(0.47)-inc(0.01)-step(0.01)/functions/p000020.npy"
# num_unticells_x, num_unitcells_y, unitcell_overhang_fraction = 1, 1, 0.0
# unitcell_mirror_x, unitcell_mirror_y = True, True

# results_ktt
# filename = "results/square_domain/date(1020_2044)-model(NeoHookeanModel)-mesh(161x161)-dims(1x1)-flaws(2)-exx(3)-eyy(3)-reg(0.47)-inc(0.01)-step(0.01)/functions/p000022.npy"
filename = "results/square_domain/date(1020_2044)-model(NeoHookeanModel)-mesh(161x161)-dims(1x1)-flaws(2)-exx(3)-eyy(3)-reg(0.47)-inc(0.01)-step(0.01)/functions/p000021.npy"
num_unticells_x, num_unitcells_y, unitcell_overhang_fraction = 4, 4, 0.0
unitcell_mirror_x, unitcell_mirror_y = True, True


assert os.path.isfile(filename), f'No such file: \"{filename}\"'
results_outdir = os.path.split(os.path.dirname(filename))[0]
results_outdir_figures = os.path.join(results_outdir, RESULTS_SUBDIR_FIGURES)
results_outdir_functions = os.path.join(results_outdir, RESULTS_SUBDIR_FUNCTIONS)

minimum_material_integrity = 1e-5

horizontal_strain_values = np.linspace(0.0, 3.0, 100)
# NOTE: Vertical strain will be computed automatically

material_parameters = {
    'E': dolfin.Constant(1.0),
    'nu': dolfin.Constant(0.0),
    }

element_degree = 1
element_family = "CG"
mesh_diagonal = "crossed"


### Load unitcell solution

unitcell_nx, unitcell_ny = \
    [int(s) for s in utility.extract_substring(
     filename, str_beg="mesh(", str_end=")").split("x")]

unitcell_L, unitcell_H = \
    [float(s) for s in utility.extract_substring(
     filename, str_beg="dims(", str_end=")").split("x")]

unitcell_exx = float(utility.extract_substring(filename, "exx(", ")"))
unitcell_eyy = float(utility.extract_substring(filename, "eyy(", ")"))

material_model_name = utility.extract_substring(filename, "model(", ")")

unitcell_p0 = [0,0]
unitcell_p1 = [unitcell_L, unitcell_H]

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

p_unitcell.rename("p_unitcell", '')
m_unitcell.rename("m_unitcell", '')


### Tile the unitcell solution to obtain the periodic solution

domain_nx = 300
domain_ny = domain_nx

# domain_L = unitcell_L
# domain_H = unitcell_H

domain_L = unitcell_L * num_unticells_x
domain_H = unitcell_H * num_unitcells_y

domain_p0 = [0,0]
domain_p1 = [domain_L, domain_H]

mesh = utility.rectangle_mesh(
    domain_p0, domain_p1, domain_nx, domain_ny, mesh_diagonal)

V_p = V_m = dolfin.FunctionSpace(mesh, element_family, element_degree)

p = utility.project_function_periodically(
    p_unitcell, num_unticells_x, num_unitcells_y,
    V_p, unitcell_mirror_x, unitcell_mirror_y,
    unitcell_overhang_fraction)

optim.filter.apply_interval_bounds(p, 0.0, 1.0)

m = dolfin.project(material_integrity(p, minimum_material_integrity), V_m)
optim.filter.apply_interval_bounds(m, minimum_material_integrity, 1.0)

p.rename("p", '')
m.rename("m", '')


### Displacement problem

V_u = dolfin.VectorFunctionSpace(mesh, 'CG', 1)

u = dolfin.Function(V_u)
u.rename("u", '')

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

bcs_values = np.array([(domain_L*exx, domain_H*ratio_eyy_to_exx*exx)
                       for exx in horizontal_strain_values])

if material_model_name == "LinearElasticModel":
    material_model = material.LinearElasticModel(material_parameters, u)
elif material_model_name == "NeoHookeanModel":
    material_model = material.NeoHookeanModel(material_parameters, u)
else:
    raise ValueError('material_model_name')

psi_0 = material_model.strain_energy_density()
pk2_0 = material_model.stress_measure_pk2()

psi = m * psi_0
pk2 = m * pk2_0

W = psi * dolfin.dx
F = dolfin.derivative(W, u)

equilibrium_solve = utility.equilibrium_solver(
    F, u, bcs, bcs_set_values, bcs_values)

if material_model_name == "LinearElasticModel":
    equilibrium_solve(incremental=False)
else:
    equilibrium_solve(incremental=True)


### Post-process

V_S = dolfin.TensorFunctionSpace(mesh, 'DG', 0)

stress_field = dolfin.project(pk2, V_S)
stress_field.rename("stress (PK2)", '')

maximal_compressive_stress_field = \
    utility.compute_maximal_compressive_stress_field(stress_field)

fraction_compressive_stress_field = \
    utility.compute_fraction_compressive_stress_field(stress_field)

maximal_compressive_stress_field.rename('maximal_compression', '')
fraction_compressive_stress_field.rename('fraction_compression', '')


def save_functions():

    dolfin.File(os.path.join(results_outdir_functions, "periodic_u.pvd")) << u
    dolfin.File(os.path.join(results_outdir_functions, "periodic_m.pvd")) << m
    dolfin.File(os.path.join(results_outdir_functions, "periodic_p.pvd")) << p

    dolfin.File(os.path.join(results_outdir_functions, "unitcell_m.pvd")) << m_unitcell
    dolfin.File(os.path.join(results_outdir_functions, "unitcell_p.pvd")) << p_unitcell

    dolfin.File(os.path.join(results_outdir_functions, "stress_field.pvd")) << stress_field

    dolfin.File(os.path.join(results_outdir_functions,
        "maximal_compressive_stress_field.pvd")) << maximal_compressive_stress_field

    dolfin.File(os.path.join(results_outdir_functions,
        "fraction_compressive_stress_field.pvd")) << fraction_compressive_stress_field


if __name__ == "__main__":

    if SAVE_RESULTS:
        save_functions()
