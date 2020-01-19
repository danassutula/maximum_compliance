
import os
import math
import dolfin
import numpy as np
import matplotlib.pyplot as plt

import material
import optim

from examples import utility
from examples.square_domain_cases import material_integrity


SAVE_RESULTS = True

results_outdir = "results/square_domain_cases/reference_solution"
results_outdir_functions = os.path.join(results_outdir, "functions")

load_mode = "vertical"
model_name = "LinearElastic"
# model_name = "NeoHookean"

material_parameters = {
    'E': dolfin.Constant(1.0),
    'nu': dolfin.Constant(0.0),
    }

minimum_material_integrity = 1e-5

domain_L = 1
ratio_H2L = 0.25

num_defects_x = 4
grid_angle = math.atan(0.5)

ratio_y2x = 0.5*math.tan(grid_angle)
hx = domain_L / (num_defects_x - 1)
hy = ratio_y2x * hx

num_defects_y = int(ratio_H2L * domain_L / hy + 0.5) + 1
domain_H = (num_defects_y - 1) * hy

defect_radius = hx / 2 * 0.75
ellipse_param_a = defect_radius
ellipse_param_b_as_nelem = 4

domain_exx = 0
domain_eyy = 1

number_of_loading_steps = 10

element_degree = 1
element_family = "CG"
mesh_diagonal = "crossed"

domain_nx = 500
domain_ny = int(domain_H / domain_L * domain_nx + 0.5)

domain_p0 = [0,0]
domain_p1 = [domain_L, domain_H]

mesh = utility.rectangle_mesh(
    domain_p0, domain_p1, domain_nx, domain_ny, mesh_diagonal)

V_p = V_m = dolfin.FunctionSpace(mesh, element_family, element_degree)

xs = utility.meshgrid_checker_symmetric(
    domain_p0, domain_p1, nx=num_defects_x*2-1, ny=num_defects_y*2-1)

ellipse_param_b = (domain_L/domain_nx) * ellipse_param_b_as_nelem

p = dolfin.Function(V_p)
p.assign(sum(utility.make_defect_like_phasefield_array(
    V_p, xs, ellipse_param_a, ellipse_param_b, norm=2)))

optim.filter.apply_diffusive_smoothing(p, kappa=1e-4)
optim.filter.trimoff_function_values(p, 0.0, 1.0)
optim.filter.rescale_function_values(p, 0.0, 1.0)

m = dolfin.project(material_integrity(p, minimum_material_integrity), V_m)
optim.filter.trimoff_function_values(m, minimum_material_integrity, 1.0)

p.rename("p", '')
m.rename("m", '')


### Displacement problem

V_u = dolfin.VectorFunctionSpace(mesh, 'CG', 1)
u = dolfin.Function(V_u)
u.rename("u", '')

bcs, bcs_set_values = utility.uniform_extension_bcs(V_u, load_mode)

if domain_exx:
    ratio_eyy_to_exx = domain_eyy / domain_exx
    bcs_values = np.array([(domain_L*exx, domain_H*ratio_eyy_to_exx*exx) for
        exx in np.linspace(0.0, domain_exx, number_of_loading_steps+1)[1:]])

else:
    bcs_values = np.array([(0.0, domain_H*eyy) for eyy in
        np.linspace(0.0, domain_eyy, number_of_loading_steps+1)[1:]])

if model_name == "LinearElastic":
    material_model = material.LinearElasticModel(material_parameters, u)
elif model_name == "NeoHookean":
    material_model = material.NeoHookeanModel(material_parameters, u)
else:
    raise ValueError('model_name')

psi_0 = material_model.strain_energy_density()
pk2_0 = material_model.stress_measure_pk2()

psi = m * psi_0
pk2 = m * pk2_0

W = psi * dolfin.dx
F = dolfin.derivative(W, u)

equilibrium_solve = utility.equilibrium_solver(
    F, u, bcs, bcs_set_values, bcs_values)

if model_name == "LinearElastic":
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

    dolfin.File(os.path.join(results_outdir_functions, "u.pvd")) << u
    dolfin.File(os.path.join(results_outdir_functions, "m.pvd")) << m
    dolfin.File(os.path.join(results_outdir_functions, "p.pvd")) << p

    dolfin.File(os.path.join(results_outdir_functions, "stress_field.pvd")) << stress_field

    dolfin.File(os.path.join(results_outdir_functions,
        "maximal_compressive_stress_field.pvd")) << maximal_compressive_stress_field

    dolfin.File(os.path.join(results_outdir_functions,
        "fraction_compressive_stress_field.pvd")) << fraction_compressive_stress_field


if __name__ == "__main__":

    if SAVE_RESULTS:
        save_functions()

    phasefield_fraction = dolfin.assemble(p*dolfin.dx) \
                        / dolfin.assemble(1*dolfin.dx(domain=mesh))

    strain_energy = dolfin.assemble(W)

    m_arr = m.vector().get_local()
    u_arr = u.vector().get_local()

    m.vector()[:] = 1.0
    equilibrium_solve()

    strain_energy_ref = dolfin.assemble(W)

    m.vector()[:] = m_arr
    u.vector()[:] = u_arr

    compliance_factor = strain_energy_ref / strain_energy
    print(f'Compliance_factor: {compliance_factor:.4f}')
