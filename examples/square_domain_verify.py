'''
The purposes of this script is to assess a unitcell solution that has been
obtained by executing the script `examples/square_domain_cases.py`.

The assessment of the unitcell solution, specifically the material phasefield
solution, is done by tilling and/or mirroring the solution along the x,y-axis
several times and then solving this new problem for the displacement field.

From the displacement field (and other relevant fields, e.g. the stress fields)
the performance of the unitcell solution can be more easily judged.

'''

import os
import math
import time
import dolfin
import numpy as np
import matplotlib.pyplot as plt

import material
import optim

from examples import utility
from examples.square_domain_cases import material_integrity


SAVE_RESULTS = True
COMPUTE_STRAIN = True
COMPUTE_STRESS = False

OVERRIDE_LOAD_MODE = None # "vertical"
OVERRIDE_MODEL_NAME = None # "LinearElasticModel"
OVERRIDE_MODEL_NAME = None # "NeoHookeanModel"
OVERRIDE_UNITCELL_EXX = None # 2
OVERRIDE_UNITCELL_EYY = None # 2

### Unitcell solution file

# dist(1/4)-exx(1)-eyy(1)
# filename = "results/square_domain_cases/date(0226_1504)-model(NeoHookeanModel)" \
#            "-load(biaxial)-mesh(80x80)-dims(1x1)-flaws(4)-exx(1)-eyy(1)-reg(0.5)" \
#            "-dist(0.25)-inc(0.02)-step(0.1)/functions/p000011_000523.npy"
# filename = "results/square_domain_cases/date(0226_1325)-model(NeoHookeanModel)" \
#            "-load(biaxial)-mesh(160x160)-dims(1x1)-flaws(4)-exx(1)-eyy(1)-reg(0.5)" \
#            "-dist(0.25)-inc(0.02)-step(0.1)/functions/p000008_000400.npy"

# dist(1/3)-exx(1)-eyy(1)
# filename = "results/square_domain_cases/date(0226_1349)-model(NeoHookeanModel)" \
#            "-load(biaxial)-mesh(80x80)-dims(1x1)-flaws(4)-exx(1)-eyy(1)-reg(0.5)" \
#            "-dist(0.333)-inc(0.02)-step(0.1)/functions/p000005_000250.npy"
# filename = "results/square_domain_cases/date(0226_0923)-model(NeoHookeanModel)" \
#            "-load(biaxial)-mesh(160x160)-dims(1x1)-flaws(4)-exx(1)-eyy(1)-reg(0.5)" \
#            "-dist(0.333)-inc(0.02)-step(0.1)/functions/p000006_000300.npy"


# dist(1/4)-exx(1.5)-eyy(1.5)
# filename = "results/square_domain_cases/date(0226_1607)-model(NeoHookeanModel)" \
#            "-load(biaxial)-mesh(80x80)-dims(1x1)-flaws(4)-exx(1.5)-eyy(1.5)-reg(0.5)" \
#            "-dist(0.25)-inc(0.02)-step(0.1)/functions/p000006_000300.npy"
# filename = "results/square_domain_cases/date(0226_1640)-model(NeoHookeanModel)" \
#            "-load(biaxial)-mesh(160x160)-dims(1x1)-flaws(4)-exx(1.5)-eyy(1.5)-reg(0.5)" \
#            "-dist(0.25)-inc(0.02)-step(0.1)/functions/p000008_000400.npy"

# dist(1/3)-exx(1.5)-eyy(1.5)
# filename = "results/square_domain_cases/date(0226_1617)-model(NeoHookeanModel)" \
#            "-load(biaxial)-mesh(80x80)-dims(1x1)-flaws(4)-exx(1.5)-eyy(1.5)-reg(0.5)" \
#            "-dist(0.333)-inc(0.02)-step(0.1)/functions/p000004_000200.npy"
# filename = "results/square_domain_cases/date(0226_1729)-model(NeoHookeanModel)" \
#            "-load(biaxial)-mesh(160x160)-dims(1x1)-flaws(4)-exx(1.5)-eyy(1.5)-reg(0.5)" \
#            "-dist(0.333)-inc(0.02)-step(0.1)/functions/p000005_000250.npy"

# filename = "results/square_domain_cases/date(0229_1914)-model(NeoHookeanModel)-load(biaxial)-mesh(159x159)-dims(1x1)-flaws(2)-exx(1)-eyy(1)-reg(0.5)-dist(0.333)-inc(0.02)-step(0.1)/functions/p000018_000450.npy"
filename = "results/square_domain_cases/date(0229_2019)-model(NeoHookeanModel)-load(biaxial)-mesh(159x159)-dims(1x1)-flaws(2)-exx(1.5)-eyy(1.5)-reg(0.5)-dist(0.333)-inc(0.02)-step(0.1)/functions/p000021_000525.npy"

if 'filename' not in globals():
    raise RuntimeError("Undefined variable `filename`. Require `filename` to be "
                       "a string that points to `numpy` `.npy` file. The file "
                       "must contain the phasefield degrees of freedom.")
elif not isinstance(filename, str):
    raise ValueError("Require `filename` to be a string that points to a `numpy` `.npy` "
                     "file. The file must contain the phasefield degrees of freedom.")
if not os.path.isfile(filename):
    raise ValueError(f"Could not find file: \"{filename}\".\nRequire `filename` "
                     "to be a string that points to a `.npy` file. The file must "
                     "contains the phasefield degrees of freedom.")


### Problem parameters

num_unticells_x = 2
num_unitcells_y = 2
unitcell_mirror_x = True
unitcell_mirror_y = True
unitcell_overhang_fraction = 0.0

number_of_loading_steps = 25
minimum_material_integrity = 1e-4
material_integrity_exponent = 2

material_parameters = {
    'E': dolfin.Constant(1.0),
    'nu': dolfin.Constant(0.0),
    }

element_degree = 1
element_family = "CG"
mesh_diagonal = "crossed"

# maximum_elements = 160**2
maximum_elements = 240**2
# maximum_elements = 320**2


### Load unitcell solution

unitcell_nx, unitcell_ny = [int(s) for s in utility.extract_substring(
                            filename, str_beg="mesh(", str_end=")").split("x")]

unitcell_L, unitcell_H = [float(s) for s in utility.extract_substring(
                          filename, str_beg="dims(", str_end=")").split("x")]

load_mode    = utility.extract_substring(filename, "load(", ")")
model_name   = utility.extract_substring(filename, "model(", ")")
unitcell_exx = float(utility.extract_substring(filename, "exx(", ")"))
unitcell_eyy = float(utility.extract_substring(filename, "eyy(", ")"))


if OVERRIDE_LOAD_MODE is not None:
    load_mode = OVERRIDE_LOAD_MODE

if OVERRIDE_MODEL_NAME is not None:
    model_name = OVERRIDE_MODEL_NAME

if OVERRIDE_UNITCELL_EXX is not None:
    unitcell_exx = OVERRIDE_UNITCELL_EXX

if OVERRIDE_UNITCELL_EYY is not None:
    unitcell_eyy = OVERRIDE_UNITCELL_EYY


unitcell_p0 = [0,0]
unitcell_p1 = [unitcell_L, unitcell_H]

mesh_unitcell = utility.rectangle_mesh(
    unitcell_p0, unitcell_p1, unitcell_nx, unitcell_ny, mesh_diagonal)

V_p_unitcell = V_m_unitcell = dolfin.FunctionSpace(
    mesh_unitcell, element_family, element_degree)

p_unitcell = dolfin.Function(V_p_unitcell)
p_unitcell.vector()[:] = np.load(filename)

m_unitcell = dolfin.project(
    material_integrity(p_unitcell, minimum_material_integrity, material_integrity_exponent),
    V_m_unitcell)

optim.filter.trimoff_function_values(m_unitcell, minimum_material_integrity, 1.0)

p_unitcell.rename("p_unitcell", '')
m_unitcell.rename("m_unitcell", '')


### Tile the unitcell solution to obtain the periodic solution

# domain_L = unitcell_L
# domain_H = unitcell_H

domain_L = unitcell_L * num_unticells_x
domain_H = unitcell_H * num_unitcells_y

domain_p0 = [0,0]
domain_p1 = [domain_L, domain_H]

domain_nx = round(math.sqrt(maximum_elements * (domain_L / domain_H)))
domain_ny = round(math.sqrt(maximum_elements * (domain_H / domain_L)))

mesh = utility.rectangle_mesh(domain_p0, domain_p1, domain_nx, domain_ny, mesh_diagonal)

V_p = V_m = dolfin.FunctionSpace(mesh, element_family, element_degree)

p = utility.project_function_periodically(
    p_unitcell, num_unticells_x, num_unitcells_y, V_p,
    unitcell_mirror_x, unitcell_mirror_y, unitcell_overhang_fraction)

optim.filter.trimoff_function_values(p, 0.0, 1.0)
optim.filter.rescale_function_values(p, 0.0, 1.0)

m = dolfin.project(
    material_integrity(p, minimum_material_integrity, material_integrity_exponent),
    V_m)

optim.filter.trimoff_function_values(m, minimum_material_integrity, 1.0)

p.rename("p", '')
m.rename("m", '')


### Displacement problem

V_u = dolfin.VectorFunctionSpace(mesh, 'CG', 1)

u = dolfin.Function(V_u)
u.rename("u", '')

bcs, bcs_set_values = utility.uniform_extension_bcs(V_u, load_mode)

if unitcell_exx:
    ratio_eyy_to_exx = unitcell_eyy / unitcell_exx
    bcs_values = np.array([(domain_L*exx, domain_H*ratio_eyy_to_exx*exx)
        for exx in np.linspace(0.0, unitcell_exx, number_of_loading_steps)])

else: # Vertical extension case
    bcs_values = np.array([(0.0, domain_H*eyy) for eyy in
        np.linspace(0.0, unitcell_eyy, number_of_loading_steps)])

if model_name == "LinearElasticModel":
    material_model = material.LinearElasticModel(material_parameters, u)
elif model_name == "NeoHookeanModel":
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

if model_name == "LinearElasticModel":
    equilibrium_solve(incremental=False)
else:
    equilibrium_solve(incremental=True)


### Post-process

V_tensor = dolfin.TensorFunctionSpace(mesh, 'DG', 0)

green_strain = None
stress_field = None

if COMPUTE_STRAIN:

    # Threshold for Green-Lagrange strain
    w_thr = dolfin.Function(V_m); w_thr.vector()[:] = 1.0
    w_thr.vector()[np.flatnonzero(m.vector() < 0.5)] = 0.0

    E_thr = w_thr*material_model.deformation_measures.E
    green_strain = dolfin.project(E_thr, V_tensor)
    green_strain.rename("Green-Lagrange strain", '')

if COMPUTE_STRESS:

    stress_field = dolfin.project(pk2, V_tensor)
    stress_field.rename("stress (PK2)", '')

    maximal_compressive_stress_field = \
        utility.compute_maximal_compressive_stress_field(stress_field)

    fraction_compressive_stress_field = \
        utility.compute_fraction_compressive_stress_field(stress_field)

    maximal_compressive_stress_field.rename('maximal_compression', '')
    fraction_compressive_stress_field.rename('fraction_compression', '')

def save_functions():

    results_outdir = os.path.split(os.path.dirname(filename))[0]
    results_outdir_functions = os.path.join(results_outdir, "functions_periodic")

    dolfin.File(os.path.join(results_outdir_functions, "periodic_u.pvd")) << u
    dolfin.File(os.path.join(results_outdir_functions, "periodic_m.pvd")) << m
    dolfin.File(os.path.join(results_outdir_functions, "periodic_p.pvd")) << p

    dolfin.File(os.path.join(results_outdir_functions, "unitcell_m.pvd")) << m_unitcell
    dolfin.File(os.path.join(results_outdir_functions, "unitcell_p.pvd")) << p_unitcell

    if COMPUTE_STRAIN:
        dolfin.File(os.path.join(results_outdir_functions, "green_strain.pvd")) << green_strain

    if COMPUTE_STRESS:
        dolfin.File(os.path.join(results_outdir_functions, "stress_field.pvd")) << stress_field
        dolfin.File(os.path.join(results_outdir_functions,
            "maximal_compressive_stress_field.pvd")) << maximal_compressive_stress_field
        dolfin.File(os.path.join(results_outdir_functions,
            "fraction_compressive_stress_field.pvd")) << fraction_compressive_stress_field


if __name__ == "__main__":

    if SAVE_RESULTS:
        save_functions()
