
import dolfin
import numpy as np
import matplotlib.pyplot as plt

import material
import optim

from example import utility
from example.square_domain import material_integrity

plt.close('all')

# File name to the degrees of freedom of the material phase solution
# filename = "/Users/danas.sutula/Documents/Programs/Python/compliance_maximization/results_ktt/square_domain/date(0915_1952)-load(biaxial_uniform)-model(LinearElasticModel)-mesh(52165)-count(4)-disp(0d100)-regul(0d475)-inc(0d005)-step(0d025)/functions/material_fraction_dofs.npy"
# filename = "/Users/danas.sutula/Documents/Programs/Python/compliance_maximization/results_ktt/square_domain/date(0915_2115)-load(biaxial_uniform)-model(LinearElasticModel)-mesh(52165)-count(4)-disp(0d100)-regul(0d475)-inc(0d005)-step(0d025)/functions/material_fraction_dofs.npy"
# filename = "/Users/danas.sutula/Documents/Programs/Python/compliance_maximization/results/square_domain/date(0916_1816)-load(biaxial_uniform)-model(LinearElasticModel)-mesh(3281)-count(1)-disp(0d100)-regul(0d500)-inc(0d005)-step(0d025)/functions/material_fraction.xml"
# filename = "/Users/danas.sutula/Documents/Programs/Python/compliance_maximization/results/square_domain/date(0916_1922)-load(biaxial_uniform)-model(LinearElasticModel)-mesh(3281)-count(2)-disp(0d100)-regul(0d500)-inc(0d005)-step(0d025)/functions/material_fraction.xml"

# filename = "/Users/danas.sutula/Documents/Programs/Python/compliance_maximization/results/square_domain/date(0916_2103)-load(biaxial_uniform)-model(LinearElasticModel)-mesh(3281)-count(4)-disp(0d100)-regul(0d450)-inc(0d005)-step(0d025)/functions/material_fraction.xml"
# filename = "/Users/danas.sutula/Documents/Programs/Python/compliance_maximization/results/square_domain/date(0916_2142)-load(biaxial_uniform)-model(LinearElasticModel)-mesh(3281)-count(4)-disp(0d100)-regul(0d475)-inc(0d005)-step(0d025)/functions/material_fraction.xml"
# filename = "/Users/danas.sutula/Documents/Programs/Python/compliance_maximization/results/square_domain/date(0916_2200)-load(biaxial_uniform)-model(LinearElasticModel)-mesh(3445)-count(8)-disp(0d100)-regul(0d475)-inc(0d005)-step(0d025)/functions/material_fraction.xml"
# filename = "/Users/danas.sutula/Documents/Programs/Python/compliance_maximization/results/square_domain/date(0916_2209)-load(biaxial_uniform)-model(LinearElasticModel)-mesh(3445)-count(5)-disp(0d100)-regul(0d475)-inc(0d005)-step(0d025)/functions/material_fraction.xml"
# filename = "/Users/danas.sutula/Documents/Programs/Python/compliance_maximization/results/square_domain/date(0916_2225)-load(biaxial_uniform)-model(LinearElasticModel)-mesh(3445)-count(2)-disp(0d100)-regul(0d475)-inc(0d005)-step(0d025)/functions/material_fraction.xml"
# filename = "/Users/danas.sutula/Documents/Programs/Python/compliance_maximization/results/square_domain/date(0917_0826)-load(biaxial_uniform)-model(LinearElasticModel)-mesh(3445)-count(3)-disp(0d100)-regul(0d475)-inc(0d005)-step(0d025)/functions/material_fraction.xml"

# filename = "/Users/danas.sutula/Documents/Programs/Python/compliance_maximization/results/square_domain/date(0917_0914)-load(biaxial_uniform)-model(LinearElasticModel)-mesh(3445)-count(2)-disp(0d100)-regul(0d475)-inc(0d005)-step(0d025)/functions/p000038.npy"
# filename = "/Users/danas.sutula/Documents/Programs/Python/compliance_maximization/results/square_domain/date(0917_1005)-load(biaxial_uniform)-model(LinearElasticModel)-mesh(3445)-count(4)-disp(0d100)-regul(0d475)-inc(0d005)-step(0d025)/functions/p000008.npy"
# filename = "/Users/danas.sutula/Documents/Programs/Python/compliance_maximization/results/square_domain/date(0917_1017)-load(biaxial_uniform)-model(LinearElasticModel)-mesh(3445)-count(4)-disp(0d100)-regul(0d475)-inc(0d005)-step(0d025)/functions/p000018.npy"
# filename = "/Users/danas.sutula/Documents/Programs/Python/compliance_maximization/results/square_domain/date(0917_1212)-load(biaxial_uniform)-model(LinearElasticModel)-mesh(7565)-count(4)-disp(0d100)-regul(0d475)-inc(0d003)-step(0d025)/functions/p000038.npy"

# filename = "/Users/danas.sutula/Documents/Programs/Python/compliance_maximization/results/square_domain/date(0917_1212)-load(biaxial_uniform)-model(LinearElasticModel)-mesh(7565)-count(4)-disp(0d100)-regul(0d475)-inc(0d003)-step(0d025)/functions/p000038.npy"
# filename = "/Users/danas.sutula/Documents/Programs/Python/compliance_maximization/results_ktt/square_domain/date(0917_1733)-load(biaxial_uniform)-model(LinearElasticModel)-mesh(7565)-count(2)-disp(0d100)-regul(0d475)-inc(0d003)-step(0d025)/functions/p000082.npy"

# filename = "/Users/danas.sutula/Documents/Programs/Python/compliance_maximization/results_ktt/square_domain/date(0919_0851)-load(biaxial_uniform)-model(LinearElasticModel)-mesh(52165)-count(2)-disp(0d100)-regul(0d475)-inc(0d002)-step(0d025)/functions/p000046.npy"

# filename = "/Users/danas.sutula/Documents/Programs/Python/compliance_maximization/results_ktt/square_domain/date(0919_1438)-load(biaxial_uniform)-model(LinearElasticModel)-mesh(52165)-count(2)-disp(0d100)-regul(0d475)-inc(0d002)-step(0d025)/functions/p000047.npy"
# filename = "results_ktt/square_domain/date(0920_0742)-load(biaxial_uniform)-model(LinearElasticModel)-mesh(52165)-count(2)-disp(0d100)-regul(0d475)-inc(0d002)-step(0d025)/functions/p000033.npy"

# filename = "results_ktt/square_domain/date(1001_1800)-model(LinearElasticModel)-mesh(81x162)-defects(2)-exx(0d100)-eyy(0d050)-regul(0d485)-inc(0d010)-step(0d010)/functions/p000090.npy"

filename = "results/square_domain/date(1002_1313)-model(LinearElasticModel)-mesh(81x81)-dims(1x1)-flaws(2)-exx(0.1)-eyy(0.05)-reg(0.475)-inc(0.01)-step(0.01)/functions/p000044.npy"

require_displacement_solution = True
using_material_density_instead_of_phasefield = True

num_elements_x, num_elements_y = \
    [int(s) for s in utility.extract_substring(
     filename, str_beg="mesh(", str_end=")").split("x")]

unitcell_L, unitcell_H = \
    [float(s) for s in utility.extract_substring(
     filename, str_beg="dims(", str_end=")").split("x")]

exx_unitcell = float(utility.extract_substring(filename, "exx(", ")"))
eyy_unitcell = float(utility.extract_substring(filename, "eyy(", ")"))

unitcell_p0, unitcell_p1 = [0,0], [unitcell_L,unitcell_H]

element_degree = 1
element_family = "CG"
mesh_diagonal = "crossed"

minimum_material_integrity = 1e-5

mesh_unitcell = utility.rectangle_mesh(
    unitcell_p0, unitcell_p1, num_elements_x, num_elements_y, mesh_diagonal)

V_unitcell = dolfin.FunctionSpace(
    mesh_unitcell, element_family, element_degree)

p_unitcell = dolfin.Function(V_unitcell)
p_unitcell.vector()[:] = np.load(filename)

if using_material_density_instead_of_phasefield:
    m_unitcell = dolfin.project(material_integrity(
        p_unitcell, minimum_material_integrity), V_unitcell)
else:
    m_unitcell = p_unitcell


num_unticells_x = 4
num_unitcells_y = 4

mirror_x = True
mirror_y = True

overhang_fraction = 0.0

extension_strain_final = 0.20
extension_strain_initial = 0.20
extension_strain_stepsize = 0.20
strain_stepsize_refinements = 2

material_parameters = {
    'E': dolfin.Constant(1.0),
    'nu': dolfin.Constant(0.4)
    }

num_elements_x = 250
num_elements_y = num_elements_x

domain_L = 1.0
domain_H = 1.0

domain_p0, domain_p1 = [0,0], [domain_L,domain_H]

mesh = utility.rectangle_mesh(
    domain_p0, domain_p1, num_elements_x, num_elements_y, mesh_diagonal)

V_m = dolfin.FunctionSpace(mesh, element_family, element_degree)

m = utility.project_function_periodically(
    m_unitcell, num_unticells_x, num_unitcells_y,
    V_m, mirror_x, mirror_y, overhang_fraction)

optim.filter.apply_interval_bounds(m,
    lower=minimum_material_integrity, upper=1.0)

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
        eyy = eyy_unitcell / exx_unitcell * exx

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


if __name__ == "__main__":

    if require_displacement_solution:

        solve_displacement_problem_incrementally(
            extension_strain_initial, extension_strain_final,
            extension_strain_stepsize, strain_stepsize_refinements)

        # dolfin.File("temp_u.pvd") << u
        # dolfin.File("temp_m.pvd") << m

    # dolfin.File("m_unitcell.pvd") << m_unitcell
    #

    plt.figure("Phasefield (unit-cell)")
    dolfin.plot(m_unitcell)

    plt.figure("Phasefield (periodic)")
    dolfin.plot(m)

    plt.show()
