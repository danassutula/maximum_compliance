'''
K.get(a, [0],[0,1,2])
K.set(np.array([[10,20,30]]), [0],[0,1,2])
K.get_diagonal(u.vector())
K.set_diagonal(u.vector())
K.apply('add')

NOTE: Defect energy will increase only if there is some initial defect.

Todo:

* How to enforce a more meaningful constrain? How to better relate the penalty
    parameter dependent on the free energy?

* Solve for the initial phasefield soluton.

* Phase field filtering
* Remove zero energy modes

FIXME:
* The regularization is mesh dependent.

AIMING FOR:
Thin defects:
    penalize phasefield to get smaller defects,
    penalize gradient to avoid islands (?)

'''

# Import first
import config

import os
import math
import time

import dolfin
import logging
import numpy as np
import scipy.linalg as linalg
import matplotlib.pyplot as plt

from dolfin import *
import optimization
import material

logger = logging.getLogger()
logger.setLevel(logging.INFO)

import importlib # TEMP
importlib.reload(optimization.config) # TEMP
importlib.reload(optimization.optim) # TEMP
importlib.reload(optimization) # TEMP

def plot(*args, **kwargs):
    plt.figure(kwargs.pop('name', None))
    dolfin.plot(*args, **kwargs)
    plt.show()

# NOTE: It's probably important to solve the phase-field equation for time 0
# If the phase-field equation is not solved, the surface energy could locally
# decrease where the phase-field would increase. This is not physical.


def M(p):
    '''Double well function. Penalty for the transition zone (p = 0.5).'''
    return (1-dolfin.cos(2*DOLFIN_PI * p))/2


def damage_density(p):
    '''Defect density in terms of the phase-field.'''
    # p**2 is not great because it allows for negligable
    return p


def phasefield_regularization(p):
    return 0.5*dot(grad(p),grad(p))


def material_integrity(p, g0, d=2):
    '''Material integrity from the phase-field.

    Important
    ---------
    The derivative of the material integrity with respect to the phasefield
    function must be negative. This guarantees that an increase in the
    phasefield results in a decrease in the material integrity.

    '''
    if not (isinstance(p, dolfin.Expression) or
            isinstance(p, dolfin.Function) or
            isinstance(p, np.ndarray)):
        raise TypeError

    if d <= 1:
        # Must forbid energy dissipation for a fully damaged material
        raise ValueError('Require d > 1')

    return g0 + (1.0-g0) * (1.0-p) ** d


def insert_defect(p, xc, r, rtol):

    V_p = p.function_space()
    hmax = V_p.mesh().hmax()
    atol = hmax * rtol

    x = V_p.tabulate_dof_coordinates()
    s = (((x-xc)/r)**2).sum(axis=1)

    p_arr = p.vector().get_local()
    p_arr[s < 1.0 + atol] = 1.0

    p.vector().set_local(p_arr)


def insert_defect_array(xlim, ylim, n, m, p, r, rtol):

    x, y = np.meshgrid(
        np.linspace(xlim[0], xlim[1], n),
        np.linspace(ylim[0], ylim[1], m)
        )

    x = x.reshape((-1,))
    y = y.reshape((-1,))

    for xc in np.stack([x,y], axis=1):
        insert_defect(p, xc, r, rtol)


def set_target_phasefield_fraction(t):
    p_mean.assign(phasefield_fraction_min * (1-t) + phasefield_fraction_max * t)


### Model Parameters

# defect_energy_density  = Gc = Constant(0.05) # Defect size inversely correlated with Gc
double_well_parameter = Mc = Constant(1.0)
damage_energy_density = Gc = Constant(1.0)
regularization_length = k0 = Constant(0.02) # Defect diffusivity correlates with k0
material_integrity_min = g0 = Constant(1e-3)
target_phasefield = p_mean = Constant(0.1)
stress_penalty_parameter = Constant(1e-2)

material_parameters = {
    'E': Constant(100.0),
    'nu': Constant(0.3)
    }

target_extension_mangitude = np.linspace(0, 0.2, 2)[1:]
target_phasefield_fractions = np.linspace(0.05, 0.1, 5)


### Discretization

element_u = {'scheme': 'CG', 'degree': 1}
element_p = {'scheme': 'CG', 'degree': 1}

# mesh_domain = UnitSquareMesh(150, 150, diagonal="left")
mesh_domain = UnitSquareMesh(250, 250, diagonal="left/right")

coord = mesh_domain.coordinates()
L_max = coord.max()-coord.min()

rtol = 1e-12
atol = rtol * L_max
atol2 = atol ** 2

V_u = VectorFunctionSpace(mesh_domain, element_u['scheme'], element_u['degree'])
V_p = FunctionSpace(mesh_domain, element_p['scheme'], element_p['degree'])

# uD_bot = Expression((' 0.0*s','-1.0*s'), s=0, degree=0)
# uD_top = Expression((' 0.0*s',' 1.0*s'), s=0, degree=0)
#
# uD_lhs = Expression(('-1.0*s',' 0.0*s'), s=0, degree=0)
# uD_rhs = Expression((' 1.0*s',' 0.0*s'), s=0, degree=0)

def scale_boundary_displacements(s):
    '''To be called inside solver loop.'''
    # uD_bot.s = uD_top.s = uD_lhs.s = uD_rhs.s = s
    uxD.ux = s
    uyD.uy = s

def boundary_bot(x, on_boundary):
    return on_boundary and x[1] < atol

def boundary_top(x, on_boundary):
    return on_boundary and x[1] + atol > 1

def boundary_lhs(x, on_boundary):
    return on_boundary and x[0] < atol

def boundary_rhs(x, on_boundary):
    return on_boundary and x[0] + atol > 1

# bcs_u = [DirichletBC(V_u, uD_bot, boundary_bot),
#          DirichletBC(V_u, uD_top, boundary_top),
#          DirichletBC(V_u, uD_lhs, boundary_lhs),
#          DirichletBC(V_u, uD_rhs, boundary_rhs)]

cppcode = "(x[0] < atol) ? (-ux) : (x[0] > L - atol ? (ux) : (0.0))"
uxD = Expression(cppcode, ux=0.0, uy=0.0, L=1.0, H=1.0, atol=atol, degree=0)

cppcode = "(x[1] < atol) ? (-uy) : (x[1] > H - atol ? (uy) : (0.0))"
uyD = Expression(cppcode, ux=0.0, uy=0.0, L=1.0, H=1.0, atol=atol, degree=0)

V_ux, V_uy = V_u.split()

# # TEMP
# bcs_u = [DirichletBC(V_ux, uxD, 'on_boundary'),
#          DirichletBC(V_uy, uyD, 'on_boundary'),]

# TEMP
bcs_u = [DirichletBC(V_ux, uxD, boundary_lhs),
         DirichletBC(V_uy, uyD, boundary_bot),
         DirichletBC(V_ux, uxD, boundary_rhs),
         DirichletBC(V_uy, uyD, boundary_top)]

# # TEMP
# bcs_u = [DirichletBC(V_ux, Constant(0.0), boundary_top),
#          DirichletBC(V_uy, Constant(0.0), boundary_bot)]


### Setup solver

u = Function(V_u, name="displacement")
p = Function(V_p, name="phasefield") # phase-field

dd = damage_density(p)
mi = material_integrity(p, g0)
rg = phasefield_regularization(p)

material_model = material.NeoHookeanModel(u, material_parameters)

# NOTE: Undamaged material properties
psi = material_model.strain_energy_density()
pk1 = material_model.stress_measure_pk1()
pk2 = material_model.stress_measure_pk2()

N = FacetNormal(mesh_domain)
T = mi * dolfin.dot(pk1, N)

# Deformation gradient
F = dolfin.Identity(len(u)) + dolfin.grad(u)

# Green-Lagrange strain tensor
E = F.T*F - dolfin.Identity(len(u))

# Determinant of deformation gradient
J = dolfin.det(F)



# Constraint equation
constraint = (p - p_mean) * dx

# Potential energy
W_potential = mi * psi * dx

# Phasefield regularization
# W_penalty = (Gc * dd + k0 * rg + Mc* M(p)) * dx

# W_penalty = 1 * dx(mesh_domain)
# W_penalty = (J-1)**2 * dx
W_penalty = k0 * rg * dx


# High stress penalty
# W_penalty += stress_penalty_parameter * mi**2 * pk2**2 * dx

# Total energy to be minimized
W_cost = W_potential + W_penalty
# W_cost = dolfin.dot(T, u)*ds + W_penalty # fails at boundary


hist_energies_cost = []
hist_energies_penalty = []
hist_energies_potential = []
hist_constraint_equation = []

def record_iteration_state():
    hist_energies_cost.append(assemble(W_cost))
    hist_energies_penalty.append(assemble(W_penalty))
    hist_energies_potential.append(assemble(W_potential))
    hist_constraint_equation.append(assemble(constraint))

# p.vector()[p.vector() < 0.0] = 0.0
# p.vector()[p.vector() > 0.0] = 1.0

insert_defect_array([0.15, 0.85], [0.15, 0.85],
                    3, 3, p, r=0.05, rtol=1e-9)

# optimization.pde_filter(p, p, 0.0001)

optimizer = optimization.TopologyOptimizer(
    W_potential, W_cost, constraint, u, p, bcs_u, record_iteration_state)

for s in target_extension_mangitude:
    print(f'Solving for load increment {s:3.2f} ...')
    scale_boundary_displacements(s)
    optimizer.nonlinear_solver.solve()

t0 = time.time()
for p_target in target_phasefield_fractions:

    print(f'Solving for phasefield fraction {p_target:3.2f} ...')

    p_mean.assign(p_target)

    is_converged, error_number = optimizer.optimize(0.020, 0.020, 0.100)
    if not is_converged and error_number == 1: break

    is_converged, error_number = optimizer.optimize(0.010, 0.010, 0.100)
    if not is_converged and error_number == 1: break


t1 = time.time()
print('CPU TIME:', t1-t0)

assert p.vector().min() + rtol > 0.0
assert p.vector().max() - rtol < 1.0


if __name__ == "__main__":

    import os
    import matplotlib.pyplot as plt
    plt.show()

    ### Write out solutions

    # results_dir = os.path.join(os.curdir, 'Results')
    # results_file_u = os.path.join(results_dir, 'displacement.pvd')
    # results_file_p = os.path.join(results_dir, 'phasefield.pvd')
    # outfile_u = File(results_file_u)
    # outfile_p = File(results_file_p)


    ### Plot solutions

    def plot_energies():

        fh = plt.figure('energies')
        fh.clear()

        plt.plot(hist_energies_cost, '-r')
        plt.plot(hist_energies_penalty, '--b')
        plt.plot(hist_energies_potential, '-.g')

        plt.legend(['cost', 'penalty', 'potential'])

        plt.ylabel('Energy, W')
        plt.xlabel('Iteration number, #')
        plt.title('Evolution of Energy')

        fh.tight_layout()
        plt.show()


    def plot_phasefiled():

        fh = plt.figure('phasefield')
        fh.clear()

        dolfin.plot(p)

        plt.title('Phase-field, p\n(p_min = {0:.5}; p_max = {1:.5})'.format(
            p.vector().get_local().min(), p.vector().get_local().max()))

        fh.tight_layout()
        plt.show()

    def plot_constraint():

        fh = plt.figure('constraint')
        fh.clear()

        plt.plot(hist_constraint_equation, '-k')

        plt.ylabel('C')
        plt.xlabel('Iteration number, #')
        plt.title('Evolution of Constraint Residual')

        fh.tight_layout()  # otherwise the right y-label is slightly clipped
        plt.show()

    plot_energies()
    plot_phasefiled()
    plot_constraint()
