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

* Need to decompose strain energy into compression and tensionself.
* Phase field filtering
* Remove zero energy modes


FIXME:
* The regularization is mesh dependent.

AIMING FOR:
Thin defects:
    penalize phasefield to get smaller defects,
    penalize gradient to avoid islands (?)

TODO:
! Require mesh-independence. Do enough smoothing

In fracture mechnicas using pf, the length scale can be controled easily thanks
to external load control and energy exchange/balance. In topo. optimization
it is more difficul because in the miinimization there is the competition
between the strain energy dissipation and the phasefield gradient

# NOTE: It is adventageous that the equality constraint is linear!

The phasefield can break arbitrarily.

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

logger = logging.getLogger()
logger.setLevel(logging.INFO)

import optimization
import material
import utility

import importlib # TEMP
# importlib.reload(optimization.common) # TEMP
importlib.reload(optimization.config) # TEMP
importlib.reload(optimization.filter) # TEMP
importlib.reload(optimization.utility) # TEMP
importlib.reload(optimization.optim) # TEMP
importlib.reload(optimization) # TEMP
importlib.reload(utility) # TEMP


def plot(*args, **kwargs):
    '''Small wrapper around `dolfin.plot` for always raising the figure.'''
    plt.figure(kwargs.pop('name', None))
    dolfin.plot(*args, **kwargs)
    plt.show()


output_directory = "results"
output_filename_u = "tmp/u.pvd" # "u.xdmf"
output_filename_p = "tmp/p.pvd" # "p.xdmf"

output_filepath_u = os.path.join(output_directory, output_filename_u)
output_filepath_p = os.path.join(output_directory, output_filename_p)

utility.cleanup_filepath(output_filepath_u)
utility.cleanup_filepath(output_filepath_p)

outfile_u = dolfin.File(output_filepath_u)
outfile_p = dolfin.File(output_filepath_p)

# outfile_u = dolfin.XDMFFile(output_filepath_u)
# outfile_p = dolfin.XDMFFile(output_filepath_p)


def phasefield_regularization(p):
    '''Large gradient penalty term.'''
    return 0.5*dot(grad(p),grad(p))


def material_integrity(p, mi_min):
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

    EPS = 1e-12 # NOTE: Important for stability of sign
    POW = 2

    # if POW <= 1:
    #     # No energy dissipation if fully damaged
    #     raise ValueError('Require POW >= 1')

    return mi_min + (1.0-mi_min) * ((1.0+EPS)-p) ** POW


### Model Parameters

# Energy gradient diffusion constant
filter_diffusivity = Constant(5e-4 * 1)

regularization_length  = c0_phi = Constant(0.0004 * 1)
material_integrity_min = mi_min = Constant(0.1)

target_external_loading_steps = np.linspace(0, 0.1, 2)[1:]
target_phasefield_values = np.linspace(0.05, 0.10, 6)

# target_phasefield_values = np.linspace(0.001, 0.06, 100)
# target_phasefield_values = np.linspace(0.001, 0.06, 100)[::-1]

# p_mean_0 = 0.001
# p_mean_1 = 0.050
#
# delta_p = p_mean_1 - p_mean_0
#
# num_periods = 3
#
# target_phasefield_oscilation = 0.1 * delta_p * \
#     np.sin((target_phasefield_values-p_mean_0)
#            * (num_periods * 2*math.pi / abs(delta_p)))
# #
# target_phasefield_values -= target_phasefield_oscilation
#
# target_phasefield_values -= target_phasefield_values.min()
# target_phasefield_values += p_mean_0


material_parameters = {
    'E': Constant(1),
    'nu': Constant(0.3)
    }

# Initialize target phasefield
p_mean = Constant(0.0)


### Discretization

nx = ny = 100 # number of elements
mesh_pattern = "left/right"

element_u = {'scheme': 'CG', 'degree': 1}
element_p = {'scheme': 'CG', 'degree': 1}

mesh_domain = UnitSquareMesh(nx, ny, diagonal=mesh_pattern)

dx_domain = dx(mesh_domain)

coord = mesh_domain.coordinates()
L_max = coord.max()-coord.min()

rtol = 1e-12
atol = rtol * L_max

V_u = VectorFunctionSpace(mesh_domain, element_u['scheme'], element_u['degree'])
V_p = FunctionSpace(mesh_domain, element_p['scheme'], element_p['degree'])

V_ux, V_uy = V_u.split()

def boundary_bot(x, on_boundary):
    return on_boundary and x[1] < atol

def boundary_top(x, on_boundary):
    return on_boundary and x[1] + atol > 1

def boundary_lhs(x, on_boundary):
    return on_boundary and x[0] < atol

def boundary_rhs(x, on_boundary):
    return on_boundary and x[0] + atol > 1

class CornerBottomLeft(SubDomain):
  def inside(self, x, on_boundary):
    return near(x[0], 0.0, DOLFIN_EPS) and \
           near(x[1], 0.0, DOLFIN_EPS)

uxD_lhs = Expression('-s', s=0.0, degree=0)
uxD_rhs = Expression(' s', s=0.0, degree=0)
uyD_bot = Expression('-s', s=0.0, degree=0)
uyD_top = Expression(' s', s=0.0, degree=0)
uxD_lbc = Expression('-s', s=0.0, degree=0)
uyD_lbc = Expression('-s', s=0.0, degree=0)

# Biaxial extension
bcs_u = [
    DirichletBC(V_u.sub(0), uxD_lhs, boundary_lhs),
    DirichletBC(V_u.sub(0), uxD_rhs, boundary_rhs),
    DirichletBC(V_u.sub(1), uyD_bot, boundary_bot),
    DirichletBC(V_u.sub(1), uyD_top, boundary_top),
    # DirichletBC(V_u.sub(0), uxD_lbc, CornerBottomLeft(), method='pointwise'),
    # DirichletBC(V_u.sub(1), uyD_lbc, CornerBottomLeft(), method='pointwise'),
    ]

# Vertical extension

# bcs_u = [
#     # DirichletBC(V_u.sub(0), uxD_lhs, boundary_lhs),
#     # DirichletBC(V_u.sub(0), uxD_rhs, boundary_rhs),
#     DirichletBC(V_u.sub(1), uyD_bot, boundary_bot),
#     DirichletBC(V_u.sub(1), uyD_top, boundary_top),
#     DirichletBC(V_u.sub(0), Constant(0.0), CornerBottomLeft(), method='pointwise'),
#     DirichletBC(V_u.sub(1), uyD_lbc, CornerBottomLeft(), method='pointwise'),
#     ]

def scale_boundary_displacements(s):
    '''To be called inside solution loop.'''

    uxD_lhs.s = s
    uxD_rhs.s = s
    uyD_bot.s = s
    uyD_top.s = s
    uxD_lbc.s = s
    uyD_lbc.s = s


### Setup solver

u = Function(V_u, name="displacement")
p = Function(V_p, name="phasefield") # phase-field

material_model = material.NeoHookeanModel(u, material_parameters)

psi = material_model.strain_energy_density()
pk1 = material_model.stress_measure_pk1()
pk2 = material_model.stress_measure_pk2()

mi = material_integrity(p, mi_min)
phi = phasefield_regularization(p)

N = FacetNormal(mesh_domain)
T = mi * dolfin.dot(pk1, N)


class KinematicQuantities:
    def __init__(self, u):

        # Deformation gradient
        self.F = F = dolfin.Identity(len(u)) + dolfin.grad(u)

        # Green-Lagrange strain tensor
        self.E = F.T*F - dolfin.Identity(len(u))

        # Determinant of deformation gradient
        self.J = dolfin.det(F)


# Potential energy
W_potential = mi * psi * dx

# Phasefield regularization
W_phasefield = c0_phi * phi * dx


# Total energy to be minimized
W_cost = W_potential + W_phasefield

hist_energies_cost = []
hist_energies_potential = []
hist_energies_phasefield = []
hist_phasefield_fraction = []
hist_phasefield_diffusion = []

def make_recorder_function():
    k_itr = 0

    def recorder_function():
        nonlocal k_itr

        hist_energies_cost.append(assemble(W_cost))
        hist_energies_potential.append(assemble(W_potential))
        hist_energies_phasefield.append(assemble(W_phasefield))

        hist_phasefield_diffusion.append(
            math.sqrt(assemble(grad(p)**2*dx) / assemble(p**2*dx)))

        hist_phasefield_fraction.append(assemble(p*dx))

        if k_itr % 5 == 0:

            # outfile_u << u
            outfile_p << p

            # outfile_u.write(u, k_itr)
            # outfile_p.write(p, k_itr)

        k_itr += 1

    return recorder_function

recorder_function = make_recorder_function()

# utility.insert_defect_array([0.35, 0.65], [0.35, 0.65], 2, 2, p, r=1/30, rtol=1e-9)
# utility.insert_defect_array([  1/3,   2/3], [  1/3,   2/3], 2, 2, p, r=0.025, rtol=1e-9)
utility.insert_defect_array([   0,   1], [ 0.5, 0.5], 2, 1, p, r=0.025, rtol=1e-9)
utility.insert_defect_array([ 0.5, 0.5], [   0,   1], 1, 2, p, r=0.025, rtol=1e-9)


# utility.insert_defect_array([0.25, 0.75], [0.25, 0.75], 5, 5, p, r=0.025, rtol=1e-9)
# utility.insert_defect_array([0.0, 0.0], [0.5, 0.5], 1, 1, p, r=0.025, rtol=1e-9)
# utility.insert_defect_array([1.0, 1.0], [0.5, 0.5], 1, 1, p, r=0.025, rtol=1e-9)

# Apply diffusion filter to smooth out initial (sharp) phasefield
optimization.filter.apply_diffusion_filter(p, kappa=c0_phi)

# Sometimes filtering causes phasefield to be outside bounds
# In this case,

p_arr = p.vector().get_local()
p_arr[p_arr < 0.0] = 0.0
p_arr[p_arr > 1.0] = 1.0
p.vector()[:] = p_arr


optimizer = optimization.TopologyOptimizer(
    W_potential, W_phasefield, p, p_mean, u, bcs_u,
    filter_diffusivity, recorder_function)

for s in target_external_loading_steps:
    print(f'Solving for load increment {s:4.3f} ...')
    scale_boundary_displacements(s)
    optimizer.nonlinear_solver.solve()

assert p.vector().min()+rtol > 0.0
assert p.vector().max()-rtol < 1.0


stepsize = 0.2
total_iterations = 0

t0 = time.time()
for p_mean_i in target_phasefield_values:
    print(f'\n *** Solving for p_mean: {p_mean_i:4.3} *** \n')

    p_mean.assign(p_mean_i)

    n_itr, is_converged, error_reason = \
        optimizer.optimize(stepsize)

    total_iterations += n_itr

    if not is_converged:
        stepsize /= 2

else:
    print('CPU TIME:', time.time()-t0)

assert p.vector().min()+rtol > 0.0
assert p.vector().max()-rtol < 1.0

# Close output files (if can be closed)
if hasattr(outfile_u, 'close'): outfile_u.close()
if hasattr(outfile_p, 'close'): outfile_p.close()


if __name__ == "__main__":

    ### Write out solutions

    # results_dir = os.path.join(os.curdir, 'Results')
    # results_file_u = os.path.join(results_dir, 'displacement.pvd')
    # results_file_p = os.path.join(results_dir, 'phasefield.pvd')
    # outfile_u = File(results_file_u)
    # outfile_p = File(results_file_p)


    ### Plot solutions

    def plot_energy_vs_iterations():

        fh = plt.figure('iterations')
        fh.clear()

        plt.plot(hist_energies_cost, '-r')
        plt.plot(hist_energies_potential, '-.g')
        plt.plot(hist_energies_phasefield, '--b')

        plt.legend(['cost', 'potential', 'penalty'])

        plt.ylabel('Energy, W')
        plt.xlabel('Iteration number, #')
        plt.title('Evolution of Energy')

        fh.tight_layout()
        plt.show()


    def plot_ratio_energy_vs_iterations():

        fh = plt.figure('energy ratio')
        fh.clear()

        plt.plot(hist_energies_ratio, '-xk')

        plt.legend(['penalty fraction'])

        plt.ylabel('Energy ratio')
        plt.xlabel('Iteration number, #')
        plt.title('Evolution of Energy Ratio')

        fh.tight_layout()
        plt.show()


    def plot_energy_vs_phasefield():

        EPS = 1e-12

        fh = plt.figure('energies')
        # fh.clear()

        n = len(hist_phasefield_fraction)
        d = np.diff(hist_phasefield_fraction)

        ind = np.abs(d) > EPS
        ind = np.flatnonzero(ind)

        ind_first = (ind+1).tolist()
        ind_first.insert(0,0)

        ind_last = ind.tolist()
        ind_last.append(n-1)

        y_cost = np.array(hist_energies_cost)[ind_last]
        y_penalty = np.array(hist_energies_phasefield)[ind_last]
        y_potential = np.array(hist_energies_potential)[ind_last]
        x_phasefield = np.array(hist_phasefield_fraction)[ind_last]

        plt.plot(x_phasefield, y_cost, '-r')
        plt.plot(x_phasefield, y_penalty, '--b')
        plt.plot(x_phasefield, y_potential, '-.g')

        plt.legend(['cost', 'penalty', 'potential'])

        plt.ylabel('Energy, W')
        plt.xlabel('Phasefield fraction')
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

    # def plot_constraint():
    #
    #     fh = plt.figure('constraint')
    #     fh.clear()
    #
    #     plt.plot(x, hist_constraint_equation, '-k')
    #
    #     plt.ylabel('C')
    #     plt.xlabel('Iteration number, #')
    #     plt.title('Evolution of Constraint Residual')
    #
    #     fh.tight_layout()  # otherwise the right y-label is slightly clipped
    #     plt.show()


    plot_energy_vs_iterations()
    plot_energy_vs_phasefield()
    plot_phasefiled()
