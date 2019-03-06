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

importlib.reload(optimization.config) # TEMP
importlib.reload(optimization.filter) # TEMP
importlib.reload(optimization.utility) # TEMP
importlib.reload(optimization.optim) # TEMP
importlib.reload(optimization) # TEMP
importlib.reload(utility) # TEMP


def plot(*args, **kwargs):
    '''Plot either `np.ndarray`s or something plottable from `dolfin`.'''
    plt.figure(kwargs.pop('name', None))
    if isinstance(args[0], (np.ndarray,list, tuple)):
        plt.plot(*args, **kwargs)
    else: # just try it anyway
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

# NOTE
# The diffusion constant should not be too big because the filter problems is
# essentially a transient diffusion problem.

kappa_W = Constant(1e-4)
kappa_P = Constant(1e-4)
kappa_I = Constant(1e-4)
kappa_p = Constant(1e-4)

# NOTE
# The weigth in favour of penalty should not be too big as otherwise,
# the phasefield tends to dissipate out rather than concentrate
# The weight should not be too small since then the phasefield is ver sharp
# (does not spread at all)

weight_P = 0.2

# Minimal material integrity
mi_min  = Constant(1e-4)

external_loading_values = np.linspace(0, 0.01, 2)[1:]
target_phasefield_values = np.linspace(0.0, 0.200, 6)

material_parameters = {
    'E': Constant(1.0),
    'nu': Constant(0.3)
    }

# Initialize target phasefield
p_mean = Constant(0.0)


### Mesh

nx = ny = 150 # number of elements
# mesh_pattern = "left"
mesh_pattern = "left/right"

mesh = UnitSquareMesh(nx, ny, diagonal=mesh_pattern)

coord = mesh.coordinates()
L_max = coord.max()-coord.min()

rtol = 1e-12
atol = rtol * L_max


### Localize phasefield

# phasefield_subdomain = dolfin.CompiledSubDomain(
#     "pow(x[0]-x0, 2) + pow(x[1]-y0, 2) < pow(r, 2)",
#     x0=0.0, y0=0.0, r=0.0)

phasefield_subdomain = dolfin.CompiledSubDomain(
    "x0 <= x[0] && y0 <= x[1] && x1 >= x[0] && y1 >= x[1]",
    x0=0.0, y0=0.0, x1=1.0, y1=1.0)

# phasefield_subdomain = dolfin.CompiledSubDomain(
#     "std::max(std::abs(x[0]-x0),std::abs(x[1]-y0)) < r",
#     x0=0.0, y0=0.0, r=0.0)

phasefield_markers = dolfin.MeshFunction("size_t",
    mesh, dim=mesh.geometric_dimension(), value=0)

phasefield_subdomain.set_property('x0', 0.0)
phasefield_subdomain.set_property('y0', 0.0)
phasefield_subdomain.set_property('x1', 0.5)
phasefield_subdomain.set_property('y1', 0.5)
phasefield_subdomain.mark(phasefield_markers, 1)

phasefield_subdomain.set_property('x0', 0.5)
phasefield_subdomain.set_property('x1', 1.0)
phasefield_subdomain.mark(phasefield_markers, 2)

phasefield_subdomain.set_property('y0', 0.5)
phasefield_subdomain.set_property('y1', 1.0)
phasefield_subdomain.mark(phasefield_markers, 3)

phasefield_subdomain.set_property('x0', 0.0)
phasefield_subdomain.set_property('y0', 0.5)
phasefield_subdomain.set_property('x1', 0.5)
phasefield_subdomain.set_property('y1', 1.0)
phasefield_subdomain.mark(phasefield_markers, 4)

dx_sub = [dolfin.dx(subdomain_id=i, domain=mesh,
                    subdomain_data=phasefield_markers,
                    degree=None, scheme=None, rule=None)
          for i in (1,2,3,4)]

# dx_sub_0 = dolfin.dx(subdomain_id=0, domain=mesh,
#                      subdomain_data=phasefield_markers,
#                      degree=None, scheme=None, rule=None)


### Discretization

element_u = {'scheme': 'CG', 'degree': 1}
element_p = {'scheme': 'CG', 'degree': 1}

V_u = VectorFunctionSpace(mesh, element_u['scheme'], element_u['degree'])
V_p = FunctionSpace(mesh, element_p['scheme'], element_p['degree'])

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

# # Vertical extension
# bcs_u = [
#     # DirichletBC(V_u.sub(0), uxD_lhs, boundary_lhs),
#     # DirichletBC(V_u.sub(0), uxD_rhs, boundary_rhs),
#     DirichletBC(V_u.sub(1), uyD_bot, boundary_bot),
#     DirichletBC(V_u.sub(1), uyD_top, boundary_top),
#     DirichletBC(V_u.sub(0), Constant(0.0), boundary_bot),
#     DirichletBC(V_u.sub(0), Constant(0.0), boundary_top),
#     # DirichletBC(V_u.sub(0), Constant(0.0), CornerBottomLeft(), method='pointwise'),
#     # DirichletBC(V_u.sub(1), uyD_lbc, CornerBottomLeft(), method='pointwise'),
#     ]

# # Shear
# bcs_u = [
#     DirichletBC(V_u.sub(1), Constant(0.0), boundary_top),
#     DirichletBC(V_u.sub(1), Constant(0.0), boundary_bot),
#     DirichletBC(V_u.sub(0), uxD_rhs, boundary_top),
#     DirichletBC(V_u.sub(0), uxD_lhs, boundary_bot),
#     # DirichletBC(V_u.sub(0), Constant(0.0), CornerBottomLeft(), method='pointwise'),
#     # DirichletBC(V_u.sub(1), uyD_lbc, CornerBottomLeft(), method='pointwise'),
#     ]

def scale_boundary_displacements(s):
    '''To be called inside solution for-loop.'''

    uxD_lhs.s = s
    uxD_rhs.s = s
    uyD_bot.s = s
    uyD_top.s = s
    uxD_lbc.s = s
    uyD_lbc.s = s


### Setup solver

u = Function(V_u, name="displacement")
p = Function(V_p, name="phasefield") # phase-field

material_model = material.NeoHookeanModel(material_parameters, u)

psi = material_model.strain_energy_density()
pk1 = material_model.stress_measure_pk1()
pk2 = material_model.stress_measure_pk2()

mi = material_integrity(p, mi_min)
phi = phasefield_regularization(p)

N = FacetNormal(mesh)
T = mi * dolfin.dot(pk1, N)


# Potential energy
potential = mi * psi * dx

# Phasefield regularization
penalty = phi * dx

# Total energy to be minimized
# cost = potential

# Material fraction constraint(s)
constraint = (p - p_mean) * dx

# constraint = [(p - p_mean) * dx_sub_i for dx_sub_i in dx_sub]


hist_cost       = []
hist_penalty    = []
hist_potential  = []
hist_phasefield = []

def make_recorder_function():
    k_itr = 0

    recorder_checkpoints = []

    def recorder_function(insist=False):
        nonlocal k_itr

        # hist_cost.append(assemble(cost))
        hist_penalty.append(assemble(penalty))
        hist_potential.append(assemble(potential))
        hist_phasefield.append(float(p_mean.values()))

        if k_itr % 5 == 0 or insist:

            # outfile_u << u
            outfile_p << p

            # outfile_u.write(u, k_itr)
            # outfile_p.write(p, k_itr)

            recorder_checkpoints.append(k_itr)

        k_itr += 1

    return recorder_function, recorder_checkpoints

recorder_function, recorder_checkpoints = make_recorder_function()


### Insert initial defects


# ps = utility.insert_defect_array(
#     [ 0, 1], [   0,   1], 4, 4, p, r=0.025, rtol=1e-9)
# ps = utility.insert_defect_array_with_checker_pattern(
#     [ 0, 1], [   0,   1], 2, 2, p, r=0.025, rtol=1e-9)

# ps = utility.insert_defect_array(
#     [ 0, 1], [   0,   1], 4, 4, p, r=0.025, rtol=1e-9)
ps = utility.insert_defect_array_with_checker_pattern(
    [ 0, 1], [ 0, 1], 4, 4, V_p, r=0.025, rtol=1e-9)

# Apply diffusion filter to smooth out initial (sharp) phasefield
for p_i in ps:
    optimization.filter.apply_diffusion_filter(p_i, kappa_p)

    p_arr_i = p_i.vector().get_local()
    p_arr_i[p_arr_i < 0.0] = 0.0
    p_arr_i[p_arr_i > 1.0] = 1.0
    p_i.vector()[:] = p_arr_i


p.assign(sum(ps))


### Define phasefield optimizer

optimizer = optimization.TopologyOptimizer(
    potential, penalty, constraint, p, ps, u, bcs_u,
    weight_P, kappa_W, kappa_P, kappa_I, recorder_function)


for s in external_loading_values:
    print(f'Solving for load increment {s:4.3f} ...')
    scale_boundary_displacements(s)
    optimizer.nonlinear_solver.solve()

assert p.vector().min()+rtol > 0.0
assert p.vector().max()-rtol < 1.0

phasefield_stepsize = 0.050
solver_iterations_count = 0

target_phasefield_initial = assemble(p*dx) / assemble(1*dx(mesh))
target_phasefield_values += target_phasefield_initial
target_phasefield_final = target_phasefield_values[-1] * 0.80


t0 = time.time()
for p_mean_i in target_phasefield_values:
    print(f'\n *** Solving for p_mean: {p_mean_i:4.3} *** \n')

    p_mean.assign(p_mean_i)

    n_itr, is_converged, error_reason = \
        optimizer.optimize(phasefield_stepsize)

    solver_iterations_count += n_itr

    if not is_converged:
        phasefield_stepsize /= 2

else:

    # p_mean.assign(target_phasefield_final)
    # optimizer.optimize(phasefield_stepsize)

    recorder_function(insist=True)
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

        fh = plt.figure('energy_vs_iterations')
        fh.clear()

        # plt.plot(hist_cost, '-r')
        plt.plot(hist_potential, '-.g')


        plt.legend(['total', 'strain', 'penalty'])

        plt.ylabel('Energy')
        plt.xlabel('Iteration number')
        plt.title('Energy vs. Iterations')

        fh.tight_layout()
        plt.show()

    def plot_energy_vs_phasefield():

        EPS = 1e-12

        fh = plt.figure('energy_vs_phasefield')
        # fh.clear()

        n = len(hist_phasefield)
        d = np.diff(hist_phasefield)

        ind = np.abs(d) > EPS
        ind = np.flatnonzero(ind)

        ind_first = (ind+1).tolist()
        ind_first.insert(0,0)

        ind_last = ind.tolist()
        ind_last.append(n-1)

        # y_cost = np.array(hist_cost)[ind_last]
        y_penalty = np.array(hist_penalty)[ind_last]
        y_potential = np.array(hist_potential)[ind_last]
        x_phasefield = np.array(hist_phasefield)[ind_last]

        # plt.plot(x_phasefield, y_cost, '-r')
        plt.plot(x_phasefield, y_potential, '-.g')
        plt.plot(x_phasefield, y_penalty, '--b')

        plt.legend(['total', 'strain', 'penalty'])

        plt.ylabel('Energy')
        plt.xlabel('Phasefield fraction')
        plt.title('Energy vs. Phasefield')

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
