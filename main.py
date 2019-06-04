# -*- coding: utf-8 -*-
"""
Created on 01/10/2018

@author: Danas Sutula

"""

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

import optim
import filters
import material
import utility

logger = logging.getLogger()
logger.setLevel(logging.INFO)


### Results output

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


### Model parameters

material_parameters = {
    'E': Constant(1.0),
    'nu': Constant(0.3)
    }

external_loading_values = np.linspace(0, 0.01, 2)[1:]
target_phasefield_values = np.linspace(0.0, 0.150, 10)

# Minimum material integrity
m_inf  = Constant(1e-5)


### Discretization parameters

nx = ny = 100 # number of elements

mesh_pattern = "left/right"
# mesh_pattern = "crossed"
# mesh_pattern = "left"

phasefield_stepsize = 0.1
# phasefield_stepsize = 0.05
# phasefield_stepsize = 0.025

regularization = 0.30 # Phasefield advance direction is determined by weighting
                      # the strain energy gradient and the penalty-like energy
                      # gradient. Parameter `regularization` weights the penalty
                      # whereas `(1-regularization)` weights the strain energy.

phasefield_kappa = 1e-3 # Diffusion-like parameter. Individual phasefields are
                        # diffused in order to detect forthcoming collisions.
                        # The diffused phasefield overlap serves as a measure
                        # of the closeness of phasefields.

optim.config.parameters_topology_solver['maximum_iterations']    = 500
optim.config.parameters_topology_solver['maximum_divergences']   = 3
optim.config.parameters_topology_solver['collision_threshold']   = 5e-2
optim.config.parameters_topology_solver['convergence_tolerance'] = 1e-4

# NOTE: `collision_threshold` affects the phasefield collision detection.
# Smaller value makes collision detection more sensitive. The parameter is
# closelly related to parameter 'phasefield_kappa'.


### Mesh

mesh = UnitSquareMesh(nx, ny, diagonal=mesh_pattern)



# Boundary subdomains

coord = mesh.coordinates()
L_max = coord.max()-coord.min()

rtol = 1e-12
atol = rtol * L_max

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
        return near(x[0], 0.0, DOLFIN_EPS) and near(x[1], 0.0, DOLFIN_EPS)

class PeriodicBoundary(SubDomain):

    # Left boundary is "target domain" G
    def inside(self, x, on_boundary):
        return x[0] < DOLFIN_EPS and x[0] > -DOLFIN_EPS and on_boundary

    # Map right boundary (H) to left boundary (G)
    def map(self, x, y):
        y[0] = x[0] - 1.0
        y[1] = x[1]


### Discretization

V_u = VectorFunctionSpace(mesh, 'CG', 1) # , constrained_domain=PeriodicBoundary()
V_p = FunctionSpace(mesh, 'CG', 1) # , constrained_domain=PeriodicBoundary()

V_ux, V_uy = V_u.split()

u = Function(V_u, name="displacement")
p = Function(V_p, name="phasefield")

# Initialize target mean phasefield
p_mean = Constant(0.0)


# Boundary conditions

uxD_lhs = Expression('-s', s=0.0, degree=0)
uxD_rhs = Expression(' s', s=0.0, degree=0)
uyD_bot = Expression('-s', s=0.0, degree=0)
uyD_top = Expression(' s', s=0.0, degree=0)
uxD_lbc = Expression('-s', s=0.0, degree=0)
uyD_lbc = Expression('-s', s=0.0, degree=0)

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


### Hyperelastic model

def phasefield_regularization(p):
    '''Penalty for large phasefield gradients.'''
    return 0.5*dot(grad(p), grad(p))

def material_integrity(p, m_inf, q=2):
    '''Material integrity from the phase-field.'''
    EPS = 1e-12 # NOTE: Important for stability of sign
    return m_inf + (1.0-m_inf) * ((1.0+EPS)-p) ** q

material_model = material.NeoHookeanModel(material_parameters, u)

psi = material_model.strain_energy_density()
pk1 = material_model.stress_measure_pk1()
pk2 = material_model.stress_measure_pk2()

m = material_integrity(p, m_inf)
phi = phasefield_regularization(p)

# N = FacetNormal(mesh)
# T = m * dolfin.dot(pk1, N)

# Potential energy
potential = m * psi * dx

# Phasefield regularization
penalty = phi * dx

# Material fraction constraint(s)
constraint = (p - p_mean) * dx


### Results recording function

hist_penalty    = []
hist_potential  = []
hist_phasefield = []

def _record_results_1():
    hist_penalty.append(assemble(penalty))
    hist_potential.append(assemble(potential))
    hist_phasefield.append(float(p_mean.values()))

def _record_results_2():
    # outfile_u << u
    outfile_p << p
    # outfile_u.write(u, index)
    # outfile_p.write(p, index)

recording_period_results_1 = 1
recording_period_results_2 = 5

record_index = 0

def state_recording_function(insist=False):

    global record_index

    if not record_index % recording_period_results_1 or insist:
        _record_results_1()
    if not record_index % recording_period_results_2 or insist:
        _record_results_2()

    record_index += 1


### Initial phasefields

ps = [Function(V_p) for _ in range(2)]
utility.insert_defect(ps[0], [0.3, 0.5], r=0.05)
utility.insert_defect(ps[1], [0.7, 0.5], r=0.05)

# ps = utility.insert_defect_array(
#     [ 0, 1], [   0,   1], 4, 4, p, r=0.025, rtol=1e-9)
# ps = utility.insert_defect_array_with_checker_pattern(
#     [ 0, 1], [   0,   1], 2, 2, p, r=0.025, rtol=1e-9)

# ps = utility.insert_defect_array(
#     [ 0, 1], [   0,   1], 4, 4, p, r=0.025, rtol=1e-9)

# ps = utility.insert_defect_array_with_checker_pattern(
#     [ 0, 1], [ 0, 1], 4, 4, V_p, r=0.025, rtol=1e-9)

# Smooth initial (sharp) phasefields
for p_i in ps:
    filters.apply_diffusion_filter(p_i, phasefield_kappa)
    p_arr_i = p_i.vector().get_local()
    p_arr_i[p_arr_i < 0.0] = 0.0
    p_arr_i[p_arr_i > 1.0] = 1.0
    p_i.vector()[:] = p_arr_i

# Superpose local phasefields
p.assign(sum(ps))


### Define phasefield optimization problem

target_phasefield_initial = assemble(p*dx) / assemble(1*dx(mesh))
target_phasefield_values += target_phasefield_initial
target_phasefield_final = target_phasefield_values[-1] * 0.75

optimizer = optim.TopologyOptimizer(
    potential, penalty, constraint, p, ps, u, bcs_u,
    phasefield_kappa, state_recording_function)

for s_i in external_loading_values:
    scale_boundary_displacements(s_i)
    optimizer.nonlinear_solver.solve()


### Solve phasefield optimization problem

solver_iterations_count = 0

t0 = time.time()

for p_mean_i in target_phasefield_values:

    print(f'\n *** Solving for p_mean: {p_mean_i:4.3} *** \n')

    p_mean.assign(p_mean_i)

    n_itr, is_converged, error_reason = optimizer \
        .optimize(phasefield_stepsize, regularization)

    solver_iterations_count += n_itr

else:

    p_mean_i = target_phasefield_final
    phasefield_stepsize /= 4

    print(f'\n *** Solving for p_mean: {p_mean_i:4.3} (final) *** \n')

    p_mean.assign(p_mean_i)

    n_itr, is_converged, error_reason = optimizer \
        .optimize(phasefield_stepsize, regularization)

    solver_iterations_count += n_itr

state_recording_function(insist=True)
print('CPU TIME:', time.time()-t0)

# Close output files (if can be closed)
if hasattr(outfile_u, 'close'): outfile_u.close()
if hasattr(outfile_p, 'close'): outfile_p.close()


if __name__ == "__main__":

    plt.interactive(True)

    ### Plot solutions

    def plot(*args, **kwargs):
        '''Plot either `np.ndarray`s or something plottable from `dolfin`.'''
        plt.figure(kwargs.pop('name', None))
        if isinstance(args[0], (np.ndarray,list, tuple)):
            plt.plot(*args, **kwargs)
        else: # just try it anyway
            dolfin.plot(*args, **kwargs)
        plt.show()

    def plot_energy_vs_iterations():

        fh = plt.figure('energy_vs_iterations')
        fh.clear()

        plt.plot(hist_potential, '-.k')
        plt.legend(['potential energy'])
        plt.ylabel('Energy')
        plt.xlabel('Iteration number')
        plt.title('Energy vs. Iterations')

        fh.tight_layout()
        plt.show()

    def plot_energy_vs_phasefield():

        EPS = 1e-12

        fh = plt.figure('energy_vs_phasefield')
        fh.clear()

        n = len(hist_phasefield)
        d = np.diff(hist_phasefield)

        ind = np.abs(d) > EPS
        ind = np.flatnonzero(ind)

        ind_first = (ind+1).tolist()
        ind_first.insert(0,0)

        ind_last = ind.tolist()
        ind_last.append(n-1)

        y_penalty = np.array(hist_penalty)[ind_last]
        y_potential = np.array(hist_potential)[ind_last]
        x_phasefield = np.array(hist_phasefield)[ind_last]

        # plt.plot(x_phasefield, y_cost, '-r')
        plt.plot(x_phasefield, y_potential, '-.g')
        plt.plot(x_phasefield, y_penalty, '--b')

        plt.legend(['potential', 'penalty'])

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

    plot_energy_vs_iterations()
    plot_energy_vs_phasefield()
    plot_phasefiled()

    plt.show()