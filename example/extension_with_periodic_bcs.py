# -*- coding: utf-8 -*-
"""
Created on 01/10/2018

@author: Danas Sutula

"""

import config

import os
import time
import math
import scipy
import dolfin
import logging
import numpy as np
import scipy.linalg as linalg
import matplotlib.pyplot as plt

# from scipy.io import loadmat
# from scipy.io import savemat
from dolfin import *

import optim
import material
import utility

logger = logging.getLogger()
logger.setLevel(logging.INFO)


### Results output

outdir = os.path.join('results', 'tmp')

if not os.path.exists(outdir):
    os.makedirs(outdir)

utility.remove_outfiles(outdir)

outfilepath_u = os.path.join(outdir, 'u.pvd')
outfilepath_p = os.path.join(outdir, 'p.pvd')

outfilepath_u_dofs = os.path.join(outdir, 'dofs_u.npy')
outfilepath_p_dofs = os.path.join(outdir, 'dofs_p.npy')

pre, ext = os.path.splitext(outfilepath_u_dofs)
outfilepath_u_dofs_fmt = (pre+'{:06d}'+ext).format

pre, ext = os.path.splitext(outfilepath_p_dofs)
outfilepath_p_dofs_fmt = (pre+'{:06d}'+ext).format

outfile_u = dolfin.File(outfilepath_u)
outfile_p = dolfin.File(outfilepath_p)

# outfile_u = dolfin.XDMFFile(outfile_u)
# outfile_p = dolfin.XDMFFile(outfile_p)


### Problem parameters

SCALE = 1 # Geometry scaling factor
LOAD_TYPE = 'biaxial'

external_loading_values = np.linspace(0, 0.01*SCALE, 2)[1:]
target_phasefield_values = np.linspace(0.0, SCALE*0.020, 25)


### Model parameters

material_parameters = {
    'E': Constant(1.0),
    'nu': Constant(0.3)
    }

# Minimum material integrity
m_inf  = Constant(1e-5)


### Discretization parameters

nx = ny = 80 # number of elements

mesh_pattern = "left/right"
# mesh_pattern = "crossed"
# mesh_pattern = "left"

phasefield_stepsize = 0.01 # Maximum phasefield nodal change per iteration.

regularization = 0.475 # Phasefield advance direction is determined by weighting
                      # the strain energy gradient and the penalty-like energy
                      # gradient. Parameter `regularization` weights the penalty
                      # whereas `(1-regularization)` weights the strain energy.

phasefield_alpha = 0.25 # Threshold-like parameter. Phasefield value greater than
                        # `phasefield_alpha` marks the subdomain whose boundary
                        # is considered to be the "phasefield boundary", which
                        # is used in phasefield collision detection.

phasefield_kappa = 5e-2 # Diffusion-like parameter (>0.0) used to regularize
                        # the distance equation so that the phasefield distance
                        # problem can be solved uniquely.

phasefield_gamma = 1e6 # Penalty-like parameter that serves to weaky impose
                       # the Dirichlet BC's in the phasefield distance problem.

optim.config.parameters_topology_solver['maximum_iterations']    = 500
optim.config.parameters_topology_solver['maximum_divergences']   = 0
optim.config.parameters_topology_solver['collision_threshold']   = 0.1*SCALE
optim.config.parameters_topology_solver['convergence_tolerance'] = 1e-4

# NOTE: `collision_threshold` affects the phasefield collision detection.
# Smaller value makes collision detection more sensitive. The parameter is
# closelly related to parameter 'phasefield_kappa'.


### Mesh

mesh = UnitSquareMesh(nx, ny, diagonal=mesh_pattern)

# mesh.translate(Point(-0.5,-0.5))
mesh.scale(SCALE)

x = mesh.coordinates()

x_min = x.min(axis=0)
x_max = x.max(axis=0)

x0, y0 = x_min
x1, y1 = x_max

L = x1-x0
H = y1-y0

rtol = 1e-12
atol = rtol*min(L,H)

domain_size = assemble(1*dx(mesh))


### Boundary subdomains

def boundary_bot(x, on_boundary):
    return on_boundary and x[1] < y0+atol

def boundary_top(x, on_boundary):
    return on_boundary and x[1] > y1-atol

def boundary_lhs(x, on_boundary):
    return on_boundary and x[0] < x0+atol

def boundary_rhs(x, on_boundary):
    return on_boundary and x[0] > x1-atol



class CornerBottomLeft(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[0], x0, atol) and near(x[1], y0, atol)

class PeriodicBoundariesVertical(SubDomain):
    '''Right boundary is master boundary; left boundary is slave boundary.'''

    def inside(self, x, on_boundary):
        return on_boundary and x[0] > x1-atol

    # Map right to left
    def map(self, x, y):
        y[0] = x[0] - L
        y[1] = x[1]

class PeriodicBoundariesHorizontal(SubDomain):
    '''Top boundary is master boundary; Bottom boundary is slave boundary.'''

    def inside(self, x, on_boundary):
        return on_boundary and x[1] > y1-atol

    # Map top to bottom
    def map(self, x, y):
        y[0] = x[0]
        y[1] = x[1] - H

pbc = [PeriodicBoundariesVertical(), PeriodicBoundariesHorizontal()]



### Discretization

V_u = VectorFunctionSpace(mesh, 'CG', 1)
V_p = FunctionSpace(mesh, 'CG', 1)

# V_u = VectorFunctionSpace(mesh, 'CG', 1, constrained_domain=pbc)
# V_p = FunctionSpace(mesh, 'CG', 1, constrained_domain=pbc)

V_ux, V_uy = V_u.split()

u = Function(V_u, name="displacement")
p = Function(V_p, name="phasefield")

# Target mean phasefield
p_mean = Constant(0.0)


# Boundary conditions

uxD_lhs = Expression('-s', s=0.0, degree=0)
uxD_rhs = Expression(' s', s=0.0, degree=0)
uyD_bot = Expression('-s', s=0.0, degree=0)
uyD_top = Expression(' s', s=0.0, degree=0)
uxD_lbc = Expression('-s', s=0.0, degree=0)
uyD_lbc = Expression('-s', s=0.0, degree=0)

def set_boundary_displacement_values(s):
    '''Set values of boundary conditions.'''
    uxD_lhs.s = s
    uxD_rhs.s = s
    uyD_bot.s = s
    uyD_top.s = s
    uxD_lbc.s = s
    uyD_lbc.s = s

if LOAD_TYPE == 'biaxial':
    bcs_u = [
        DirichletBC(V_u.sub(0), uxD_lhs, boundary_lhs),
        DirichletBC(V_u.sub(0), uxD_rhs, boundary_rhs),
        DirichletBC(V_u.sub(1), uyD_bot, boundary_bot),
        DirichletBC(V_u.sub(1), uyD_top, boundary_top),
        ]

elif LOAD_TYPE == 'vertical':
    bcs_u = [
        DirichletBC(V_u.sub(1), uyD_bot, boundary_bot),
        DirichletBC(V_u.sub(1), uyD_top, boundary_top),
        DirichletBC(V_u.sub(0), Constant(0.0), boundary_bot),
        DirichletBC(V_u.sub(0), Constant(0.0), boundary_top),
        ]

else:
    raise ValueError('Parameter `LOAD_TYPE`?')

# Weak boundary conditions



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

# TEMP.
psi = inner(sym(grad(u)), sym(grad(u)))

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

### Initial phasefields

# ps = utility.insert_defect_array([ 0.3, 0.7], [ 0.5, 0.5], 2, 1, V_p, r=0.05+1e-12)
# ps = utility.insert_defect_array([ 0.3, 0.7], [ 0.3, 0.7], 2, 2, V_p, r=0.10+1e-12)
# ps = utility.insert_defect_array([ 0.3, 0.7], [ 0.3, 0.7], 3, 3, V_p, r=0.04+1e-12)
ps = utility.insert_defect_array([ 0.1, 0.9], [ 0.1, 0.9], 4, 4, V_p, r=0.05+1e-12)

# ps = utility.insert_defect_array_with_checker_pattern(
#     [ 0, 1], [   0,   1], 2, 2, p, r=0.025)

# ps = utility.insert_defect_array(
#     [ 0, 1], [   0,   1], 4, 4, p, r=0.025)

# ps = utility.insert_defect_array_with_checker_pattern(
#     [ 0, 1], [ 0, 1], 4, 4, V_p, r=0.025)


### Smooth initial (sharp) phasefields

diffusion_filter = optim.DiffusionFilter(V_p, kappa=5e-5 * 1)

for p_i in ps:

    diffusion_filter.apply(p_i)
    p_arr_i = p_i.vector().get_local()
    p_arr_i[p_arr_i < 0.0] = 0.0
    p_arr_i[p_arr_i > 1.0] = 1.0
    p_i.vector()[:] = p_arr_i

# Superpose local phasefields
p.assign(sum(ps))


### Solution state recording

class SolutionStateRecorder:
    def __init__(self, write_period_vars=1, write_period_func=1):

        self.index = []

        self.phasefield_target = []
        self.phasefield_actual = []

        self.write_period_vars = write_period_vars
        self.write_period_func = write_period_func

        self.count = 0

    def record(self, insist=False):
        if not self.count % self.write_period_vars or insist:

            self.index.append(self.count)

            self.phasefield_target.append(float(p_mean.values()))
            self.phasefield_actual.append(assemble(p*dx) / domain_size)

        if not self.count % self.write_period_func or insist:

            # outfile_u << u
            outfile_p << p
            # outfile_u.write(u, self.count)
            # outfile_p.write(p, self.count)

            np.save(outfilepath_u_dofs_fmt(self.count), u.vector().get_local())
            np.save(outfilepath_p_dofs_fmt(self.count), p.vector().get_local())

        self.count += 1


solution_state_recorder = SolutionStateRecorder(
    write_period_vars=1, write_period_func=10)

### Optimization problem

target_phasefield_values += assemble(p*dx)/assemble(1*dx(mesh))
target_phasefield_final = target_phasefield_values[-1] * 0.80

optimizer = optim.TopologyOptimizer(potential, penalty, constraint, p, ps, u, bcs_u,
    phasefield_alpha, phasefield_kappa, phasefield_gamma, solution_state_recorder.record)

for s_i in external_loading_values:
    set_boundary_displacement_values(s_i)
    optimizer.solve_equilibrium_problem()


### Solve phasefield optimization problem

hist_potential = []
iteration_count = 0

t0 = time.time()

for p_mean_i in target_phasefield_values:

    print(f'\n *** Solving for p_mean: {p_mean_i:4.3} *** \n')

    p_mean.assign(p_mean_i)

    list_W, is_converged, error_reason = optimizer \
        .optimize(phasefield_stepsize, regularization)

    hist_potential.extend(list_W)
    iteration_count += len(list_W)

else:

    phasefield_stepsize /= 2

    p_mean_i = target_phasefield_final

    print(f'\n *** Solving for p_mean: {p_mean_i:4.3} (final) *** \n')

    p_mean.assign(p_mean_i)

    list_W, is_converged, error_reason = optimizer \
        .optimize(phasefield_stepsize, regularization)

    hist_potential.extend(list_W)
    iteration_count += len(list_W)

    print(f'\n *** CPU TIME: {time.time()-t0}\n')


hist_phasefield_target = solution_state_recorder.phasefield_target
hist_phasefield_actual = solution_state_recorder.phasefield_actual

# Close output files (if can be closed)
if hasattr(outfile_u, 'close'): outfile_u.close()
if hasattr(outfile_p, 'close'): outfile_p.close()


if __name__ == "__main__":

    plt.interactive(True)

    ### Plot solutions

    def plot(*args, **kwargs):
        '''Plot either `np.ndarray`s or something plottable from `dolfin`.'''
        plt.figure(kwargs.pop('name', None))
        if isinstance(args[0], (np.ndarray, list, tuple)):
            plt.plot(*args, **kwargs)
        else: # just try it anyway
            dolfin.plot(*args, **kwargs)
        plt.show()

    def plot_energy_vs_iterations():

        fh = plt.figure('energy_vs_iterations')
        fh.clear()

        plt.plot(hist_potential, '-ok')

        plt.ylabel('Strain energy')
        plt.xlabel('Iteration number')
        plt.title('Energy vs. Iterations')

        fh.tight_layout()
        plt.show()

    def plot_energy_vs_phasefields():

        EPS = 1e-12

        fh = plt.figure('energy_vs_phasefield')
        fh.clear()

        n = len(hist_phasefield_target)
        d = np.diff(hist_phasefield_target)

        ind = np.abs(d) > EPS
        ind = np.flatnonzero(ind)

        ind_first = (ind+1).tolist()
        ind_first.insert(0,0)

        ind_last = ind.tolist()
        ind_last.append(n-1)

        y_potential = np.array(hist_potential)[ind_last]
        x_phasefield = np.array(hist_phasefield_target)[ind_last]

        plt.plot(x_phasefield, y_potential, '-ob')

        plt.ylabel('Strain energy')
        plt.xlabel('Phasefield fraction')
        plt.title('Energy vs. Phasefield')

        fh.tight_layout()
        plt.show()

    def plot_target_vs_actual_phasefield():

        EPS = 1e-12

        fh = plt.figure('target_vs_actual_phasefield')
        fh.clear()

        _first

        ind_first = (ind+1).tolist()
        ind_first.insert(0,0)

        ind_last = ind.tolist()
        ind_last.append(n-1)

        y_potential = np.array(hist_potential)[ind_last]
        x_phasefield = np.array(hist_phasefield_target)[ind_last]

        plt.plot(x_phasefield, y_potential, '-ob')

        plt.ylabel('Strain energy')
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
    plot_energy_vs_phasefields()
    plot_phasefiled()

    plt.show()
