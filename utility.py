
import os
import dolfin
import numpy as np
import matplotlib.pyplot as plt

from dolfin import Constant
from dolfin import Function
from dolfin import assemble
from dolfin import dx


WRITE_PVD_DISPLACEMENTS = False
WRITE_NPY_DISPLACEMENTS = False

WRITE_PVD_PHASEFIELDS = True
WRITE_NPY_PHASEFIELDS = False

EPS = 1e-12


class SolutionWriter:

    SAFE_TO_REMOVE_FILE_TYPES = ('.pvd', '.vtu', '.npy')

    def __init__(self, outdir, u, p, writing_period=1):

        if not isinstance(u, Function):
            raise TypeError

        if not isinstance(p, Function):
            raise TypeError

        if not os.path.exists(outdir):
            os.makedirs(outdir)

        # Remove any existing files in the subdirectory `outdir` provided they
        # have extensions that are in the sequence `self.SAFE_TO_REMOVE_FILE_TYPES`.
        remove_outfiles(outdir, self.SAFE_TO_REMOVE_FILE_TYPES)

        self.u = u
        self.p = p

        self.count_calls = 0
        self.index_write = 0

        self.writing_period = writing_period

        domain_size = assemble(1*dx(p.function_space().mesh()))
        self._ufl_form_p_mean = Constant(1.0/domain_size)*p*dx

        self.mean_phasefield_values = []

        self.outfile_u = dolfin.File(os.path.join(outdir, 'u.pvd'))
        self.outfile_p = dolfin.File(os.path.join(outdir, 'p.pvd'))

        prefix, extension = os.path.splitext(os.path.join(outdir, 'dofs_u.npy'))
        self.outfilepath_u_dofs_format = (prefix + '{:04d}' + extension).format

        prefix, extension = os.path.splitext(os.path.join(outdir, 'dofs_p.npy'))
        self.outfilepath_p_dofs_format = (prefix + '{:04d}' + extension).format

    def periodic_write(self):

        if self.count_calls % self.writing_period:
            self.count_calls += 1; return

        self.write()

    def write(self,
              forcewrite_displacements=False,
              forcewrite_phasefields=False,
              forcewrite_all=False):

        forcewrite_displacements |= forcewrite_all
        forcewrite_phasefields |= forcewrite_all

        self.mean_phasefield_values.append(
            assemble(self._ufl_form_p_mean))

        if WRITE_PVD_DISPLACEMENTS or forcewrite_displacements:
            self.outfile_u << self.u

        if WRITE_PVD_PHASEFIELDS or forcewrite_phasefields:
            self.outfile_p << self.p

        if WRITE_NPY_DISPLACEMENTS or forcewrite_displacements:
            np.save(self.outfilepath_u_dofs_format(self.index_write),
                    self.u.vector().get_local())

        if WRITE_NPY_PHASEFIELDS or forcewrite_phasefields:
            np.save(self.outfilepath_p_dofs_format(self.index_write),
                    self.p.vector().get_local())

        self.count_calls += 1
        self.index_write += 1

    def __del__(self):

        if hasattr(self.outfile_u, 'close'):
            self.outfile_u.close()

        if hasattr(self.outfile_p, 'close'):
            self.outfile_p.close()


def remove_outfiles(subdir, file_extensions):

    if not isinstance(file_extensions, (list, tuple)):
        file_extensions = (file_extensions,)

    if not all(isinstance(ext, str) for ext in file_extensions):
        raise ValueError('Parameter `file_extensions` must be '
                         'a (`list` or `tuple` of) `str`(s).')

    file_extensions = tuple(ext if ext.startswith('.')
        else ('.' + ext) for ext in file_extensions)

    for item in os.listdir(subdir):
        item = os.path.join(subdir, item)

        if os.path.isfile(item):
            _, ext = os.path.splitext(item)

            if ext in file_extensions:
                os.remove(item)


def unit_square_mesh(number_of_cells_along_edge,
                     mesh_pattern="left/right",
                     offset=(-0.5,-0.5), scale=1):
    '''
    Parameters
    ----------
    number_of_cells_along_edge: int
        Number of cells along edge.
    mesh_pattern: str
        Possible options: "left/right", "crossed", "left", or "right".

    '''

    if not hasattr(offset, '__len__') or len(offset) != 2:
        raise TypeError('Parameter `offset` must be a sequence of length `2`.')

    nx = ny = number_of_cells_along_edge
    mesh = dolfin.UnitSquareMesh(nx, ny, diagonal=mesh_pattern)

    mesh.translate(dolfin.Point(*offset))
    mesh.scale(scale)

    return mesh


def boundaries_of_rectangular_domain(mesh):

    x0, y0 = mesh.coordinates().min(0)
    x1, y1 = mesh.coordinates().max(0)

    atol_x = (x1 - x0) * EPS
    atol_y = (y1 - y0) * EPS

    bot = dolfin.CompiledSubDomain('x[1] < y0 && on_boundary', y0=y0+atol_y)
    rhs = dolfin.CompiledSubDomain('x[0] > x1 && on_boundary', x1=x1-atol_x)
    top = dolfin.CompiledSubDomain('x[1] > y1 && on_boundary', y1=y1-atol_y)
    lhs = dolfin.CompiledSubDomain('x[0] < x0 && on_boundary', x0=x0+atol_x)

    return bot, rhs, top, lhs


class PeriodicBoundaries(dolfin.SubDomain):
    '''Top boundary is slave boundary wrt bottom (master) boundary.
    Right boundary is slave boundary wrt left (master) boundary.'''

    def __init__(self, mesh):
        super().__init__() # !!!

        x0, y0 = mesh.coordinates().min(0)
        x1, y1 = mesh.coordinates().max(0)

        self.L = x1 - x0
        self.H = y1 - y0

        self.x_lhs = x0 + EPS*self.L
        self.y_bot = y0 + EPS*self.H

        self.x_rhs = x1 - EPS*self.L
        self.y_top = y1 - EPS*self.H

    def inside(self, x, on_boundary):
        '''Check if `x` is on the slave boundary'''
        return on_boundary and \
            ((x[0] > self.x_rhs and x[1] > self.y_bot) or
             (x[0] > self.x_lhs and x[1] > self.y_top))

    def map(self, x, y):
        '''Map master point `x` to slave point `y`.'''

        y[0] = x[0]
        y[1] = x[1]

        if x[0] < self.x_lhs and x[1] > self.y_top:
            pass

        elif x[0] > self.x_rhs and x[1] < self.y_bot:
            pass

        elif x[0] < self.x_lhs:
            y[0] += self.L

        elif x[1] < self.y_bot:
            y[1] += self.H


def uniform_biaxial_extension_bcs(V):
    '''

    Parameters
    ----------
    V: dolfin.FunctionSpace
        Vector function space for the displacement field.

    '''

    mesh = V.mesh()

    boundary_bot, boundary_rhs, boundary_top, boundary_lhs = \
        boundaries_of_rectangular_domain(mesh)

    uy_bot = Constant(0)
    ux_rhs = Constant(0)
    uy_top = Constant(0)
    ux_lhs = Constant(0)

    bcs = [
        dolfin.DirichletBC(V.sub(1), uy_bot, boundary_bot),
        dolfin.DirichletBC(V.sub(0), ux_rhs, boundary_rhs),
        dolfin.DirichletBC(V.sub(1), uy_top, boundary_top),
        dolfin.DirichletBC(V.sub(0), ux_lhs, boundary_lhs),
        ]

    def bcs_set_value(value):
        uy_bot.assign(-value)
        ux_rhs.assign(+value)
        uy_top.assign(+value)
        ux_lhs.assign(-value)

    return bcs, bcs_set_value


def uniform_uniaxial_extension_bcs(V):
    '''

    Parameters
    ----------
    V: dolfin.FunctionSpace
        Vector function space for the displacement field.

    '''

    mesh = V.mesh()

    boundary_bot, _, boundary_top, _ = \
        boundaries_of_rectangular_domain(mesh)

    uy_bot = Constant(0)
    uy_top = Constant(0)

    ux_swc = Constant(0) # South-West corner
    ux_nwc = Constant(0) # North-West corner

    bcs = [
        dolfin.DirichletBC(V.sub(1), uy_bot, boundary_bot),
        dolfin.DirichletBC(V.sub(1), uy_top, boundary_top),
        dolfin.DirichletBC(V.sub(0), ux_swc, boundary_top),
        dolfin.DirichletBC(V.sub(0), ux_nwc, boundary_bot),
        ]

    def bcs_set_value(value):
        uy_bot.assign(-value)
        uy_top.assign(+value)

    return bcs, bcs_set_value


def plot_energy_vs_iterations(potential_vs_iteration,
                              figname="energy_vs_iterations",
                              ylabel='Energy'):

    fh = plt.figure(figname)
    fh.clear(); ah=fh.subplots()

    ah.plot(potential_vs_iteration, '-k')

    ah.set_ylabel(ylabel)
    ah.set_xlabel('Iteration number')

    # axis_limits = ah.axis()
    # axis_limits[-2] = 0.0 # Set y0 at zero
    # ah.axis()

    fh.tight_layout()
    fh.show()

    return fh, figname


def plot_energy_vs_phasefields(potential_vs_phasefield,
                               mean_phasefield_values,
                               figname="energy_vs_phasefield",
                               ylabel='Energy'):

    fh = plt.figure(figname)
    fh.clear(); ah=fh.subplots()

    ah.plot(mean_phasefield_values, potential_vs_phasefield, '-k')

    ah.set_ylabel(ylabel)
    ah.set_xlabel('Phasefield fraction')

    fh.tight_layout()
    fh.show()

    return fh, figname


def plot_phasefiled(p, figname="final_phasefield"):

    fh = plt.figure(figname)
    fh.clear(); ah=fh.subplots()

    dolfin.plot(p)

    plt.title('Phase-field, p\n(p_min = {0:.5}; p_max = {1:.5})'.format(
        p.vector().get_local().min(), p.vector().get_local().max()))

    fh.tight_layout()
    fh.show()

    return fh, figname
