
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


class SolutionWriter:

    SAFE_TO_REMOVE_FILE_TYPES = ('.pvd', '.vtu', '.npy')

    def __init__(self, outdir, u, p, period=1):

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
        self.period = period

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

        if self.count_calls % self.period:
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


### Phasefield initialization

def insert_defect(p, xc, r):

    V = p.function_space()
    p_arr = p.vector().get_local()

    x = V.tabulate_dof_coordinates()
    s = ((x-xc)**2).sum(axis=1)

    p_arr[s < r**2] = 1.0
    p.vector()[:] = p_arr

def meshgrid_uniform(xlim, ylim, nx, ny):

    x0, x1 = xlim
    y0, y1 = ylim

    x = np.linspace(x0, x1, nx)
    y = np.linspace(y0, y1, ny)

    x, y = np.meshgrid(x, y)

    x = x.reshape((-1,))
    y = y.reshape((-1,))

    return np.stack((x, y), axis=1)

def meshgrid_checker(xlim, ylim, nx, ny):

    x0, x1 = xlim
    y0, y1 = ylim

    nx_B = int(nx/2)
    nx_A = nx - nx_B

    ny_B = int(ny/2)
    ny_A = ny - ny_B

    assert nx_A >= nx_B
    assert ny_A >= ny_B

    dx = (x1 - x0) / (nx - 1)
    dy = (y1 - y0) / (ny - 1)

    if nx_A == nx_B:
        xlim_A = (x0, x1-dx)
        xlim_B = (x0+dx, x1)
    else:
        xlim_A = (x0, x1)
        xlim_B = (x0+dx, x1-dx)

    if ny_A == ny_B:
        ylim_A = (y0, y1-dy)
        ylim_B = (y0+dy, y1)
    else:
        ylim_A = (y0, y1)
        ylim_B = (y0+dy, y1-dy)

    grid_A = meshgrid_uniform(xlim_A, ylim_A, nx_A, ny_A)
    grid_B = meshgrid_uniform(xlim_B, ylim_B, nx_B, ny_B)

    return np.concatenate((grid_A, grid_B), axis=0)


### Plotting

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

