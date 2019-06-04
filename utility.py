import os
import dolfin
import numpy as np

# Extensions of files that are allowed to be deleted
FILE_EXTENSIONS = ['.pvd', '.vtu', '.h5', '.xdmf']


def cleanup_filepath(filepath):
    '''Clean up file path. If file directory does not exist, it is created;
    otherwise, if directory already exists, any existing files are removed.
    '''

    dirpath, filename = os.path.split(filepath)
    _, ext = os.path.splitext(filename)

    if ext not in FILE_EXTENSIONS:
        raise ValueError('File name is missing a valid file extension')

    if not dirpath or dirpath == os.path.curdir or dirpath == os.path.sep:
        raise ValueError('File path should contain at least one sub-directory')

    if not os.path.exists(dirpath):
        os.makedirs(dirpath)

    dirpath += os.sep

    for item in os.listdir(dirpath):
        item = dirpath + item

        if os.path.isfile(item):
            _, ext = os.path.splitext(item)

            if ext in FILE_EXTENSIONS:
                os.remove(item)


# def make_function_caller(func, period=1):

#     seq = []
#     idx = 0

#     def function_caller(insist=False):

#         nonlocal idx

#         if idx % period == 0 or insist:

#             func(idx)

#             seq.append(idx)

#         idx += 1

#     return function_caller, seq


def insert_defect(p, xc, r):

    V = p.function_space()
    p_arr = p.vector().get_local()

    x = V.tabulate_dof_coordinates()
    s = ((x-xc)**2).sum(axis=1)

    p_arr[s < r**2] = 1.0
    p.vector()[:] = p_arr


def insert_defect_array(xlim, ylim, n, m, V, r, rtol):

    x = np.linspace(xlim[0], xlim[1], n)
    y = np.linspace(ylim[0], ylim[1], m)

    x, y = np.meshgrid(x, y)

    x = x.reshape((-1,))
    y = y.reshape((-1,))

    ps = []

    for xc in np.stack([x,y], axis=1):
        p = dolfin.Function(V)
        insert_defect(p, xc, r)
        ps.append(p)

    return ps


def insert_defect_array_with_checker_pattern(xlim, ylim, n, m, V, r, rtol):

    ps = []

    ps.extend(insert_defect_array(xlim, ylim, n, m, V, r, rtol))

    dx = (xlim[1] - xlim[0]) / (n-1)
    dy = (ylim[1] - ylim[0]) / (m-1)

    xlim = [xlim[0] + 0.5*dx, xlim[1] - 0.5*dx]
    ylim = [ylim[0] + 0.5*dy, ylim[1] - 0.5*dy]

    n -= 1
    m -= 1

    ps.extend(insert_defect_array(xlim, ylim, n, m, V, r, rtol))

    return ps
