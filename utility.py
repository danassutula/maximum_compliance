import os
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
