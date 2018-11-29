import os

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
