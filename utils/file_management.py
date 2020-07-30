import os


def create_dir(path, verbose=False):
    '''
    Creates a directory if it doesn't already exists

    :param path: path to directory
    :param verbose: prints when the directory is created if it is set to True
    :return: True if directory was created and False if it already existed
    '''
    if not os.path.exists(path):
        os.mkdir(path)
        if verbose:
            print('Directory %s created' % path)
        return True
    else:
        return False


def create_dirs(paths, verbose=False):
    '''
    Try to create directories specified by a list and returns True if one or more directory was created

    :param paths: list of directories to create
    :param verbose: prints when a directory is created if it is set to True
    :return: returns True if one or more directories were created and False otherwise
    '''
    created = False
    for path in paths:
        created = create_dir(path, verbose) or created

    return created
