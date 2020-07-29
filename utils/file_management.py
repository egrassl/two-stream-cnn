import os


def create_dir(path):
    if not os.path.exists(path):
        os.mkdir(path)
        return True
    else:
        return False
