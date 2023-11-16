import os

def get_files_paths(root):

    paths = []

    for dirname, dirnames, filenames in os.walk(root):
        for filename in filenames:
            paths.append(os.path.join(dirname, filename))

    return paths

def get_files_names(root):

    names = []

    for dirname, dirnames, filenames in os.walk(root):
        for filename in filenames:
            names.append(filename)

    return names
