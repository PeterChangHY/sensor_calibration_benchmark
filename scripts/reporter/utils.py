#!/usr/bin/env python
""" common utilities for reporter """
import os


def find_dir(dir_name):
    search_dirs = []
    venv_d = os.getenv("VIRTUAL_ENV")
    if venv_d:
        search_dirs.append(os.path.join(venv_d, "../reporter", dir_name))
        search_dirs.append(os.path.join(venv_d, "../reporter"))
    search_dirs.append(os.path.join(os.getcwd(), dir_name))
    found_d = None
    for d in search_dirs:
        if os.path.isdir(d):
            found_d = d
            break
    return found_d, search_dirs


def walk_files(base_path, prefix='', suffix='.tsv', full_path=True):
    for name in os.listdir(base_path):
        full_file_path = os.path.join(base_path, name)
        if name.startswith(prefix) and name.endswith(suffix) and os.path.isfile(full_file_path):
            if full_path:
                yield full_file_path
            else:
                yield name
