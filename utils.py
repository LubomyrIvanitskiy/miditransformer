import os


def list_files(dir):
    files = os.listdir(dir)
    all_files = list()
    for entry in files:
        full_path = os.path.join(dir, entry)
        if os.path.isdir(full_path):
            all_files = all_files + list_files(full_path)
        else:
            all_files.append(full_path)

    return all_files


def get_parent_dir(caller_module_file):
    """
    :param caller_module_file: pass __file__ from the caller module
    :return: a path for the parent dir of the caller module
    """
    from pathlib import Path
    return str(Path(caller_module_file).resolve().parents[1])

