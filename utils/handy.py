import re
from pathlib import Path
import pickle
import numpy as np

def print_list(*lists_like):
    for j,list_like in enumerate(lists_like):
        for i in range(len(list_like)):
            print(list_like[i])
        
        if j != len(lists_like)-1:
            print('\n')

def find_files(dir_path, file_func, mode='current'):
    dir_path = Path(dir_path)
    files = []
    if mode == 'global':
        for path in dir_path.rglob("*"):
            if path.is_file() and file_func(path.name):
                files.append(path)
    else:
        for path in dir_path.iterdir():
            if path.is_file() and file_func(path.name):
                files.append(path)

    return files

def sort_files(file_paths, sort_pattern, return_keys=False):
    if not isinstance(file_paths[0], Path):
        raise TypeError("file paths should be instance of pathlib.Path")
    
    keys = []
    for path in file_paths:
        match = sort_pattern.findall(path.name)
        if len(match) == 0:
            raise ValueError("Can not find match for {path.name}")
        keys.append(match[0])
    
    keys, sorted_file_paths = zip(*sorted(zip(keys,file_paths)))
    if return_keys:
        return sorted_file_paths, keys
    else:
        return sorted_file_paths

def bunch_load_pickle(*paths):
    out = []
    for path in paths:
        with open(path, 'rb') as f:
            out.append(pickle.load(f))
    return out

# path = 'datasets/model_data'
# path = Path(path)
# t_elements = ['Ti', 'Cu', 'Fe', 'Mn']
# t_target = ['cs', 'cn', 'bl']

# t_features = []
# for file in path.iterdir():
    # if file.is_file() and file.suffix=='.npy':
        # tags = file.stem.split('_')
        # if tags[3] not in ['dpdf', 'pdf', 'xanes']:
            # print(file.name)
        # t_features.append(tags[3])

# path = "datasets/model_data/cs_Ti_y_test.npy"
# data = np.load(path)
# print(data)


if __name__ == '__main__':
    pass