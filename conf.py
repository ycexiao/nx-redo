import os
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import make_scorer, f1_score, root_mean_squared_error
import numpy as np
import re
from utils import *

# set load_path
elements = ["Ti", "Cu", "Fe", "Mn"]
file_names = [
    element + "_collection.json" for element in elements
]  # filename or path to the collection
load_dir = "datasets"
load_path = [os.path.join(load_dir, file_names[i]) for i in range(len(file_names))]

features = [
    ['xanes'],
    ['xanes', 'x_pdf'],
    ['xanes', 'nx_pdf'],
    ['xanes', 'diff_x_pdf'],
    ['xanes', 'x_pdf', 'n_pdf'],
    ['xanes', 'x_pdf', 'nx_pdf'],
    ['xanes', 'n_pdf', 'nx_pdf'],
]
# set target of interest
target = [
    'cn', 'cs', 'bl'
]

def generate_fea_tar_model(features, target):
    """
    Generate a list of feature-target model configurations.
    """
    fea_tar_model = [
        {'features': features[i],
         'target': target[j]} for i in range(len(features)) for j in range(len(target))
    ]
    for i in range(len(fea_tar_model)):
        if fea_tar_model[i]['target'] == 'bl':
            fea_tar_model[i]['model'] = RandomForestRegressor()
            fea_tar_model[i]['score_method'] = make_scorer(lambda y_true, y_pred: root_mean_squared_error(y_true, y_pred)/np.mean(y_true))
        else:
            fea_tar_model[i]['model'] = RandomForestClassifier()
            fea_tar_model[i]['score_method'] = make_scorer(lambda y_true, y_pred: f1_score(y_true, y_pred, average='weighted'))
    return fea_tar_model



def convert_features_name(features):
    map_name = {
    'xanes': 'XANES',
    'x_pdf':'xPDF',
    'n_pdf':'nPDF',
    'nx_pdf': 'nxPDF', 
    'diff_x_pdf': 'dPDF',
    'n/x_pdf': 'n/xPDF'
    }
    features_name = [[map_name[features[i][j]] for j in range(len(features[i]))] for i in range(len(features))]
    features_name = ['+'.join(features_name[i]) for i in range(len(features_name))]
    return features_name



random_seed_pattern = re.compile(r'(?<=-)(\d+).pkl$')

is_data = lambda x: x.count('-') == 7 and x.startswith('train-test-data')
data_path, random_seeds = sort_files(find_files('active-data-house', is_data), random_seed_pattern, return_keys=True)

is_track = lambda x: x.count('-') == 7 and x.startswith('data-split-track')
track_path = sort_files(find_files('active-data-house', is_track), random_seed_pattern)

is_results = lambda x: x.count('-')==5 and x.startswith('results')
results_path  = sort_files(find_files('active-data-house', is_results), random_seed_pattern)

if __name__ == "__main__":
    x = generate_fea_tar_model()
    pass
