import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import f1_score, root_mean_squared_error, make_scorer
import time
import pickle
import pandas as pd
import json
import os
from utils.handy import *
from utils.analysis import *

def has_enough_components(Y):
    m,n = Y.shape
    mask = np.ones(m)
    for i in range(n):
        if i!= n-1:
            labels, counts = np.unique(Y[:,i], return_counts=True)
            if np.all(counts >= 5):
                continue
            else:
                not_enough_labels = labels[np.argwhere(counts<5)]
                for l in not_enough_labels:
                    mask[Y[:,i]==l] = 0
    return mask.astype('bool')

def build_train_test_data(elements, load_path, features, target, rseed):
    out = []
    track_data_split = []
    for i in range(len(elements)):
        print("Start train-test-split on element {}".format(elements[i]))
        X, Y, mp_ids, mask = get_model_data(load_path[i])

        mask = np.all(~np.isnan(Y), axis=1)  
        X, Y, mp_ids = X[mask], Y[mask], mp_ids[mask]

        mask = has_enough_components(Y)
        X, Y, mp_ids = X[mask], Y[mask], mp_ids[mask]

        X = pd.DataFrame(X, index=mp_ids)
        Y = pd.DataFrame(Y, index=mp_ids)
        for j in range(len(features)):
            for k in range(len(target)):
                if target[k] in ['cs', 'cn']:
                    X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=rseed, stratify=Y.iloc[:,k])
                else:
                    X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=rseed)
                constructed_X_train = get_model_features(X_train, features[j])
                constructed_X_test = get_model_features(X_test, features[j])

                tmp = {}
                tmp['element'] = elements[i]
                tmp['features'] = features[j] 
                tmp['target'] = target[k]
                tmp['train_test'] = constructed_X_train, constructed_X_test, y_train.iloc[:,k], y_test.iloc[:,k]
                out.append(tmp)

                tmp_track = {}
                tmp_track['element'] = elements[i]
                tmp_track['features'] = features[j] 
                tmp_track['target'] = target[k]
                tmp_track['train_mp_ids'] = y_train.index.to_list()
                tmp_track['test_mp_ids'] = y_test.index.to_list()
                track_data_split.append(tmp_track)

    return out, track_data_split


def get_model_data(data_path):
    """Get data from datasets and do some basic data filtering.

    Parameters
    ----------
    data_path: str
        path to load the data

    Returns
    -------
    X: array_like
        inputs of the ML model
    Y: array_like
        outputs of the ML model
    """
    with open(data_path, "r") as f:
        docs = json.load(f)

    xanes = np.zeros([len(docs), len(docs[0]["xanes"])])
    x_pdf = np.zeros([len(docs), len(docs[0]["x_pdf"])])
    n_pdf = np.zeros([len(docs), len(docs[0]["n_pdf"])])
    diff_x_pdf = np.zeros([len(docs), len(docs[0]["diff_x_pdf"])])
    diff_n_pdf = np.zeros([len(docs), len(docs[0]["diff_n_pdf"])])

    Y = np.zeros([len(docs), 3])

    ids = []

    masks = np.zeros(len(docs), dtype=bool)

    for i in range(len(docs)):
        try:
            xanes[i] = docs[i]["xanes"]
            x_pdf[i] = docs[i]["x_pdf"]
            n_pdf[i] = docs[i]["n_pdf"]
            diff_x_pdf[i] = docs[i]["diff_x_pdf"]
            diff_n_pdf[i] = docs[i]["diff_n_pdf"]
            Y[i, 0] = docs[i]["cs"]
            Y[i, 1] = docs[i]["cn"]
            Y[i, 2] = docs[i]["bl"]
            masks[i] = 1
            ids.append(docs[i]["mp_id"])
        except ValueError:
            ids.append('none')
            continue

    X = np.hstack([xanes, x_pdf, n_pdf, diff_x_pdf, diff_n_pdf])
    ids = np.array(ids)
    return X[masks], Y[masks], ids[masks], masks


def get_model_features(X, names, feature_length=200, force_length=100):
    """Select the features from the loaded dats. Information about the order and
    length of each features are known outside of this function.

    Parameters
    ----------
    X: array_like
        Inputs of the ML model
    names: list.
        str of the features want to use
        options: 'xanes', 'x_pdf', 'n_pdf', 'diff_x_pdf' 'diff_n_pdf'
    feature_length: int
        length of the feature in the datasets.
    force_length: int
        length of the feature we want to use as model input.

    Returns
    -------
    new_X: array_like
    """
    optional_names = ["xanes", "x_pdf", "n_pdf", "diff_x_pdf", "diff_n_pdf", 'nx_pdf', 'n/x_pdf']
    dict_names = {optional_names[i] : (i*feature_length, (i+1)*feature_length) for i in range(len(optional_names))}
    new_X = np.zeros([len(X), force_length * len(names)])
    new_X_inds = [(i*force_length, (i+1)*force_length) for i in range(len(names))]

    for i in range(len(names)):
        if names[i] == 'nx_pdf':
            xpdf = X.iloc[:, dict_names['x_pdf'][0]: dict_names['x_pdf'][1]]
            npdf = X.iloc[:, dict_names['n_pdf'][0] : dict_names['n_pdf'][1]]
            feature = xpdf.values - npdf.values
        elif names[i] == 'n/x_pdf':
            xpdf = X.iloc[:, dict_names['x_pdf'][0]: dict_names['x_pdf'][1]]
            npdf = X.iloc[:, dict_names['n_pdf'][0] : dict_names['n_pdf'][1]]
            feature = xpdf.values / npdf.values
        else:
            feature = X.iloc[:, dict_names[names[i]][0]:dict_names[names[i]][1]].values
        for j in range(len(X)):
            new_X[j, new_X_inds[i][0]:new_X_inds[i][1]] = np.interp(
                np.linspace(0, len(feature), force_length),
                np.linspace(0, len(feature), feature_length),
                feature[j]
            )

    return new_X


def get_model_target(Y, name):
    """Select the target used for training. Information about the order and
    length of each target are known outside this function.

    Parameters
    ----------
    X: array_like
        Inputs of the ML model
    name: str
        str of the target
        options: 'cs', 'cn' or 'bl'

    Returns
    -------
    new_Y
    """
    optional_names = ["cs", "cn", "bl"]
    dict_names = {optional_names[i]: i for i in range(len(optional_names))}
    new_Y = np.zeros(len(Y))
    new_Y = Y[:, dict_names[name]]
    return new_Y


def train_model_hyper(model, X, y, param_grid, score_method, show=False, stratify=True):
    """Use cv to hyper-tune the model.

    Parameters
    ----------
    model:
        sklearn model to be trained
    X: array_like
    y: array_like
    model_params: dict
        dict of params to be tuning

    """
    if stratify:
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        grid = GridSearchCV(model, param_grid=param_grid, cv=cv,scoring=score_method)
    else:
        grid = GridSearchCV(model, param_grid=param_grid, scoring=score_method)
    grid.fit(X, y)
    if show:
        pass
    return grid.best_params_


def train_model(
    X_train,
    X_test,
    y_train,
    y_test,
    model,
    features,
    target,
    model_params,
    score_method,
    tune_hyper=True,
    n_itr=10,
):
    """
    1. load data, choose feautres and targets
    2. cross_val on trainning set to get param
    3. for each (features, target) combination
        3.1. train the model
        3.2. redo train_test_split and repeat 2.1-2.2 <n_iter> times and store the scores

    Parameters
    ----------
    data_path: str
        path to the data.
    model: 
        sklearn model used to train
    features: list of str
    target: str
    grid_search_params: list
    score_method: 
    dump_prefix: str
        specifying additional information associated with te datasets.
    """

    # start train
    test_scores = []
    train_scores = []
    start = time.time()

    if target in ['cs', 'cn']:
        stratify = True
    else:
        stratify = False

    print(model_params)
    if tune_hyper:        
        model_params = train_model_hyper(
                model, X_train, y_train, param_grid=model_params, score_method=score_method, stratify=stratify
            )
        print('Tune parameter finished. Cost {} seconds. Params: {}'.format(time.time()-start, model_params))    
    
    importances = np.zeros([n_itr, X_train.shape[1]])
    if target == 'bl':
        y_pred = np.zeros([n_itr+1, len(y_test)])
        y_pred[0] = y_test
    rseed=40

    for i in range(n_itr):
        start = time.time()

        model.set_params(**model_params)
        model.set_params(random_state=rseed+i)
        model.fit(X_train, y_train)

        test_scores.append(score_method(estimator=model, X=X_test, y_true=y_test))
        train_scores.append(score_method(estimator=model, X=X_train, y_true=y_train))
        if target == 'bl':
            y_pred[i+1] = model.predict(X_test)

        end = time.time()
        print(
            "{} iteration finished. Cost {} seconds. Score {}".format(
                i, end - start, test_scores[-1]
            )
        )
        importances[i] = model.feature_importances_

    scores = np.array([train_scores, test_scores])

    if target == 'bl':
        return scores, importances, model_params, y_pred

    return scores, importances, model_params

def lookup_model_params(results_path, *selection):
    """
    Use the already trained model parameters to avoid run GridSearchCV everytime.

    Parameters
    ----------
    results_path: str
        path of the results
    dict: 
        dictionary of the queried training tasks.

    Returns
    -------
    out_dict: 
        dictionary of the trained parameters
    """
    if not os.path.exists(results_path):
        raise FileNotFoundError
    try:
        with open(results_path, 'rb') as f:
            trained_results = pickle.load(f)
    except EOFError:
        print("File {} is empty, no trained results.".format(results_path))
        return {}, False

    try:
        model_params = lookup(trained_results, "model_params", *selection)
    except IndexError:
        return {}, False
    return model_params, True


if __name__ == '__main__':
    pass