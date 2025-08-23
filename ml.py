import numpy as np
from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    StratifiedKFold,
)
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import f1_score, root_mean_squared_error, make_scorer
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.utils.multiclass import type_of_target
import time
import pandas as pd
import json


def generate_fea_tar_model(features, target, model_type="rf"):
    """
    Generate a list of feature-target model configurations.
    """
    fea_tar_model = [
        {"features": features[i], "target": target[j]}
        for i in range(len(features))
        for j in range(len(target))
    ]
    if model_type == "rf":
        reg_model = RandomForestRegressor()
        cls_model = RandomForestClassifier()
    elif model_type == "knn":
        reg_model = KNeighborsRegressor()
        cls_model = KNeighborsClassifier()
    else:
        raise ValueError(f"Can not recognize the model {model_type}")

    for i in range(len(fea_tar_model)):
        if "bl" in fea_tar_model[i]["target"]:
            fea_tar_model[i]["model"] = reg_model
            fea_tar_model[i]["score_method"] = make_scorer(
                lambda y_true, y_pred: root_mean_squared_error(y_true, y_pred)
                / np.mean(y_true)
            )
        else:
            fea_tar_model[i]["model"] = cls_model
            fea_tar_model[i]["score_method"] = make_scorer(
                lambda y_true, y_pred: f1_score(
                    y_true, y_pred, average="weighted"
                )
            )
    return fea_tar_model


def convert_features_name(features):
    map_name = {
        "xanes": "XANES",
        "x_pdf": "xPDF",
        "n_pdf": "nPDF",
        "nx_pdf": "nxPDF",
        "diff_x_pdf": "dPDF",
        "n/x_pdf": "n/xPDF",
    }
    if isinstance(features, list):
        features_name = [
            [map_name[features[i][j]] for j in range(len(features[i]))]
            for i in range(len(features))
        ]
        features_name = [
            "+".join(features_name[i]) for i in range(len(features_name))
        ]
    elif isinstance(features, str):
        features_name = map_name[features]
    return features_name


def build_train_test_data(load_path, features, target, random_seed=40):
    with open(load_path, "rb") as f:
        docs = json.load(f)
    print("Start train-test-split on element {}".format(load_path))
    mp_ids = []
    features_data = []
    target_data = []
    for doc_ind, doc in enumerate(docs):
        features_data_item = []
        try:
            for i in range(len(features)):
                if features[i] not in doc.keys():
                    raise KeyError(f"{features[i]} not exists in {doc.keys()}")
                if features[i] == "nx_pdf":
                    xpdf = doc["x_pdf"]
                    npdf = doc["n_pdf"]
                    features_data_item.extend(xpdf - npdf)
                else:
                    features_data_item.extend(doc[features[i]])
            target_data.append(doc[target])
            mp_ids.append(doc["mp_id"])
        except:
            print(f"Unable to fetch {features, target} from {doc["mp_id"]}")
            continue
        features_data.append(features_data_item)
    mp_ids = np.array(mp_ids)
    features_data = pd.DataFrame(features_data, mp_ids)
    target_data = pd.DataFrame(target_data, mp_ids)
    mask1 = features_data.isna().any(axis=1)
    mask2 = target_data.isna().any(axis=1)
    mask3 = target_data.iloc[:, 0].values == "na"
    mask = mask1 + mask2 + mask3
    features_data = features_data[~mask]
    target_data = target_data[~mask].values.ravel()
    if "bl" in target:
        target_data = target_data.astype(np.float64)
        X_train, X_test, y_train, y_test = train_test_split(
            features_data, target_data, random_state=random_seed
        )
    else:
        stratify_mask = get_stratifiable_mask(target_data)
        features_data = features_data[stratify_mask]
        target_data = target_data[stratify_mask]
        target_data = target_data.astype(np.int8)
        X_train, X_test, y_train, y_test = train_test_split(
            features_data,
            target_data,
            random_state=random_seed,
            stratify=target_data,
        )

    return X_train, X_test, y_train, y_test


def get_stratifiable_mask(y):
    counts = pd.Series(y).value_counts()
    valid_classes = counts[counts >= 2].index
    mask = np.isin(y, valid_classes)
    return mask


def train_model_hyper(
    model, X, y, param_grid, score_method, show=False, stratify=True
):
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
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=40)
        grid = GridSearchCV(
            model, param_grid=param_grid, cv=cv, scoring=score_method
        )
    else:
        grid = GridSearchCV(model, param_grid=param_grid, scoring=score_method)
    grid.fit(X, y)
    if show:
        print(f"Best params found by GridSearch: {grid.best_params_}")
    return grid.best_params_


def train_model(
    X_train,
    X_test,
    y_train,
    y_test,
    tune_hyper,
    model_params,
    model_type,
    model,
    score_method,
    n_itr=10,
):
    start_rseed = 40
    test_scores = []
    train_scores = []
    start = time.time()
    # stratify
    if "continuous" not in type_of_target(y_train):
        stratify = True
    else:
        stratify = False
    # tune_hyper
    if tune_hyper:
        model_params = train_model_hyper(
            model,
            X_train,
            y_train,
            param_grid=model_params,
            score_method=score_method,
            stratify=stratify,
        )
        print(
            "Tune parameter finished. Cost {} seconds. Params: {}".format(
                time.time() - start, model_params
            )
        )
    # importances for RandomForest
    if model_type == "rf":
        importances = np.zeros([n_itr, X_train.shape[1]])
    elif model_type == "knn":
        pass
    else:
        raise KeyError(
            f"Unable to recognize {model_type}. Please choose from ['knn', 'rf']"
        )
    # return test scores
    for i in range(n_itr):
        start = time.time()
        model.set_params(**model_params)
        model.set_params(random_state=start_rseed + i)
        model.fit(X_train, y_train)
        test_scores.append(
            score_method(estimator=model, X=X_test, y_true=y_test)
        )
        train_scores.append(
            score_method(estimator=model, X=X_train, y_true=y_train)
        )
        end = time.time()
        print(
            "{} iteration finished. Cost {} seconds. Score {}".format(
                i, end - start, test_scores[-1]
            )
        )
        if model_type == "rf":
            importances[i] = model.feature_importances_
            outcome = {
                "train_scores": train_scores,
                "test_scores": test_scores,
                "importances": importances,
                "model_params": model_params,
            }
        else:
            outcome = {
                "train_scores": train_scores,
                "test_scores": test_scores,
                "model_params": model_params,
            }
    return outcome


if __name__ == "__main__":
    pass
