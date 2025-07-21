import os
import time
import pickle
from ml import build_train_test_data, generate_fea_tar_model, train_model
from helper import easyDatabase

# tune_hyper = True
# results_path = 'empty_pickle.pkl'

elements = ["Ti", "Cu", "Fe", "Mn"]
file_names = [
    element + "_collection.json" for element in elements
]  # filename or path to the collection
load_dir = "datasets"
load_path = [
    os.path.join(load_dir, file_names[i]) for i in range(len(file_names))
]

targets = ["cs", "cn", "bl"]

features = [
    ["x_pdf", "n_pdf"],
    ["nx_pdf"],
    ["xanes", "x_pdf", "n_pdf"],
    ["xanes", "nx_pdf"],
    ["xanes"],
    ["x_pdf"],
    ["n_pdf"],
    ["xanes", "x_pdf"],
    ["xanes", "n_pdf"],
    ["xanes", "diff_x_pdf"],
]

n_estimators = [25, 50, 100, 200, 300]
max_features = [10, 15, 20, 25, 30, 35]
default_model_params = {
    "n_estimators": n_estimators,
    "max_features": max_features,
}


def main(
    training_history="combined_data.pkl",
    random_seed=40,
):
    """

    Perform the training using configurations from `conf.py`.

    train-test-data is changed because new features in the X_train and X_test
    results-data is changed because new resuts.
    """
    with open(training_history, "rb") as f:
        training_history = pickle.load(f)
    train_test_data, track_data = build_train_test_data(
        elements, load_path, features, targets, random_seed
    )
    keynames = ["target", "element", "features"]
    split_valuekeys = ["train_mp_ids", "test_mp_ids"]
    split_database = easyDatabase(keynames)
    for k in range(len(track_data)):
        keys = [track_data[k][key] for key in keynames]
        value = {key: track_data[k][key] for key in split_valuekeys}
        split_database.add_member(keynames, keys, value)
    train_test_valuekeys = ["train_test"]
    train_test_database = easyDatabase(keynames)
    for k in range(len(train_test_data)):
        keys = [train_test_data[k][key] for key in keynames]
        value = {key: train_test_data[k][key] for key in train_test_valuekeys}
        train_test_database.add_member(keynames, keys, value)

    fea_tar_model = generate_fea_tar_model(features, targets)
    total_start = time.time()
    total_outs = []

    for j in range(len(elements)):
        print("Start on element {}".format(elements[j]))
        for k in range(len(fea_tar_model)):

            round_start = time.time()
            print(
                "Start training for \n\tFeatures:{}\n\tTarget:{}".format(
                    fea_tar_model[k]["features"], fea_tar_model[k]["target"]
                )
            )

            fea = fea_tar_model[k]["features"]
            tar = fea_tar_model[k]["target"]
            ele = elements[j]

            X_train, X_test, y_train, y_test = (
                train_test_database.filter_data(
                    ["element", "features", "target"], [ele, fea, tar]
                )
            ).value["train_test"]

            try:
                model_params = (
                    training_history["results"]
                    .records[-1]
                    .filter_data(
                        ["element", "features", "target"], [ele, fea, tar]
                    )
                    .value["model_params"]
                )
                findit = True
            except KeyError:
                model_params = default_model_params
                findit = False

            # # findit = False  # Retrain
            if findit:
                print(
                    "Use trained parameters. Params: {}".format(model_params)
                )
                fea_tar_model[k]["model_params"] = model_params
            else:
                fea_tar_model[k]["model_params"] = default_model_params
                print("Hyper-parameters not Found. Run GridSearch.")
            tune_hyper = not findit  # if no find the data, run gridsearchcv.
            train_outcome = train_model(  # main step
                X_train=X_train,
                X_test=X_test,
                y_train=y_train,
                y_test=y_test,
                tune_hyper=tune_hyper,
                **fea_tar_model[k]
            )
            out = {}
            out["train_scores"] = train_outcome[0][0]
            out["test_scores"] = train_outcome[0][1]
            out["importances"] = train_outcome[1]
            out["model_params"] = train_outcome[2]
            if len(train_outcome) == 4:
                out["y_pred"] = train_outcome[3]
            out["element"] = ele
            out["features"] = fea
            out["target"] = tar
            total_outs.append(out)
            print(
                "{}/{} round finished, cost {} seconds in this round.".format(
                    k + j * len(fea_tar_model),
                    len(fea_tar_model) * len(elements),
                    time.time() - round_start,
                )
            )
            print("\n\n")

    print("Total {} seconds".format(time.time() - total_start))

    keynames = ["target", "element", "features"]
    result_valuekeys = [
        "train_scores",
        "test_scores",
        "importances",
        "model_params",
    ]
    result_database = easyDatabase(keynames)
    for k in range(len(total_outs)):
        keys = [total_outs[k][key] for key in keynames]
        value = {key: total_outs[k][key] for key in result_valuekeys}
        result_database.add_member(keynames, keys, value)
    return {
        "results": result_database,
        "split": split_database,
        "train-test": train_test_database,
    }


if __name__ == "__main__":
    out = main(
        "combined_data.pkl",
        40,
    )
    with open("newest_combined_data.pkl", "wb") as f:
        pickle.dump(out, f)
