import os
import time
import pickle
from ml import build_train_test_data, generate_fea_tar_model, train_model
from helper import resultDatabase
import copy

elements = ["Ti", "Cu", "Fe", "Mn"]
file_names = [element + "_collection.json" for element in elements]
load_dir = "datasets"  # (Na Narong et al., 2025)
load_path = [
    os.path.join(load_dir, file_names[i]) for i in range(len(file_names))
]
targets = ["cs", "cn", "bl", "bl_2nd"]
# features = [
#     ["x_pdf", "n_pdf"],
#     ["nx_pdf"],
#     ["xanes", "x_pdf", "n_pdf"],
#     ["xanes", "nx_pdf"],
#     ["xanes"],
#     ["x_pdf"],
#     ["n_pdf"],
#     ["xanes", "x_pdf"],
#     ["xanes", "n_pdf"],
#     ["xanes", "diff_x_pdf"],
# ]
features = [["x_pdf"]]
rf_model_params = {
    "n_estimators": [25, 50, 100, 200, 300],
    "max_features": [10, 15, 20, 25, 30, 35],
}

knn_model_params = {
    "n_neighbors": [5, 15, 25, 35, 45, 55, 65, 75],
    "weights": ["uniform", "distance"],
}


def main(trained_database, random_seed=40, model_type="rf"):
    keynames = ["target", "element", "features"]
    fea_tar_model = generate_fea_tar_model(
        features, targets, model_type=model_type
    )
    if model_type == "rf":
        default_model_params = rf_model_params
    elif model_type == "knn":
        default_model_params = knn_model_params
    total_start = time.time()
    total_outs = []
    for i in range(len(elements)):
        print("Start on element {}".format(elements[i]))
        for j in range(len(fea_tar_model)):
            fea = fea_tar_model[j]["features"]
            tar = fea_tar_model[j]["target"]
            ele = elements[i]
            X_train, X_test, y_train, y_test = build_train_test_data(
                load_path[i], fea, tar, random_seed=random_seed
            )
            # Find hyper parameters
            round_start = time.time()
            print(
                "Start training for \n\tFeatures:{}\n\tTarget:{}".format(
                    fea, tar
                )
            )
            try:
                model_params = trained_database.filter_data(
                    ["element", "features", "target", "model_type"], [ele, fea, tar, model_type]
                ).value["model_params"]
                print(
                    "Use trained parameters. Params: {}".format(model_params)
                )
                findit = True
            except KeyError:
                model_params = default_model_params
                print("Hyper-parameters not Found. Run GridSearch.")
                findit = False
            tune_hyper = (
                not findit
            )  # if no find the trained hyper parameters, run gridsearchcv.
            train_outcome = train_model(  # main step
                X_train=X_train,
                X_test=X_test,
                y_train=y_train,
                y_test=y_test,
                tune_hyper=tune_hyper,
                model_params=model_params,
                model_type=model_type,
                model=fea_tar_model[j]["model"],
                score_method=fea_tar_model[j]["score_method"],
            )
            out = copy.deepcopy(fea_tar_model[j])
            out.update(train_outcome)
            out["train_mp_ids"] = X_train.index
            out["test_mp_ids"] = X_test.index
            out["model_type"] = model_type
            out["element"] = ele
            total_outs.append(out)
            print(
                "{}/{} round finished, cost {} seconds in this round.".format(
                    j + i * len(fea_tar_model)+1,
                    len(fea_tar_model) * len(elements),
                    time.time() - round_start,
                )
            )
            print("\n\n")
    print("Total {} seconds".format(time.time() - total_start))
    # Write into database
    keynames = ["target", "element", "features", "model_type"]
    if model_type == "rf":
        result_valuekeys = [
            "train_mp_ids",
            "test_mp_ids",
            "train_scores",
            "test_scores",
            "importances",
            "model_params",
        ]
    elif model_type == "knn":
        result_valuekeys = [
            "train_mp_ids",
            "test_mp_ids",
            "train_scores",
            "test_scores",
            "model_params",
        ]
    result_database = resultDatabase(keynames)
    for k in range(len(total_outs)):
        key_dict = {key: total_outs[k][key] for key in keynames}
        value = {key: total_outs[k][key] for key in result_valuekeys}
        result_database.add_entry(key_dict, value)
    return result_database


if __name__ == "__main__":

    keynames = ["target", "element", "features", "model_type"]
    with open("results/result_database_1754776816.pkl", "rb") as f:
        trained_database = resultDatabase(keynames).from_pkl(pickle.load(f))

    out = main(trained_database, 40, "rf")
    # out.to_pkl()

