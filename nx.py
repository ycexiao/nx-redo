import datetime
import random
from conf import *
from utils.ml import *
from utils.handy import *
from utils.analysis import *

# tune_hyper = True
# results_path = 'empty_pickle.pkl'


def main(
    read_path,
    write_results_path,
    write_data_path,
    write_track_path,
    random_seed=40,
):
    """

    Perform the training using configurations from `conf.py`.

    train-test-data is changed because new features in the X_train and X_test
    results-data is changed because new resuts.
    """
    train_test_data, track_data = build_train_test_data(
        elements, load_path, features, target, random_seed
    )
    with open(write_data_path, "wb") as f:
        pickle.dump(train_test_data, f)  # with open(data_path[i], 'rb') as f:
    with open(write_track_path, "wb") as f:
        pickle.dump(track_data, f)  # with open(data_path[i], 'rb') as f:

    fea_tar_model = generate_fea_tar_model(features, target)
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
            dict = {
                "element": elements[j],
                "features": fea_tar_model[k]["features"],
                "target": fea_tar_model[k]["target"],
            }

            fea = fea_tar_model[k]["features"]
            tar = fea_tar_model[k]["target"]
            ele = elements[j]

            X_train, X_test, y_train, y_test = lookup(
                train_test_data, "train_test", fea, tar, ele
            )

            try:
                model_params, findit = lookup_model_params(
                    read_path, fea, tar, ele
                )
                model_params = {
                    key: model_params[key]
                    for key, val in default_model_params.items()
                }
            except:
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
                tune_hyper = (
                    not findit
                )  # if no find the data, run gridsearchcv.
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
    with open(write_results_path, "wb") as f:
        pickle.dump(total_outs, f)


if __name__ == "__main__":
    random_seed = 40
    train_test_data, track_data = build_train_test_data(
        elements, load_path, features, target, random_seed
    )
    print(train_test_data[0].keys())
    keynames = ["element", "features", "target"]
    train_test_database = easyDatabase(*keynames)
    for i in range(len(train_test_data)):
        key_dict = {key: train_test_data[i][key] for key in keynames}
        _, _, y_train, y_test = train_test_data[i]["train_test"]
        values = {"y_train": y_train, "y_test": y_test}
        train_test_database.add(values, **key_dict)

    search_dict = {"element": "Cu", "features": ["xanes"], "target": "cn"}
    out = train_test_database.filter_data(search_dict)
    print(out.value["y_train"])

    # main(
#     None,
#     "results-40.pkl",
#     "train-test-data-40.pkl",
#     "data-split-track-40.pkl",
#     40,
# )
