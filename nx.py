import datetime
import random
from conf import *
from utils import *

tune_hyper = True
# results_path = 'empty_pickle.pkl'

n_estimators = [25, 50, 100, 200, 300]
max_features = [10, 15, 20, 25, 30, 35]
default_model_params = {
    'n_estimators': n_estimators,
    'max_features': max_features
}


def update_results():
    """
    Add feature to already trained results. After 'features' is changed in the conf.py

    train-test-data is changed because new features in the X_train and X_test
    results-data is changed because new resuts.
    """
    for i in range(len(random_seeds)):
        train_test_data, track_data = build_train_test_data(elements, load_path, features, target, int(random_seeds[i]))
        with open(data_path[i].name, 'wb') as f:
            pickle.dump(train_test_data, f)
        # with open(data_path[i], 'rb') as f:
            # train_test_data = pickle.load(f)

        fea_tar_model = generate_fea_tar_model(features, target)
        total_start = time.time()
        total_outs = []

        for j in range(len(elements)):
            print("Start on element {}".format(elements[j]))
            for k in range(len(fea_tar_model)):

                round_start = time.time()
                print("Start training for \n\tFeatures:{}\n\tTarget:{}".format(fea_tar_model[k]['features'], fea_tar_model[k]['target']))
                dict = {
                    'element': elements[j],
                    'features': fea_tar_model[k]['features'],
                    'target': fea_tar_model[k]['target']
                }

                fea = fea_tar_model[k]['features']
                tar = fea_tar_model[k]['target']
                ele = elements[j]

                X_train, X_test, y_train, y_test = lookup(train_test_data, 'train_test', fea, tar, ele)

                model_params, findit = lookup_model_params(results_path[i], fea, tar, ele)
                try:
                    model_params = {key:model_params[key] for key, val in default_model_params.items()}
                except KeyError:
                    findit=False
                    
                # # findit = False  # Retrain
                if findit:
                    print("Use trained parameters. Params: {}".format(model_params))
                    fea_tar_model[k]['model_params'] = model_params
                else:
                    fea_tar_model[k]['model_params'] = default_model_params

                print(fea_tar_model[k]['model_params'])
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
                out['train_scores'] = train_outcome[0][0]
                out['test_scores'] = train_outcome[0][1]
                out['importances'] = train_outcome[1]
                out['model_params'] = train_outcome[2]
                if len(train_outcome) == 4:
                    out['y_pred'] = train_outcome[3]
                out['element'] = ele
                out['features'] = fea
                out['target'] = tar

                total_outs.append(out)
                print("{}/{} round finished, cost {} seconds in this round.".format(
                    k+j*len(fea_tar_model)+i*len(results_path), len(fea_tar_model)*len(elements)*len(results_path), time.time()-round_start)
                    )
            print('\n\n') 

        print("Total {} seconds".format(time.time() - total_start))
        with open(results_path[i].name, 'wb') as f:
            pickle.dump(total_outs, f)




if __name__ == "__main__":
    # for i in range(8):
        # print(f"Run {i+1} of 8")
        # main()

    # print(load_path)
    update_results()
    # pass