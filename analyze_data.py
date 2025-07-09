from conf import *
from utils import *

elements = ['Ti', 'Cu', 'Fe', 'Mn']
i=0
results = bunch_load_pickle(*results_path)[i]
plt.style.use('seaborn-v0_8')

plot_features = [
    ['xanes', 'x_pdf', 'n_pdf'],
    ['xanes', 'x_pdf', 'nx_pdf'],
    ['xanes', 'n_pdf', 'nx_pdf'],
]

baseline_features = [
    ['xanes', 'diff_x_pdf'],
]

## Get trivial classifier
trivial_scores = []
trivial_dump_path = 'active-data-house/trivial-scores.pkl'
fea_tar_model = generate_fea_tar_model(feature, target)
for i, rseed in enumerate(random_seeds):
    with open(data_path[i], 'rb') as f:
        train_test_data = pickle.load(f)
    for j, ele in enumerate(elements):
        for k in range(len(fea_tar_model)):
            fea = fea_tar_model[k]['features']
            tar = fea_tar_model[k]['target']
            if tar not in ['cs', 'cn']:
                 continue
            try:
                X_train, X_test, y_train, y_test = lookup(train_test_data, 'train_test', fea, tar, ele)
            except IndexError:
                print(fea, tar, ele)
                labels, counts = np.unique(y_test, return_counts=True)
                y_trivial_pred = np.ones(len(y_test)) * labels[np.argmax(counts)]
                score = f1_score(y_test, y_trivial_pred, average='weighted')
                
            tmp = {}
            tmp['element'] = ele
            tmp['features'] = fea
            tmp['target'] = tar
            tmp['trivial_score'] = score
            trivial_scores.append(tmp)

with open(trivial_dump_path, 'wb') as f:
    pickle.dump(trivial_scores, f)

                
