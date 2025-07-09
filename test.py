from helper import *
from conf import *
import pandas as pd
import re

# ---------------------------------------------------------------------------------------
# data management
# ---------------------------------------------------------------------------------------

is_old_results = lambda x: x.count('-')==4
is_new_results = lambda x: x.count('-')==5 and x.startswith('results')
is_old_data = lambda x: x.count('-') == 7 and x.startswith('train-test-data')
is_new_data = lambda x: x.count('-') == 7 and x.startswith('train-test-data')

is_track = lambda x: x.count('-') == 7 and x.startswith('data-split-track')

random_seed_pattern = re.compile(r'(?<=-)(\d+).pkl$')

old_results_path, random_seeds = sort_files(find_files('data-house/no-stratify-datasets', is_old_results), random_seed_pattern, return_keys=True)
new_results_path  = sort_files(find_files('active-data-house', is_new_results), random_seed_pattern)

old_data_path = sort_files(find_files('data-house/no-stratify-datasets', is_old_data), random_seed_pattern)
new_data_path = sort_files(find_files('active-data-house', is_new_data), random_seed_pattern)
# print_list(old_results_path, new_results_path)
# print_list(old_data_path, new_data_path)


# ---------------------------------------------------------------------------------------
# Same random seed different results?
# Same random seed, same training-testing set, different results because of different training strategy "stratify?Not stratify?"
# ----------------------------------------------------------------------------------------

sum_len = lambda x: sum([len(x[i]) for i in range(len(x))])

fea_tar_model = generate_fea_tar_model(feature, target)
for i in range(len(old_results_path)):
    # print(random_seeds[i])
    # print(old_results_path[i])
    # print(new_results_path[i])
    # print(old_data_path[i])
    # print(new_data_path[i])

    data, track = build_train_test_data(elements, load_path, feature, target, int(random_seeds[i]))

    with open(old_results_path[i], 'rb') as f:
        old_results = pickle.load(f)
    with open(new_results_path[i], 'rb') as f:
        new_results = pickle.load(f)

    with open(old_data_path[i], 'rb') as f:
        old_data = pickle.load(f)
    with open(new_data_path[i], 'rb') as f:
        new_data = pickle.load(f)

    for j in range(len(elements)):
        if elements[j] not in ['Fe', 'Mn']:
            continue
        for k in range(len(fea_tar_model)):
            ele = elements[j]
            fea = fea_tar_model[k]['features']
            tar = fea_tar_model[k]['target']
            try:
                old_scores = lookup(old_results, "test_scores", fea, tar, ele)
                new_scores = lookup(new_results, "test_scores", fea, tar, ele)

                old_data_entries = lookup(old_data, "train_test", fea, tar, ele)
                new_data_entries = lookup(new_data, "train_test", fea, tar, ele)
                data_entries = lookup(data, "train_test", fea, tar, ele)
                # print(sum_len(old_data_entries))
                # print(sum_len(new_data_entries))
                # print(sum_len(data_entries))

                # print(np.mean(old_scores))
                # print(old_data_entries[0][0][:5])
                # print(new_data_entries[0][0][:5])
                # print(data_entries[0][0][:5])
                # print(np.mean(new_scores))
                # print('\n')
            except IndexError:
                continue




# ---------------------------------------------------------------------------------------
# Data track test
# ----------------------------------------------------------------------------------------
# seed = 0
# train_test_data, data_split = build_train_test_data(rseed=seed, load_path=load_path, features=features, target=target, elements=elements)

# print(data_split[0].keys())
# print(data_split[0]['test_mp_ids'])
