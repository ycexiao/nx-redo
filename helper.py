import numpy as np
import copy
import time
from matplotlib import pyplot as plt


class easyMember:
    def __init__(self, keynames, keys, value):
        self.keynames = keynames
        self.keys = keys
        self.value = value


class easyDatabase:
    def __init__(self, keynames, description=""):
        self.keynames = keynames
        self.members = []
        self.list_of_keys = []
        self.timestamp = time.time()

    def _update_list_of_keys(self):
        self.list_of_keys = []
        for i, member in enumerate(self.members):
            self.list_of_keys.append(member.keys)

    def add_member(self, keynames, keys, value):
        if keynames != self.keynames:
            raise ValueError(f"f{keynames} doesn't match with {self.keynames}")
        member = easyMember(keynames, keys, value)
        if member.keys in self.list_of_keys:
            raise KeyError(f"f{keys.values()} already exists")
        self.list_of_keys.append(member.keys)
        self.members.append(member)

    def filter_data(self, keynames, keys):
        actual_keys = []
        try:
            for keyname in self.keynames:
                actual_keys.append(keys[keynames.index(keyname)])
        except ValueError:
            raise ValueError(f"{keynames} doesn't match with {self.keynames}")
        self._update_list_of_keys()
        indx = list(
            filter(
                lambda ind: self.members[ind].keys == actual_keys,
                range(len(self.members)),
            )
        )
        if len(indx) == 0:
            raise KeyError(f"Unable to find {keys} in the database.")
        return self.members[indx[0]]

    def combine(self, another_databse):
        if self.keynames != another_databse.keynames:
            raise KeyError(
                "The keys are not the same for the two database. "
                f"One is {self.keynames}, and the other one is "
                f"{another_databse.keynames}"
            )
        copied_original_database = copy.deepcopy(self)
        for member in another_databse.members:
            if member.keys not in copied_original_database.list_of_keys:
                copied_original_database.members.append(member)
                copied_original_database.list_of_keys.append(member.keys)
        copied_original_database._update_list_of_keys()
        return copied_original_database

    def modify_keynames(self, new_keynames):
        if len(self.keynames) != len(new_keynames):
            raise KeyError(
                f"The length of the old keynames is {len(self.keynames)}\n"
                f"But the current length is {len(new_keynames)}."
            )
        self.keynames = new_keynames
        for member in self.members:
            member.keynames = new_keynames

    def unique_keys(self):
        self._update_list_of_keys()
        out_keys = [[] for _ in range(len(self.list_of_keys[0]))]
        for i in range(len(self.list_of_keys)):
            for j in range(len(out_keys)):
                out_keys[j].append(self.list_of_keys[i][j])
        out_dict = {
            self.keynames[i]: drop_duplicate_keys(out_keys[i])
            for i in range(len(self.keynames))
        }
        return out_dict


def drop_duplicate_keys(lst):
    set = []
    nlst = copy.deepcopy(lst)
    set.append(nlst.pop(0))
    while nlst:
        item = nlst.pop(0)
        if item not in set:
            set.append(item)
        else:
            continue
    return set


class easyHistory:
    def __init__(self, keynames, description=""):
        self.keynames = keynames
        self.description = description
        self.records = []

    def add_database(self, object):
        if self.keynames != object.keynames:
            raise ValueError(
                f"{object.keynames} doesn't match with {self.keynames}"
            )
        self.records.append(object)

    def print_records(self):
        for item in self.records:
            print(item.timestamp)


class SmartPloter:
    def __init__(self, length):
        self.length = length
        self.metrics = np.empty(length)
        self.errors = np.empty(length)
        self.counter = 0

    def add_with_error(self, *arr):
        if len(arr[0]) != self.length:
            raise ValueError("Added array is of different length")

        if self.counter != 0:
            self.metrics = np.vstack([self.metrics, arr[0]])
            self.errors = np.vstack([self.errors, arr[1]])
            self.counter += 1
        else:
            self.metrics = arr[0]
            self.errors = arr[1]
            self.counter += 1

    def bar(self, ax=None, **plot_params):
        handles = []
        if ax is None:
            fig, ax = plt.subplots()
        metrics = self.metrics
        errors = self.errors
        x0 = np.arange(self.length)
        if self.counter == 1:
            ax.bar(
                x=x0, height=metrics, width=2 / 3, yerr=errors, **plot_params
            )
        else:
            width = 1 / (self.counter + 1)
            for i in range(self.counter):
                x = x0 + i * width
                h = ax.bar(
                    x=x,
                    height=metrics[i],
                    width=width,
                    yerr=errors[i],
                    **plot_params,
                )
                handles.append(h)
                for j in range(len(x)):
                    ax.text(
                        x[j],
                        metrics[i][j],
                        "{:.3f}".format(metrics[i][j]),
                        horizontalalignment="center",
                    )

        return ax, handles


if __name__ == "__main__":
    pass
    # keynames = ["name", "age"]
    # history = easyHistory(keynames, "A test for database with memory")
    # for i in range(10):
    #     database = easyDatabase(keynames, f"Trial {i} databse")
    #     for j in range(10):
    #         keys = [random.randint(1, 1000) for _ in range(2)]
    #         database.add_member(keynames, keys, random.randint(1, 1000))
    #     history.add_database(database)
    # history.print_records()

    # data_path = "../active-data-house"
    # for item in Path(data_path).iterdir():
    #     if item.name.startswith("results"):
    #         keynames = ["target", "element", "features"]
    #         valuekeys = [
    #             "train_scores",
    #             "test_scores",
    #             "importances",
    #             "model_params",
    #         ]
    #         database = easyDatabase(keynames)
    #         with open(item, "rb") as f:
    #             results = pickle.load(f)
    #             for k in range(len(results)):
    #                 keys = [results[k][key] for key in keynames]
    #                 value = {key: results[k][key] for key in valuekeys}
    #                 database.add_member(keynames, keys, value)
    #         break
    # print(database.unique_keys())
