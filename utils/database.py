from collections import defaultdict
import numpy as np
import pickle


class easyMember:
    def __init__(self, keynames):
        if not np.all([isinstance(k, str) for k in keynames]):
            raise TypeError(f"{keynames} is not a list of string.")
        self._keynames = keynames
        self.keys = (None for i in range(len(keynames)))
        self.value = None

    def assign_keys(self, keys):
        if not isinstance(keys, list):
            raise TypeError("Keys should be in a list.")

        self.keys = keys

    def assign_value(self, value):
        self.value = value

    def get_keynames(self):
        return self._keynames


class easyDatabase:
    def __init__(self, *keynames):
        if not np.all([isinstance(k, str) for k in keynames]):
            raise TypeError(f"{keynames} is not a list of string.")
        self._keynames = keynames
        self.members = []
        self.list_of_keys = []

    def _update_list_of_keys(self):
        self.list_of_keys = []
        for i, member in enumerate(self.members):
            self.list_of_keys.append(member.keys)

    def _append_list_of_keys(self, member):
        self.list_of_keys.append(member.keys)

    def _add_member(self, member: easyMember):
        self.members.append(member)

    def _create_member(self, value, keys_dict):
        sorted_keys = sorted(
            keys_dict.items(),
            key=lambda x: self._keynames.index(x[0]),
        )
        keys = [key for keyname, key in sorted_keys]
        member = easyMember(self._keynames)
        member.assign_keys(keys)
        member.assign_value(value)
        return member

    def add(self, value, **keys_dict):
        if not set(keys_dict.keys()) == set(self._keynames):
            raise KeyError(
                f"f{keys_dict.keys()} doesn't match with {self._keynames}"
            )
        member = self._create_member(value, keys_dict)
        if member.keys in self.list_of_keys:
            raise KeyError(f"f{keys_dict.values()} already exists")
        self._append_list_of_keys(member)
        self._add_member(member)

    def filter_data(self, keys_dict: dict):
        self._update_list_of_keys()
        sorted_keys = sorted(
            keys_dict.items(), key=lambda x: self._keynames.index(x[0])
        )
        keys = [key for keyname, key in sorted_keys]
        indx = list(
            filter(
                lambda ind: self.members[ind].keys == keys,
                range(len(self.members)),
            )
        )
        if len(indx) == 0:
            raise KeyError(f"Unable to find {keys} in the database.")
        return self.members[indx[0]]

    def has_duplicate(self):
        seen = []
        for member in self.members:
            if any(member == s for s in seen):
                raise (f"{member} is duplicated in the database.")
            seen.append(member)
        print("No duplicate items found in the database.")


if __name__ == "__main__":
    results_path = "../results-40.pkl"
    with open(results_path, "rb") as f:
        results = pickle.load(f)

    keynames = ["element", "target", "features"]
    valuenames = ["test_scores", "importances"]

    results_database = easyDatabase("element", "target", "features")
    for i in range(5):
        print(
            {
                key: value
                for key, value in results[i].items()
                if key in keynames
            }
        )
        scores = np.mean(results[i]["test_scores"])
        print(np.mean(scores))

    for i in range(len(results)):
        keys_dict = {
            key: value for key, value in results[i].items() if key in keynames
        }
        value = {
            "score": np.mean(results[i]["test_scores"]),
            "error": np.std(results[i]["test_scores"]),
            "importances": results[i]["importances"],
        }
        results_database.add(value, **keys_dict)

    results_database.has_duplicate()
    filename = "results-40-in-database.pkl"
    with open(filename, "wb") as f:
        pickle.dump(results_database, f)
