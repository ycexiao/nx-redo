from pymatgen.analysis.local_env import CrystalNN
from pymatgen.core.structure import Structure
import os
from pathlib import Path
import json
import numpy as np
import time

cnn = CrystalNN()


def get_coord_shell(structure_dict, element, shell, num_decimals=15):
    """Extract shell radius

    Parameters
    ----------
    structure_dict : dict
        material structure dictionary.
    element: str
        the element name whose shell radius  will be extracted.
    shell : int
        Which neighbor shell to retrieve (1 == 1st NN shell).
    num_decimals : int
        Default 12. Same with 1st bl.
    """

    structure = Structure.from_dict(structure_dict)
    # get site indices of the transition metal
    metal_sites = list(structure.indices_from_symbol(element))
    distances_sites = []
    for site in metal_sites:
        xyz_origin = structure.cart_coords[site]
        neighbors = cnn.get_nn_shell_info(structure, site, shell=shell)
        # species = [neighbor["site"].species_string for neighbor in neighbors]
        # xyz_array = [
        #     np.round(neighbor["site"].coords - xyz_origin, num_decimals)
        #     for neighbor in neighbors
        # ]
        distance = [
            np.round(
                np.linalg.norm(neighbor["site"].coords - xyz_origin),
                num_decimals,
            )
            for neighbor in neighbors
        ]
        distances_sites.extend(distance)

    mean_distance = np.round(np.mean(distances_sites), num_decimals)
    return mean_distance


if __name__ == "__main__":
    # elements = ["Ti", "Fe", "Mn", "Cu"]
    # file_names = [
    #     element + "_collection.json" for element in elements
    # ]  # filename or path to the collection
    # read_dataset_path = "datasets"
    # read_files_path = [
    #     os.path.join(read_dataset_path, file_names[i])
    #     for i in range(len(file_names))
    # ]
    # dump_dataset_path = "datasets_with_secbl"
    # dump_files_path = [
    #     os.path.join(dump_dataset_path, file_names[i])
    #     for i in range(len(file_names))
    # ]

    # start_time = time.time()
    # for i in range(len(read_files_path)):
    #     print(f"Begin for {elements[i]}.")
    #     file_path = Path(read_files_path[i])
    #     mid1_time = time.time()
    #     with open(file_path, "r") as f:
    #         docs = json.load(f)
    #         for j in range(len(docs)):
    #             tmp_doc = docs[j]
    #             tmp_stru_dict = tmp_doc["structure"]
    #             distance_2nd = get_coord_shell(tmp_stru_dict, elements[i], 2)
    #             docs[j]["bl_2nd"] = distance_2nd
    #     mid2_time = time.time()
    #     print(f"Finished for {elements[i]}. Caused {mid2_time-mid1_time}")
    #     dump_file_path = Path(dump_files_path[i])
    #     dump_file_path.parent.mkdir(parents=True, exist_ok=True)
    #     with open(dump_file_path, "w") as f:
    #         json.dump(docs, f)
    # end_time = time.time()
    # 0.585s per structure
    # roughly 4000 structures
    # estimated to be 40 min
    # print(f"Totoal time consuming: {np.round(end_time-start_time, 3)}s")
    # Compilation finished at Sun Aug  3 20:51:27, duration 1:23:55
    pass
