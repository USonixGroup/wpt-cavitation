import os
import pickle

FILE_PATH = os.path.abspath(__file__)
RELATIVE_PATH = "../../../notebooks/kmeans_results"
FOLDER_PATH = os.path.abspath(os.path.join(FILE_PATH, RELATIVE_PATH))


def get_data(result_id):
    file_path = os.path.join(FOLDER_PATH, str(result_id) + ".pkl")
    with open(file_path, "rb") as file:
        data = pickle.load(file)
        results = tuple(data.values())

    return results


def save_data(
    result_id,
    data_train,
    tr_obj,
    centers,
    labels,
    sampling_rate=None,
    freq_threshold=None,
):
    file_path = os.path.join(FOLDER_PATH, str(result_id) + ".pkl")
    with open(file_path, "wb") as file:
        data = {
            "data": data_train,
            "tr_obj": tr_obj,
            "centers": centers,
            "labels": labels,
            "sampling rate": sampling_rate,
            "freq_threshold": freq_threshold,
        }
        pickle.dump(data, file)
