import os
import pickle

FILE_PATH = os.path.abspath(__file__)
RELATIVE_PATH = "../../../notebooks/kmeans_results"
FOLDER_PATH = os.path.abspath(os.path.join(FILE_PATH, RELATIVE_PATH))


# store db object as well
def get_data(result_id):
    """
    Get the stored data from the experiment.
    Additional information about the parameters used for the wavelet transform can be found in
    tr_obj.parameters
    """
    file_path = os.path.join(FOLDER_PATH, str(result_id) + ".pkl")
    with open(file_path, "rb") as file:
        data = pickle.load(file)
        results = data

    return results


def save_data(
    result_id,
    data_train,
    tr_obj,
    centers,
    labels,
    sampling_rate=None,
    freq_threshold=1.,
    **kwargs
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
        data.update(kwargs)
        pickle.dump(data, file)
