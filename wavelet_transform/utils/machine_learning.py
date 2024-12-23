import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn import metrics

# ---------------------- Kmeans -----------------------

evaluations = []
evaluations_std = []


class noScale:
    "Replacement for standard scaler if no scaling is required"

    def fit_transform(self, data):
        return data

    def inverse_transform(self, data):
        return data


class minScaler:
    "Scales columns according to their minimum value"

    def __init__(self):
        self.min_datas = None  # ndarray tracking minimums for inverse transform

    def fit_transform(self, data):
        data = np.array(data)
        n_columns = data.shape[1]
        self.min_datas = np.zeros(n_columns)
        for i in range(n_columns):
            col_data = data[:, i]
            min_data = np.min(col_data)
            self.min_datas[i] = min_data
            data[:, i] = col_data / min_data
        return data

    def inverse_transform(self, data):
        for i in range(len(self.min_datas)):
            data[:, i] *= self.min_datas[i]
        return data


def train_k_means(
    wpt_df,
    classifying_metrics,
    no_clusters,
    *,
    use_pca=False,
    kmeans_params=None,
    scale_data=True,
):
    """
    Trains a k-means algorithm based on a set of training data given by wpt_df and classifying metrics
    no_clusters can be kept at None to choose an optimal number of clusters by silhouette score.
    use_pca can be set to True to apply pca to the normalised data set before training the algorithm.

    Parameters
    ----------
    wpt_df: pandas.DataFrame
        dataframe with columns containing potential classifying metrics
    classifying_metrics: list of str
        list of classifying metrics to index wpt_df with.
    no_clusters: int or tuple
        The number of clusters to use. specify None to automatically choose the best number of
        clusters using the silhouette score.
    use_pca: bool
        whether to use pca in the analysis
    kmeans_params: dict
        passes parameters into sklearn.KMeans.
    scale_data: bool or str
        set to True to use sklearn.preprocessing.StandardScaler.
        set to False to skip scaling.
        set to 'min' to scale by the minimum value in each metric.
    """
    if kmeans_params is None:
        raise ValueError("no kmeans parameters specified")

    data_train = wpt_df[classifying_metrics]

    scaler = StandardScaler()
    if scale_data is False:
        scaler = noScale()
    elif scale_data == "min":
        scaler = minScaler()

    data_train_normalised = scaler.fit_transform(data_train.to_numpy())

    if use_pca is True:
        pca = PCA(n_components=3)
        data_train_normalised = pca.fit_transform(data_train_normalised)

    if isinstance(no_clusters, int):
        kmeans = KMeans(n_clusters=no_clusters, **kmeans_params)
        labels = kmeans.fit_predict(data_train_normalised)
    elif isinstance(no_clusters, tuple):
        # Find optimal numbemr of clusters using silhouette score
        algorithms = []
        for no_clusters in range(*no_clusters):
            kmeans = KMeans(n_clusters=no_clusters, **kmeans_params)
            labels = kmeans.fit_predict(data_train_normalised)
            silhouette_score = metrics.silhouette_score(data_train_normalised, labels)
            algorithms.append((silhouette_score, kmeans, labels))
        i = max(algorithms, key=lambda x: x[0])
        _, kmeans, labels = i
    else:
        raise ValueError("no_clusters must be an int or tuple")

    centers = kmeans.cluster_centers_
    if use_pca is False:
        centers = scaler.inverse_transform(centers)
        centers_df = pd.DataFrame(centers, columns=classifying_metrics)

        return data_train, labels, data_train_normalised, centers_df, scaler, kmeans
    if use_pca is True:
        centers = scaler.inverse_transform(pca.inverse_transform(centers))
        return data_train, labels, data_train_normalised, centers, scaler, kmeans, pca
