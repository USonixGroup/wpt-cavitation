from collections import defaultdict
from time import time
import numpy as np

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn import metrics

evaluations = []
evaluations_std = []


def train_k_means(
    wpt_df, classifying_metrics, *, no_clusters=None, use_pca=False, kmeans_params=None
):
    """
    Trains a k-means algorithm based on a set of training data given by wpt_df and classifying metrics
    no_clusters can be kept at None to choose an optimal number of clusters by silhouette score.
    use_pca can be set to True to apply pca to the normalised data set before training the algorithm.

    """
    if kmeans_params is None:
        raise ValueError("no kmeans parameters specified")

    data_train = wpt_df[classifying_metrics]

    scaler = StandardScaler()
    data_train_normalised = scaler.fit_transform(data_train)

    if use_pca is True:
        pca = PCA(n_components=3)
        data_train_normalised = pca.fit_transform(data_train_normalised)

    if no_clusters is not None:
        kmeans = KMeans(n_clusters=no_clusters, **kmeans_params)
        labels = kmeans.fit_predict(data_train_normalised)
    else:
        # Find optimal numbemr of clusters using silhouette score
        algorithms = []
        for no_clusters in range(3, 6):
            kmeans = KMeans(n_clusters=no_clusters, **kmeans_params)
            labels = kmeans.fit_predict(data_train_normalised)
            silhouette_score = metrics.silhouette_score(data_train_normalised, labels)
            algorithms.append((silhouette_score, kmeans, labels))
        i = max(algorithms, key=lambda x: x[0])
        _, kmeans, labels = i

    centers = kmeans.cluster_centers_
    if use_pca is False:
        centers = scaler.inverse_transform(centers)

        return data_train, labels, data_train_normalised, centers, scaler, kmeans
    if use_pca is True:
        centers = scaler.inverse_transform(pca.inverse_transform(centers))
        return data_train, labels, data_train_normalised, centers, scaler, kmeans, pca


def fit_and_evaluate(km, X, labels, name=None, n_runs=5):
    name = km.__class__.__name__ if name is None else name

    train_times = []
    scores = defaultdict(list)
    for seed in range(n_runs):
        km.set_params(random_state=seed)
        t0 = time()
        km.fit(X)
        train_times.append(time() - t0)
        scores["Homogeneity"].append(metrics.homogeneity_score(labels, km.labels_))
        scores["Completeness"].append(metrics.completeness_score(labels, km.labels_))
        scores["V-measure"].append(metrics.v_measure_score(labels, km.labels_))
        scores["Adjusted Rand-Index"].append(
            metrics.adjusted_rand_score(labels, km.labels_)
        )
        scores["Silhouette Coefficient"].append(
            metrics.silhouette_score(X, km.labels_, sample_size=2000)
        )
    train_times = np.asarray(train_times)

    print(f"clustering done in {train_times.mean():.2f} ± {train_times.std():.2f} s ")
    evaluation = {
        "estimator": name,
        "train_time": train_times.mean(),
    }
    evaluation_std = {
        "estimator": name,
        "train_time": train_times.std(),
    }
    for score_name, score_values in scores.items():
        mean_score, std_score = np.mean(score_values), np.std(score_values)
        print(f"{score_name}: {mean_score:.3f} ± {std_score:.3f}")
        evaluation[score_name] = mean_score
        evaluation_std[score_name] = std_score
    evaluations.append(evaluation)
    evaluations_std.append(evaluation_std)


def assess_parameter(
    parameter_of_interest,
    values,
    wpt_df,
    classifying_metrics,
    no_clusters=None,
    use_pca=False,
):
    kmeans_params = {
        "init": "k-means++",
        "n_init": 10,
        "max_iter": 100,
        "tol": 1e-8,
        "verbose": 0,
        "algorithm": "lloyd",
    }

    # Get labels to compare against
    _, ground_truth, *_ = train_k_means(
        wpt_df,
        classifying_metrics,
        no_clusters=no_clusters,
        kmeans_params=kmeans_params,
    )

    rand_indices = []
    for value in values:
        # Set parameter of interest
        kmeans_params[parameter_of_interest] = value

        # Train k-means
        _, labels, *_ = train_k_means(
            wpt_df,
            classifying_metrics,
            no_clusters=None,
            use_pca=use_pca,
            kmeans_params=kmeans_params,
        )

        rand_index = metrics.adjusted_rand_score(labels, ground_truth)
        rand_indices.append(rand_index)

    return rand_indices
