import unittest
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

from wavelet_transform.utils.machine_learning import fit_k_means


class TestFitKMeans(unittest.TestCase):
    def setUp(self):
        np.random.seed(0)
        self.df = pd.DataFrame({
            "A": np.random.rand(10),
            "B": np.random.rand(10),
            "C": np.random.rand(10),
        })

    def test_basic_kmeans(self):
        """Test normal KMeans fitting with an integer number of clusters."""
        result = fit_k_means(
            self.df,
            ["A", "B", "C"],
            2,
            kmeans_params={"n_init": 5},
        )

        data_train, labels, normed, centers_df, scaler, kmeans = result

        # Basic structural checks
        self.assertIsInstance(data_train, pd.DataFrame)
        self.assertEqual(len(labels), len(self.df))
        self.assertEqual(centers_df.shape[1], 3)
        self.assertIsInstance(scaler, StandardScaler)
        self.assertIsInstance(kmeans, KMeans)

    def test_auto_cluster_selection(self):
        """Test tuple range behaviour for silhouette scoring."""
        result = fit_k_means(
            self.df,
            ["A", "B", "C"],
            (2, 4),
            kmeans_params={"n_init": 5},
        )

        data_train, labels, normed, centers_df, scaler, kmeans = result

        self.assertIn(len(np.unique(labels)), [2, 3, 4])
        self.assertEqual(centers_df.shape[1], 3)

    def test_detects_correct_number_of_clusters(self):
        """Test if algorithm identifies the correct number and centers of clusters on clear data."""

        # Make 3 distinct clusters in 2D with known centers
        centers = np.array([[0, 0], [4, 4], [-4, 4]])  # Define specific centers
        X = make_blobs(n_samples=150, centers=centers, 
                      cluster_std=0.5, random_state=42)[0]  # Only take the samples
        df = pd.DataFrame(X, columns=["x", "y"])
        true_centers = centers  # We know the centers directly

        result = fit_k_means(
            df,
            ["x", "y"],
            (2, 5),  # Let it choose between 2–4 clusters
            kmeans_params={"n_init": 10, "random_state": 42},  # Increased n_init for better convergence
        )

        _, labels, _, centers_df, _, kmeans = result

        # Check if the chosen number of clusters matches the true structure
        self.assertEqual(len(np.unique(labels)), 3)

        # Test if cluster centers match the blob centers (allowing for some numerical tolerance)
        # Note: Centers may not be in the same order as the original blobs
        found_centers = centers_df.to_numpy()
        
        # For each true center, check if there's a matching found center within tolerance
        for true_center in true_centers:
            # Calculate distances to all found centers
            distances = np.linalg.norm(found_centers - true_center, axis=1)
            min_distance = np.min(distances)
            self.assertLess(min_distance, 0.1, 
                          "No matching center found within tolerance")

    def test_invalid_kmeans_params(self):
        """Test missing kmeans_params raises ValueError."""
        with self.assertRaises(ValueError):
            fit_k_means(self.df, ["A"], 2)

    def test_invalid_no_clusters_type(self):
        """Test invalid no_clusters type raises ValueError."""
        with self.assertRaises(ValueError):
            fit_k_means(self.df, ["A"], None, kmeans_params={"n_init": 5})


if __name__ == "__main__":
    unittest.main()
