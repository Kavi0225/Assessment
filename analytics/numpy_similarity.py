from __future__ import annotations
import numpy as np
import pandas as pd

class SimilarityEngine:
    

    # 1. Cosine Similarity

    @staticmethod
    def cosine_similarity(A: np.ndarray, B: np.ndarray) -> np.ndarray:

        A_norm = np.linalg.norm(A, axis=1, keepdims=True)
        B_norm = np.linalg.norm(B, axis=1, keepdims=True)

        # Avoid division by zero
        A_norm[A_norm == 0] = 1e-10
        B_norm[B_norm == 0] = 1e-10

        similarity = (A @ B.T) / (A_norm @ B_norm.T)

        return similarity

    # 2. Euclidean Distance (Broadcasting)

    @staticmethod
    def euclidean_distance(A: np.ndarray, B: np.ndarray) -> np.ndarray:
        
        A_sq = np.sum(A**2, axis=1).reshape(-1, 1)
        B_sq = np.sum(B**2, axis=1).reshape(1, -1)

        distances = np.sqrt(A_sq + B_sq - 2 * (A @ B.T))

        return distances

    # 3. Pearson Correlation Matrix

    @staticmethod
    def pearson_correlation(X: np.ndarray) -> np.ndarray:
        
        X_centered = X - np.mean(X, axis=0)
        covariance = (X_centered.T @ X_centered) / (X.shape[0] - 1)

        std_dev = np.std(X, axis=0, ddof=1)
        std_matrix = np.outer(std_dev, std_dev)

        correlation = covariance / std_matrix

        return correlation

    # 4. Customer Similarity Feature Matrix

    @staticmethod
    def build_customer_feature_matrix(
        total_spend: np.ndarray,
        avg_quantity: np.ndarray,
        tenure: np.ndarray,
        purchase_freq: np.ndarray,
    ) -> np.ndarray:
        
        features = np.column_stack(
            [total_spend, avg_quantity, tenure, purchase_freq]
        )

        # Standardization (important for similarity)
        mean = np.mean(features, axis=0)
        std = np.std(features, axis=0)

        std[std == 0] = 1e-10

        return (features - mean) / std

    # 5. Pairwise Customer Similarity

    @staticmethod
    def customer_similarity(feature_matrix: np.ndarray) -> np.ndarray:
        
        norms = np.linalg.norm(feature_matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1e-10

        normalized = feature_matrix / norms

        return normalized @ normalized.T
    

# if __name__ == "__main__":
    
#     df = pd.read_csv(
#         "C:\\Users\\demon\\OneDrive\\Desktop\\Assessment\\test_data\\sample_data.csv",
#         parse_dates=["timestamp"]
#     )

#     print("Preparing customer features...")

#     # Aggregate customer-level features

#     customer_df = df.groupby("customer_id").agg(
#         total_spend=("total_amount", "sum"),
#         avg_quantity=("quantity", "mean"),
#         tenure=("customer_tenure_days", "max"),
#         purchase_freq=("transaction_id", "count"),
#     ).reset_index()

#     # Convert to numpy arrays
#     total_spend = customer_df["total_spend"].values
#     avg_quantity = customer_df["avg_quantity"].values
#     tenure = customer_df["tenure"].values
#     purchase_freq = customer_df["purchase_freq"].values

#     # Build Feature Matrix

#     feature_matrix = SimilarityEngine.build_customer_feature_matrix(
#         total_spend,
#         avg_quantity,
#         tenure,
#         purchase_freq
#     )

#     print("Feature matrix shape:", feature_matrix.shape)

#     # Compute Similarity

#     similarity_matrix = SimilarityEngine.customer_similarity(feature_matrix)

#     print("Similarity matrix shape:", similarity_matrix.shape)

#     # Show sample similarity scores
#     print("\nSample similarity scores for first customer:")
#     print(similarity_matrix[0][:10])

