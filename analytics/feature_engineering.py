from __future__ import annotations
import pandas as pd
import numpy as np


class FeatureEngineer:

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self._prepare_data()

    # Internal Preparation

    def _prepare_data(self) -> None:
        self.df["timestamp"] = pd.to_datetime(self.df["timestamp"], utc=True)
        self.df.sort_values(["customer_id", "timestamp"], inplace=True)

        # critical fix
        self.df.reset_index(drop=True, inplace=True)

    # 1. Polynomial Features (Degree 2)

    def polynomial_features(self, numeric_cols: list[str]) -> pd.DataFrame:
        """
        Generate degree-2 polynomial features using NumPy.
        """
        data = self.df[numeric_cols].values
        n_features = data.shape[1]

        poly_features = []

        for i in range(n_features):
            for j in range(i, n_features):
                poly_features.append(data[:, i] * data[:, j])

        poly_matrix = np.column_stack(poly_features)

        col_names = [
            f"{numeric_cols[i]}_x_{numeric_cols[j]}"
            for i in range(n_features)
            for j in range(i, n_features)
        ]

        return pd.DataFrame(poly_matrix, columns=col_names, index=self.df.index)

    # 2. Interaction Terms

    def interaction_features(self) -> pd.DataFrame:
        df = self.df

        interaction = pd.DataFrame(index=df.index)

        interaction["price_quantity"] = df["unit_price"] * df["quantity"]
        interaction["tenure_purchase_freq"] = (
            df["customer_tenure_days"] * df["previous_purchases"]
        )

        return interaction

    # 3. Binning Features

    def binned_features(self) -> pd.DataFrame:
        df = self.df.copy()

        bins = pd.DataFrame(index=df.index)

        bins["age_group"] = pd.cut(
            df["customer_age"],
            bins=[0, 25, 40, 60, 120],
            labels=["Young", "Adult", "MidAge", "Senior"],
        )

        bins["spending_tier"] = pd.qcut(
            df["total_amount"],
            q=4,
            labels=["Low", "Medium", "High", "Premium"],
            duplicates="drop",
        )

        bins["session_bucket"] = pd.cut(
            df["session_duration_sec"],
            bins=[0, 60, 300, 900, np.inf],
            labels=["Short", "Medium", "Long", "VeryLong"],
        )

        return bins

    # 4. Lag Features

    def lag_features(self) -> pd.DataFrame:
        df = self.df

        lag_df = pd.DataFrame(index=df.index)

        lag_df["prev_transaction_amount"] = (
            df.groupby("customer_id")["total_amount"].shift(1)
        )

        lag_df["days_since_last_purchase"] = (
            df.groupby("customer_id")["timestamp"]
            .diff()
            .dt.days
        )

        return lag_df

    # 5. Rolling Behavioral Features

    def rolling_features(self) -> pd.DataFrame:
        df = self.df.copy()

        g = df.groupby("customer_id", group_keys=False)

        spend = (
            g.rolling("30D", on="timestamp")["total_amount"]
            .sum()
            .reset_index(level=0, drop=True)
            .to_numpy()
        )

        count = (
            g.rolling("30D", on="timestamp")["transaction_id"]
            .count()
            .reset_index(level=0, drop=True)
            .to_numpy()
        )

        rate = (
            g.rolling("30D", on="timestamp")["is_returned"]
            .mean()
            .reset_index(level=0, drop=True)
            .to_numpy()
        )

        return pd.DataFrame(
            {
                "rolling_30d_spend": spend,
                "rolling_purchase_count": count,
                "rolling_return_rate": rate,
            },
            index=df.index,
        )



    # Master Feature Builder

    def build_all_features(self) -> pd.DataFrame:
        """
        Combines all engineered features into one dataframe.
        """
        numeric_cols = [
            "quantity",
            "unit_price",
            "total_amount",
            "customer_tenure_days",
        ]

        poly = self.polynomial_features(numeric_cols)
        interaction = self.interaction_features()
        bins = self.binned_features()
        lag = self.lag_features()
        rolling = self.rolling_features()

        return pd.concat([poly, interaction, bins, lag, rolling], axis=1)


if __name__ == "__main__":
    import pandas as pd

    df = pd.read_csv("C:\\Users\\demon\\OneDrive\\Desktop\\Assessment\\test_data\\sample_data.csv", parse_dates=["timestamp"])
    
    engineer = FeatureEngineer(df)
    
    print("Generating features...")
    new_df = engineer.build_all_features()

    print("\nNew columns added:")
    print(set(new_df.columns) - set(df.columns))

    print("\nSample rows:")
    print(new_df.head())
