from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List, Optional
import pandas as pd
import numpy as np


class BasePreprocessor(ABC):
    """Abstract base class for all preprocessors."""

    @abstractmethod
    def fit(self, df: pd.DataFrame) -> "BasePreprocessor":
        pass

    @abstractmethod
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        pass

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        self.fit(df)
        return self.transform(df)


class MissingValueHandler(BasePreprocessor):
    """Handles missing values using median/mode strategy."""

    def __init__(self):
        self.fill_values: dict = {}

    def fit(self, df: pd.DataFrame) -> "MissingValueHandler":
        for col in df.columns:
            if df[col].dtype in ["float64", "int64"]:
                self.fill_values[col] = df[col].median()
            else:
                self.fill_values[col] = df[col].mode()[0]
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.fillna(self.fill_values)


class OutlierHandler(BasePreprocessor):
    """Removes outliers using IQR method."""

    def __init__(self, columns: Optional[List[str]] = None):
        self.columns = columns
        self.bounds = {}

    def fit(self, df: pd.DataFrame) -> "OutlierHandler":
        cols = self.columns or df.select_dtypes(include=np.number).columns
        for col in cols:
            q1, q3 = df[col].quantile([0.25, 0.75])
            iqr = q3 - q1
            self.bounds[col] = (q1 - 1.5 * iqr, q3 + 1.5 * iqr)
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df_copy = df.copy()
        for col, (low, high) in self.bounds.items():
            df_copy[col] = df_copy[col].clip(low, high)
        return df_copy


class Normalizer(BasePreprocessor):
    """Min-Max normalization."""

    def __init__(self, columns: Optional[List[str]] = None):
        self.columns = columns
        self.min_vals = {}
        self.max_vals = {}

    def fit(self, df: pd.DataFrame) -> "Normalizer":
        cols = self.columns or df.select_dtypes(include=np.number).columns
        for col in cols:
            self.min_vals[col] = df[col].min()
            self.max_vals[col] = df[col].max()
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df_copy = df.copy()
        for col in self.min_vals:
            denom = self.max_vals[col] - self.min_vals[col]
            if denom != 0:
                df_copy[col] = (df_copy[col] - self.min_vals[col]) / denom
        return df_copy

class DataCleaner(BasePreprocessor):
    """
    Removes invalid business records before transformations.
    """

    def fit(self, df: pd.DataFrame) -> "DataCleaner":
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        initial_rows = len(df)

        # Remove duplicate transactions
        df = df.drop_duplicates(subset=["transaction_id"])

        # Remove impossible ages
        df = df[df["customer_age"] <= 100]

        # Remove negative transaction amounts (refunds)
        df = df[df["total_amount"] >= 0]

        removed = initial_rows - len(df)
        print(f"DataCleaner removed {removed} invalid records")

        return df

class Pipeline:
    """Chains multiple preprocessors sequentially."""

    def __init__(self, steps: List[BasePreprocessor]):
        self.steps = steps

    def fit(self, df: pd.DataFrame) -> "Pipeline":
        for step in self.steps:
            df = step.fit_transform(df)
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        for step in self.steps:
            df = step.transform(df)
        return df

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)

if __name__ == "__main__":
    import pandas as pd

    print("Loading dataset...")
    df = pd.read_csv("C:\\Users\\demon\\OneDrive\\Desktop\\Assessment\\test_data\\sample_data.csv", parse_dates=["timestamp"])

    processor = Pipeline(steps=[
        DataCleaner(),
        MissingValueHandler(),
        OutlierHandler(columns=["quantity", "unit_price", "total_amount"]),
        Normalizer(columns=["quantity", "unit_price", "total_amount"]),
    ])

    print("Running preprocessing pipeline...")
    clean_df = processor.fit_transform(df)

    print("Original rows:", len(df))
    print("Clean rows:", len(clean_df))
