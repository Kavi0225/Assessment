from __future__ import annotations
import pandas as pd


class EcommerceAnalytics:

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self._prepare_datetime()

    # Internal Helpers

    def _prepare_datetime(self) -> None:
        """Ensure timestamp is parsed and timezone normalized."""
        self.df["timestamp"] = pd.to_datetime(self.df["timestamp"], utc=True)
        self.df.sort_values("timestamp", inplace=True)

    # 1. Monthly Revenue by Category + MoM Growth

    def monthly_revenue_growth(self) -> pd.DataFrame:
        df = self.df.copy()

        df["month"] = df["timestamp"].dt.to_period("M")

        revenue = (
            df.groupby(["month", "product_category"])["total_amount"]
            .sum()
            .reset_index()
        )

        revenue["month"] = revenue["month"].astype(str)

        revenue["mom_growth_pct"] = (
            revenue.groupby("product_category")["total_amount"]
            .pct_change() * 100
        )

        return revenue

    # 2. Rolling Avg Transaction Value per Customer

    def rolling_customer_metrics(self) -> pd.DataFrame:
        df = self.df.copy()

        df = df.sort_values(["customer_id", "timestamp"])

        # Set index ONCE for rolling operations
        df = df.set_index("timestamp")

        # 7-day rolling avg
        df["rolling_7d_avg"] = (
            df.groupby("customer_id")["total_amount"]
            .rolling("7D")
            .mean()
            .reset_index(level=0, drop=True)
        )

        # 30-day rolling avg
        df["rolling_30d_avg"] = (
            df.groupby("customer_id")["total_amount"]
            .rolling("30D")
            .mean()
            .reset_index(level=0, drop=True)
        )

        df = df.reset_index()

        return df[["customer_id", "timestamp", "rolling_7d_avg", "rolling_30d_avg"]]



    # 3. revenue Pivot Table

    def revenue_pivot_table(self) -> pd.DataFrame:
        pivot = pd.pivot_table(
            self.df,
            index="customer_region",
            columns=["product_category", "ltv_segment"],
            values="total_amount",
            aggfunc=["sum", "count"],
            fill_value=0,
        )

        return pivot

    # 4. Datetime Feature Extraction

    def extract_datetime_features(self) -> pd.DataFrame:
        df = self.df.copy()

        df["hour"] = df["timestamp"].dt.hour
        df["day_of_week"] = df["timestamp"].dt.dayofweek
        df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
        df["quarter"] = df["timestamp"].dt.quarter

        return df

    # 5. Coefficient of Variation per Customer

    def customer_spend_variability(self) -> pd.DataFrame:
        grouped = self.df.groupby("customer_id")["total_amount"]

        result = grouped.agg(["mean", "std"]).reset_index()

        result["coefficient_of_variation"] = result["std"] / result["mean"]

        return result[["customer_id", "coefficient_of_variation"]]

if __name__ == "__main__":

    print("Loading dataset...")
    df = pd.read_csv(
        "C:\\Users\\demon\\OneDrive\\Desktop\\Assessment\\test_data\\sample_data.csv", #use your path here
        parse_dates=["timestamp"]
    )

    analytics = EcommerceAnalytics(df)

    print("\nMonthly Revenue Growth:")
    print(analytics.monthly_revenue_growth().head()) 

    print("\nRolling Customer Metrics:")
    print(analytics.rolling_customer_metrics().head())

    print("\nRevenue Pivot Table:")
    print(analytics.revenue_pivot_table().head())

    print("\nDatetime Feature Extraction:")
    print(analytics.extract_datetime_features().head())

    print("\nCustomer Spend Variability:")
    print(analytics.customer_spend_variability().head())
