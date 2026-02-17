"""
Synthetic E-Commerce Dataset Generator

Creates realistic transactional dataset with:
- Proper distributions
- Data quality issues
- Configurable scale
"""

import numpy as np
import pandas as pd
import uuid
from datetime import datetime, timedelta


# CONSTANTS

CATEGORIES = ["Electronics", "Clothing", "Home", "Sports", "Books"]
PAYMENT_METHODS = ["Credit Card", "Debit Card", "PayPal", "Crypto"]
REGIONS = ["North", "South", "East", "West", "Central"]
REFERRALS = ["Organic", "Paid Search", "Social", "Email", "Affiliate"]
LTV_SEGMENTS = ["Low", "Medium", "High", "Premium"]

PRODUCTS = {
    "Electronics": ["Phone", "Laptop", "Headphones", "Tablet"],
    "Clothing": ["Shirt", "Jeans", "Jacket", "Shoes"],
    "Home": ["Mixer", "Sofa", "Lamp", "Cookware"],
    "Sports": ["Football", "Tennis Racket", "Yoga Mat"],
    "Books": ["Novel", "Biography", "Textbook"]
}


# GENERATOR FUNCTION

def generate_dataset(n_rows=100_000, seed=42):
    np.random.seed(seed)

    # Basic Fields
    transaction_ids = [str(uuid.uuid4()) for _ in range(n_rows)]
    customer_ids = [f"CUST_{np.random.randint(1000, 9999)}" for _ in range(n_rows)]

    start_date = datetime(2022, 1, 1)
    timestamps = [
        start_date + timedelta(minutes=np.random.randint(0, 1_000_000))
        for _ in range(n_rows)
    ]

    categories = np.random.choice(CATEGORIES, n_rows)

    product_names = [
        np.random.choice(PRODUCTS[cat]) for cat in categories
    ]

    quantity = np.random.randint(1, 11, n_rows)
    unit_price = np.round(np.random.uniform(5, 2000, n_rows), 2)
    discount_pct = np.round(np.random.uniform(0, 30, n_rows), 2)

    total_amount = quantity * unit_price * (1 - discount_pct / 100)

    payment_method = np.random.choice(PAYMENT_METHODS, n_rows)
    customer_age = np.random.randint(18, 75, n_rows)
    customer_region = np.random.choice(REGIONS, n_rows)

    tenure_days = np.random.randint(1, 2000, n_rows)
    is_returned = np.random.choice([True, False], n_rows, p=[0.1, 0.9])

    satisfaction_score = np.round(np.random.uniform(1, 5, n_rows), 1)

    referral_source = np.random.choice(REFERRALS, n_rows)
    session_duration = np.random.randint(10, 3600, n_rows)
    items_viewed = np.random.randint(1, 50, n_rows)
    previous_purchases = np.random.randint(0, 200, n_rows)

    ltv_segment = np.random.choice(LTV_SEGMENTS, n_rows)

    df = pd.DataFrame({
        "transaction_id": transaction_ids,
        "customer_id": customer_ids,
        "timestamp": timestamps,
        "product_category": categories,
        "product_name": product_names,
        "quantity": quantity,
        "unit_price": unit_price,
        "discount_pct": discount_pct,
        "total_amount": total_amount,
        "payment_method": payment_method,
        "customer_age": customer_age,
        "customer_region": customer_region,
        "customer_tenure_days": tenure_days,
        "is_returned": is_returned,
        "satisfaction_score": satisfaction_score,
        "referral_source": referral_source,
        "session_duration_sec": session_duration,
        "items_viewed": items_viewed,
        "previous_purchases": previous_purchases,
        "ltv_segment": ltv_segment
    })

    # Inject Data Issues

    # Missing satisfaction scores
    missing_idx = np.random.choice(df.index, int(0.15 * n_rows), replace=False)
    df.loc[missing_idx, "satisfaction_score"] = np.nan

    # Negative amounts (refunds)
    refund_idx = np.random.choice(df.index, int(0.02 * n_rows), replace=False)
    df.loc[refund_idx, "total_amount"] *= -1

    # Duplicate transaction IDs
    dup_idx = np.random.choice(df.index, int(0.005 * n_rows), replace=False)
    df.loc[dup_idx, "transaction_id"] = df.loc[dup_idx[0], "transaction_id"]

    # Age outliers
    outlier_idx = np.random.choice(df.index, int(0.01 * n_rows), replace=False)
    df.loc[outlier_idx, "customer_age"] = np.random.randint(101, 120, len(outlier_idx))

    return df


# EXPORT HELPER

def save_sample(df, path="test_data/sample_data.csv"):
    df.to_csv(path, index=False)


if __name__ == "__main__":
    df = generate_dataset()
    save_sample(df)
    print("Dataset generated successfully.")
