"""
build_house_f_eng.py

Applies the same feature engineering logic used in house_f_eng.csv to the FULL
house_data.csv (21,613 rows), producing a complete engineered dataset.

Usage:
    pip install pandas
    python build_house_f_eng.py house_data.csv house_f_eng_full.csv

All features are deliberately kept "real-world" / buyer-relevant and
interpretable -- no polynomial terms, PCA, embeddings, or target encoding.
"""

import sys
import pandas as pd


def price_tier(p):
    if p < 300_000:
        return "Entry"
    elif p < 600_000:
        return "Mid"
    elif p < 1_000_000:
        return "Upper"
    else:
        return "Luxury"


def season(m):
    if m in (12, 1, 2):
        return "Winter"
    elif m in (3, 4, 5):
        return "Spring"
    elif m in (6, 7, 8):
        return "Summer"
    else:
        return "Fall"


def yes_no(cond):
    return "Yes" if cond else "No"


def build(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # --- Date decomposition ---
    dt = pd.to_datetime(df["date"].str.slice(0, 8), format="%Y%m%d")
    df["yr_sale"] = dt.dt.year
    df["m_sale"] = dt.dt.month
    df["qr_sale"] = dt.dt.quarter
    df["season_sold"] = df["m_sale"].apply(season)

    # --- Price-based ---
    df["price_tier"] = df["price"].apply(price_tier)
    df["price_per_sqft"] = (df["price"] / df["sqft_living"]).round(0).astype(int)

    # --- Size / lot ---
    df["lot_size_acres"] = (df["sqft_lot"] / 43_560).round(2)
    df["pct_size_vs_neighbors"] = (
        (df["sqft_living"] - df["sqft_living15"]) / df["sqft_living15"] * 100
    ).round(1)

    # --- Bed/bath ---
    df["total_bed_bath"] = df["bedrooms"] + df["bathrooms"]
    df["bath_per_bed"] = (df["bathrooms"] / df["bedrooms"].replace(0, pd.NA)).round(2)
    df["bath_per_bed"] = df["bath_per_bed"].fillna(0)

    # --- Structural flags ---
    df["has_basement"] = df["sqft_basement"].apply(lambda x: yes_no(x > 0))
    df["multi_story"] = df["floors"].apply(lambda x: yes_no(x > 1))
    df["view_flag"] = df["view"].apply(lambda x: yes_no(x > 0))
    df["waterfront_flag"] = df["waterfront"].apply(lambda x: yes_no(x == 1))

    # --- Age / renovation ---
    df["house_age_yrs"] = df["yr_sale"] - df["yr_built"]
    df["was_renovated"] = df["yr_renovated"].apply(lambda x: yes_no(x > 0))
    df["yrs_since_renovation"] = df.apply(
        lambda r: r["yr_sale"] - r["yr_renovated"] if r["yr_renovated"] > 0 else r["house_age_yrs"],
        axis=1,
    )

    cols = [
        "id", "yr_sale", "m_sale", "qr_sale", "season_sold",
        "price", "price_tier", "price_per_sqft",
        "bedrooms", "bathrooms", "total_bed_bath", "bath_per_bed",
        "sqft_living", "sqft_lot", "lot_size_acres",
        "floors", "multi_story",
        "waterfront_flag", "view_flag", "condition", "grade",
        "has_basement", "yr_built", "house_age_yrs",
        "was_renovated", "yrs_since_renovation",
        "pct_size_vs_neighbors", "zipcode", "lat", "long",
    ]
    return df[cols]


if __name__ == "__main__":
    src = sys.argv[1] if len(sys.argv) > 1 else "house_data.csv"
    dst = sys.argv[2] if len(sys.argv) > 2 else "house_f_eng_full.csv"

    raw = pd.read_csv(src, dtype={"zipcode": str})
    out = build(raw)
    out.to_csv(dst, index=False)
    print(f"Wrote {len(out):,} rows to {dst}")
