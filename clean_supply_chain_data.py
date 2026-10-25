"""
Supply chain data cleaning pipeline

Reads raw_shipments.csv (simulated, see simulate_raw_data.py) and turns it into
an analysis-ready table. Every step is logged with how many rows and cells it
changed, so the cleaning is easy to check and explain.

Steps
1. Standardise shipment IDs and remove duplicate rows
2. Standardise supplier names
3. Standardise country and transport mode labels
4. Parse mixed date formats
5. Convert distance, weight and cost to numbers (handling units and symbols)
6. Flag impossible values and treat them as missing
7. Fill missing distances from similar shipments (and flag them)
8. Estimate missing emissions from distance, weight and mode (and flag them)
9. Run final checks and save the outputs

Run:  python clean_supply_chain_data.py
Needs: numpy, pandas, matplotlib
"""

import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT = Path(__file__).parent

MISSING_TOKENS = {"", "n/a", "na", "unknown", "tbc", "none", "null"}
COUNTRY_MAP = {
    "nigeria": "Nigeria", "ng": "Nigeria",
    "ghana": "Ghana", "gh": "Ghana",
    "kenya": "Kenya", "ke": "Kenya",
    "south africa": "South Africa", "s. africa": "South Africa", "za": "South Africa",
    "egypt": "Egypt", "eg": "Egypt",
}
MODE_MAP = {
    "road": "Road", "truck": "Road",
    "sea": "Sea", "ship": "Sea", "sea freight": "Sea",
    "air": "Air", "air freight": "Air",
}
# Illustrative emission factors, kg CO2e per tonne-km (not official figures)
EMISSION_FACTOR = {"Road": 0.10, "Sea": 0.016, "Air": 0.60}
DATE_FORMATS = ["%Y-%m-%d", "%d/%m/%Y", "%d %b %Y", "%b %d, %Y"]  # slashes are read as day/month/year
DATE_MIN, DATE_MAX = pd.Timestamp("2024-01-01"), pd.Timestamp("2025-12-31")
MAX_DISTANCE_KM = 20000
MAX_WEIGHT_T = 200

log = []


def record(step, description, rows_before, rows_after, cells_changed):
    log.append({
        "step": step,
        "description": description,
        "rows_before": rows_before,
        "rows_after": rows_after,
        "cells_changed": cells_changed,
    })


def n_changed(before, after):
    """Count how many cells differ between two columns, treating missing as empty."""
    return int((before.fillna("").astype(str) != after.fillna("").astype(str)).sum())


def parse_dates(series):
    parsed = pd.Series(pd.NaT, index=series.index, dtype="datetime64[ns]")
    for fmt in DATE_FORMATS:
        attempt = pd.to_datetime(series, format=fmt, errors="coerce")
        parsed = parsed.fillna(attempt)
    return parsed


def parse_weight(value):
    """Return weight in tonnes. Values with 'kg' are converted; plain numbers are read as tonnes."""
    text = value.strip().lower()
    if text in MISSING_TOKENS:
        return np.nan
    match = re.search(r"-?\d+(?:\.\d+)?", text.replace(",", ""))
    if not match:
        return np.nan
    number = float(match.group())
    return number / 1000 if "kg" in text else number


def to_number(series):
    cleaned = series.str.replace(r"[^\d.\-]", "", regex=True)
    return pd.to_numeric(cleaned, errors="coerce")


def main():
    df = pd.read_csv(OUT / "raw_shipments.csv", dtype=str, keep_default_na=False)
    raw_rows = len(df)

    # Share of values that are missing in the raw file, counting placeholders like "N/A"
    raw_missing = df.apply(lambda c: c.str.strip().str.lower().isin(MISSING_TOKENS)).mean() * 100

    # 1. IDs and duplicates
    before_ids = df["shipment_id"].copy()
    df["shipment_id"] = df["shipment_id"].str.strip().str.upper()
    ids_changed = n_changed(before_ids, df["shipment_id"])
    rows_before = len(df)
    df = df.drop_duplicates(subset="shipment_id", keep="first").reset_index(drop=True)
    record(1, "Standardise shipment IDs, then drop duplicate rows", rows_before, len(df),
           ids_changed + (rows_before - len(df)))

    # 2. Supplier names
    rows = len(df)
    before = df["supplier"].copy()
    stripped = df["supplier"].str.strip().str.replace(r"\s+", " ", regex=True)
    key = (stripped.str.lower()
           .str.replace(r"\blimited\b", "ltd", regex=True)
           .str.replace(r"\bcompany\b", "co", regex=True)
           .str.replace(".", "", regex=False))
    canonical = stripped.groupby(key).agg(lambda s: s.value_counts().index[0])
    df["supplier"] = key.map(canonical)
    record(2, "Standardise supplier names (case, spacing, Ltd/Limited, Co/Company)", rows, len(df),
           n_changed(before, df["supplier"]))

    # 3. Country and transport mode
    before = df["country"].copy()
    df["country"] = df["country"].str.strip().str.lower().map(COUNTRY_MAP)
    record(3, "Map country spellings and codes to one standard name", len(df), len(df),
           n_changed(before, df["country"]))
    before = df["transport_mode"].copy()
    df["transport_mode"] = df["transport_mode"].str.strip().str.lower().map(MODE_MAP)
    record(3, "Map transport mode labels to Road, Sea or Air", len(df), len(df),
           n_changed(before, df["transport_mode"]))

    # 4. Dates
    before = df["ship_date"].copy()
    df["ship_date"] = parse_dates(df["ship_date"].str.strip())
    out_of_range = (df["ship_date"] < DATE_MIN) | (df["ship_date"] > DATE_MAX)
    df.loc[out_of_range, "ship_date"] = pd.NaT
    record(4, "Parse four date formats into one; dates outside 2024 to 2025 set to missing", len(df), len(df),
           n_changed(before, df["ship_date"].dt.strftime("%Y-%m-%d")))

    # 5. Numbers
    before = df["distance_km"].copy()
    df["distance_km"] = to_number(df["distance_km"])
    record(5, "Remove 'km' and thousands separators from distance", len(df), len(df),
           n_changed(before, df["distance_km"]))
    before = df["weight"].copy()
    df["weight_tonnes"] = df["weight"].map(parse_weight)
    df = df.drop(columns="weight")
    record(5, "Convert weight to tonnes (kg values divided by 1000)", len(df), len(df),
           n_changed(before, df["weight_tonnes"]))
    before = df["cost_usd"].copy()
    df["cost_usd"] = to_number(df["cost_usd"])
    record(5, "Remove '$' and thousands separators from cost", len(df), len(df),
           n_changed(before, df["cost_usd"]))
    df["emissions_kg_co2e"] = pd.to_numeric(df["emissions_kg_co2e"], errors="coerce")

    # 6. Impossible values
    bad_distance = (df["distance_km"] <= 0) | (df["distance_km"] > MAX_DISTANCE_KM)
    n_bad_distance = int(bad_distance.sum())
    df.loc[bad_distance, "distance_km"] = np.nan
    record(6, f"Distances at or below 0 or above {MAX_DISTANCE_KM:,} km set to missing", len(df), len(df), n_bad_distance)
    bad_weight = (df["weight_tonnes"] <= 0) | (df["weight_tonnes"] > MAX_WEIGHT_T)
    df.loc[bad_weight, "weight_tonnes"] = np.nan
    record(6, f"Weights at or below 0 or above {MAX_WEIGHT_T} t set to missing", len(df), len(df), int(bad_weight.sum()))

    # 7. Fill missing distances from the median of similar shipments
    df["distance_imputed"] = df["distance_km"].isna()
    group_median = df.groupby(["supplier", "transport_mode"])["distance_km"].transform("median")
    mode_median = df.groupby("transport_mode")["distance_km"].transform("median")
    df["distance_km"] = df["distance_km"].fillna(group_median).fillna(mode_median).round(0)
    record(7, "Fill missing distance with the median for the same supplier and mode (flagged)", len(df), len(df),
           int(df["distance_imputed"].sum()))

    # 8. Estimate missing emissions
    calculated = df["distance_km"] * df["weight_tonnes"] * df["transport_mode"].map(EMISSION_FACTOR)
    df["emissions_estimated"] = df["emissions_kg_co2e"].isna() & calculated.notna()
    df["emissions_kg_co2e"] = df["emissions_kg_co2e"].fillna(calculated).round(1)
    record(8, "Estimate missing emissions as distance x weight x mode factor (flagged)", len(df), len(df),
           int(df["emissions_estimated"].sum()))

    # Reported emissions far from the calculated value are flagged for review
    ratio = df["emissions_kg_co2e"] / calculated
    df["emissions_check"] = np.where(~df["emissions_estimated"] & ((ratio < 0.7) | (ratio > 1.3)), "review", "ok")

    # 9. Final checks and outputs
    assert df["shipment_id"].is_unique, "duplicate IDs remain"
    assert df["transport_mode"].isin(EMISSION_FACTOR).all(), "unmapped transport mode"
    assert df["country"].notna().all(), "unmapped country"
    assert df["supplier"].nunique() == 7, "supplier names not fully standardised"

    df = df.sort_values("ship_date").reset_index(drop=True)
    columns = ["shipment_id", "supplier", "country", "ship_date", "transport_mode", "distance_km", "distance_imputed",
               "weight_tonnes", "cost_usd", "emissions_kg_co2e", "emissions_estimated", "emissions_check"]
    df = df[columns]
    df.to_csv(OUT / "clean_shipments.csv", index=False)
    pd.DataFrame(log).to_csv(OUT / "cleaning_log.csv", index=False)

    # Summary
    clean_missing = df[["shipment_id", "supplier", "country", "ship_date", "transport_mode", "distance_km",
                        "weight_tonnes", "cost_usd", "emissions_kg_co2e"]].isna().mean() * 100
    print(f"Rows: {raw_rows} raw -> {len(df)} clean")
    print(pd.DataFrame(log).to_string(index=False))
    print("\nMissing values (percent), raw vs clean:")
    compare = pd.DataFrame({"raw": raw_missing.rename({"weight": "weight_tonnes"}).round(1), "clean": clean_missing.round(1)})
    print(compare.to_string())
    print(f"\nEmissions flagged for review: {(df['emissions_check'] == 'review').sum()}")

    share = df.groupby("transport_mode").agg(shipments=("shipment_id", "count"), emissions_t=("emissions_kg_co2e", lambda s: s.sum() / 1000))
    share["shipment_share_pct"] = (share["shipments"] / share["shipments"].sum() * 100).round(1)
    share["emissions_share_pct"] = (share["emissions_t"] / share["emissions_t"].sum() * 100).round(1)
    print("\nBy transport mode:")
    print(share.round(1).to_string())

    plot(compare, share, OUT / "data_quality.png")


def plot(compare, share, path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))

    ax = axes[0]
    cols = ["distance_km", "weight_tonnes", "cost_usd", "emissions_kg_co2e", "ship_date"]
    labels = ["Distance", "Weight", "Cost", "Emissions", "Ship date"]
    x = np.arange(len(cols))
    ax.bar(x - 0.2, compare.loc[cols, "raw"], width=0.4, color="#c2410c", label="Raw file")
    ax.bar(x + 0.2, compare.loc[cols, "clean"], width=0.4, color="#0e7490", label="After cleaning")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Missing values (%)")
    ax.set_title("Missing data before and after cleaning")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)

    ax = axes[1]
    order = ["Road", "Sea", "Air"]
    x = np.arange(len(order))
    ax.bar(x - 0.2, share.loc[order, "shipment_share_pct"], width=0.4, color="#64748b", label="Share of shipments")
    ax.bar(x + 0.2, share.loc[order, "emissions_share_pct"], width=0.4, color="#c2410c", label="Share of emissions")
    ax.set_xticks(x)
    ax.set_xticklabels(order)
    ax.set_ylabel("Percent of total")
    ax.set_title("Shipments vs emissions by transport mode")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)

    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
