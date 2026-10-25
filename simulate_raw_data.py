"""
Simulate a messy supply chain shipments file (SIMULATED data)

Real supply chain exports are rarely clean: names are typed in different ways,
dates come in several formats, numbers carry units and symbols, values go
missing, and rows get entered twice. This script builds a file with those
problems on purpose, so the cleaning pipeline in clean_supply_chain_data.py
has realistic work to do.

The suppliers are made up. The emission factors are illustrative values, not
official figures.

Run:  python simulate_raw_data.py
Writes: raw_shipments.csv
"""

from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).parent
rng = np.random.default_rng(7)  # fixed seed so the file is the same every run

N_SHIPMENTS = 520

SUPPLIERS = {
    "Lagos Agro Foods Ltd": "Nigeria",
    "Ibadan Packaging Ltd": "Nigeria",
    "Kano Textile Co": "Nigeria",
    "Accra Cocoa Traders Ltd": "Ghana",
    "Nairobi Fresh Produce Ltd": "Kenya",
    "Durban Steel Works Ltd": "South Africa",
    "Cairo Plastics Co": "Egypt",
}

# Illustrative emission factors, kg CO2e per tonne-km
EMISSION_FACTOR = {"Road": 0.10, "Sea": 0.016, "Air": 0.60}
# Illustrative freight cost, USD per tonne-km
COST_PER_TKM = {"Road": 0.08, "Sea": 0.02, "Air": 0.90}
DISTANCE_RANGE = {"Road": (50, 1800), "Sea": (800, 9000), "Air": (600, 6000)}

COUNTRY_VARIANTS = {
    "Nigeria": ["Nigeria", "nigeria", "NIGERIA", "NG", "Nigeria "],
    "Ghana": ["Ghana", "ghana", "GH", "Ghana "],
    "Kenya": ["Kenya", "kenya", "KE"],
    "South Africa": ["South Africa", "south africa", "ZA", "S. Africa"],
    "Egypt": ["Egypt", "egypt", "EG"],
}
MODE_VARIANTS = {
    "Road": ["Road", "road", "ROAD", "Truck", "Road "],
    "Sea": ["Sea", "sea", "Ship", "Sea freight", "SEA "],
    "Air": ["Air", "air", "Air freight", "AIR"],
}
DATE_FORMATS = ["%Y-%m-%d", "%d/%m/%Y", "%d %b %Y", "%b %d, %Y"]


def supplier_variant(name):
    r = rng.random()
    if r < 0.55:
        return name
    options = [name.upper(), name.lower(), name + "  ", "  " + name]
    if name.endswith(" Ltd"):
        options.append(name[:-3] + "Limited")
    if name.endswith(" Co"):
        options.append(name[:-2] + "Company")
    return options[rng.integers(len(options))]


def build_clean_truth():
    rows = []
    start = pd.Timestamp("2024-01-01")
    for i in range(1, N_SHIPMENTS + 1):
        supplier = list(SUPPLIERS)[rng.integers(len(SUPPLIERS))]
        mode = rng.choice(list(EMISSION_FACTOR), p=[0.55, 0.30, 0.15])
        lo, hi = DISTANCE_RANGE[mode]
        distance = rng.uniform(lo, hi)
        weight = float(np.clip(rng.lognormal(2.3, 0.7), 0.5, 80))
        cost = (150 + distance * weight * COST_PER_TKM[mode]) * rng.lognormal(0, 0.1)
        emissions = weight * distance * EMISSION_FACTOR[mode] * rng.lognormal(0, 0.05)
        date = start + pd.Timedelta(days=int(rng.integers(0, 546)))
        rows.append((f"SC-{i:04d}", supplier, SUPPLIERS[supplier], date, mode, distance, weight, cost, emissions))
    return pd.DataFrame(
        rows,
        columns=["shipment_id", "supplier", "country", "date", "mode", "distance", "weight", "cost", "emissions"],
    )


def messify(t):
    raw = pd.DataFrame(index=t.index)
    raw["shipment_id"] = t["shipment_id"]
    raw["supplier"] = [supplier_variant(s) for s in t["supplier"]]
    raw["country"] = [COUNTRY_VARIANTS[c][rng.integers(len(COUNTRY_VARIANTS[c]))] for c in t["country"]]

    # Dates: mixed formats, a few blanks and placeholders
    dates = []
    for d in t["date"]:
        r = rng.random()
        if r < 0.015:
            dates.append("")
        elif r < 0.025:
            dates.append("TBC")
        else:
            dates.append(d.strftime(DATE_FORMATS[rng.integers(len(DATE_FORMATS))]))
    raw["ship_date"] = dates

    raw["transport_mode"] = [MODE_VARIANTS[m][rng.integers(len(MODE_VARIANTS[m]))] for m in t["mode"]]

    # Distance: units and separators, missing values, impossible values
    dist = []
    for d in t["distance"]:
        r = rng.random()
        if r < 0.06:
            dist.append("")
        elif r < 0.075:
            dist.append(f"-{d:.0f}")
        elif r < 0.085:
            dist.append("999999")
        else:
            dist.append([f"{d:,.0f} km", f"{d:.0f}", f"{d:.1f}"][rng.integers(3)])
    raw["distance_km"] = dist

    # Weight: tonnes or kilograms, with placeholders for missing
    wt = []
    for w in t["weight"]:
        r = rng.random()
        if r < 0.07:
            wt.append(["", "N/A", "unknown"][rng.integers(3)])
        else:
            wt.append([f"{w:.1f} t", f"{w * 1000:.0f} kg", f"{w:.2f}"][rng.integers(3)])
    raw["weight"] = wt

    # Cost: currency symbols and thousands separators
    cost = []
    for c in t["cost"]:
        if rng.random() < 0.05:
            cost.append("")
        else:
            cost.append([f"${c:,.0f}", f"{c:.2f}", f"${c:,.2f}"][rng.integers(3)])
    raw["cost_usd"] = cost

    # Emissions: reported for about 70 percent of shipments
    raw["emissions_kg_co2e"] = ["" if rng.random() < 0.30 else f"{e:.1f}" for e in t["emissions"]]
    return raw


def main():
    raw = messify(build_clean_truth())

    # Duplicates: some exact copies, some re-typed with different formatting
    exact = raw.sample(30, random_state=1)
    retyped = raw.sample(20, random_state=2).copy()
    retyped["shipment_id"] = retyped["shipment_id"].str.lower() + " "
    retyped["supplier"] = retyped["supplier"].str.upper()
    raw = pd.concat([raw, exact, retyped]).sample(frac=1, random_state=3).reset_index(drop=True)

    raw.to_csv(OUT / "raw_shipments.csv", index=False)
    print(f"Wrote raw_shipments.csv with {len(raw)} rows ({N_SHIPMENTS} real shipments plus 50 duplicate rows)")


if __name__ == "__main__":
    main()
