# -*- coding: utf-8 -*-
"""
Audit the four independent Web of Science Core Collection corpora (when licensed TSV
exports are supplied in data/raw_wos/) and check unadjusted vs multivariable
OLS residuals across processed country cohort tables.
"""
import os
from pathlib import Path
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_COHORTS = REPO_ROOT / "data" / "cohorts"
DATA_PANELS = REPO_ROOT / "data" / "panels"
WOS_RAW_DIR = Path(os.environ.get("WOS_RAW_DIR", REPO_ROOT / "data" / "raw_wos"))


def check_cohort_residuals():
    cohorts = [
        ("Chikungunya", "master_chik_percapita.csv"),
        ("Zika", "master_zika_percapita.csv"),
        ("Dengue", "master_dengue_percapita.csv"),
        ("West Nile virus", "master_wnv_percapita.csv"),
    ]
    for virus, fn in cohorts:
        d = pd.read_csv(DATA_COHORTS / fn)
        d = d[d["Cases"] > 0].copy()
        cmed = d["Cases"].median()
        d["gap_final"] = (d["Cases"] > cmed) & (d["resid_mv"] < -0.25)
        corr = d["resid"].corr(d["resid_mv"])
        print(f"{virus:16s}: n={len(d):2d} | corr(resid, resid_mv)={corr:.3f} | gaps={sorted(d.loc[d['gap_final'], 'ISO3'].tolist())}")


def audit_raw_wos():
    wos_files = {
        "Chikungunya": ("Merged_Chikungunya_WOS_Full.txt", (2013, 2023), (2008, 2024), "Americas (PAHO ARCA + CDC ArboNET, n=33)"),
        "Dengue": ("Merged_dengue_WOS_Full.txt", (2008, 2024), (2008, 2024), "Global benchmark (OpenDengue v1.3, n=96)"),
        "Zika": ("Merged_ZIKA_WOS_Full.txt", (2016, 2023), (2015, 2024), "Americas (PAHO ARCA + CDC ArboNET, n=33)"),
        "West Nile virus": ("Merged_west_nile_WOS_Full.txt", (2008, 2023), (2008, 2024), "Europe & USA (ECDC + CDC ArboNET, n=24)"),
    }
    rows = []
    for virus, (fname, bw, mw, scope) in wos_files.items():
        fp = WOS_RAW_DIR / fname
        if not fp.exists():
            print(f"Skipping raw audit for {virus}: {fp} not supplied.")
            continue
        df = pd.read_csv(fp, sep="\t", dtype=str, encoding="utf-8-sig", low_memory=False, on_bad_lines="skip")
        df.columns = [c.strip() for c in df.columns]
        df_ar = df[df["DT"].fillna("").str.contains("Article|Review", case=False, regex=True)].copy()
        df_dedup = df_ar.drop_duplicates(subset=["UT"]).copy() if "UT" in df_ar else df_ar.copy()
        py = pd.to_numeric(df_dedup["PY"], errors="coerce")
        rows.append({
            "Virus": virus,
            "Raw_rows": len(df),
            "ArtRev_rows": len(df_ar),
            "Unique_UT_AllYears": len(df_dedup),
            "Year_Span": f"{int(py.min())}-{int(py.max())}",
            "Unique_UT_le2024": int((py <= 2024).sum()),
            "Unique_UT_le2025": int((py <= 2025).sum()),
            "Match_Window": f"{mw[0]}-{mw[1]}",
            "Unique_UT_MatchWindow": int(((py >= mw[0]) & (py <= mw[1])).sum()),
            "Burden_Window": f"{bw[0]}-{bw[1]}",
            "Unique_UT_BurdenWindow": int(((py >= bw[0]) & (py <= bw[1])).sum()),
            "AB_pct": round(df_dedup["AB"].fillna("").str.strip().ne("").mean() * 100, 1),
            "DI_pct": round(df_dedup["DI"].fillna("").str.strip().ne("").mean() * 100, 1),
            "C1_pct": round(df_dedup["C1"].fillna("").str.strip().ne("").mean() * 100, 1),
            "FU_pct": round(df_dedup["FU"].fillna("").str.strip().ne("").mean() * 100, 1),
            "Surveillance_Scope": scope,
        })
    if rows:
        out_df = pd.DataFrame(rows)
        out_df.to_csv(DATA_PANELS / "corpus_screening_summary.csv", index=False, encoding="utf-8-sig")
        print(out_df.to_string(index=False))


if __name__ == "__main__":
    check_cohort_residuals()
    audit_raw_wos()
