# -*- coding: utf-8 -*-
"""
Compute multivariable burden-output regressions, fractional-count sensitivity checks,
HHI concentration trends, MNCS citation impact, collaboration isolation, topic bias,
and IHME GBD 2021 sensitivity estimates.
"""
import csv
import os
import sys
import re
from collections import Counter, defaultdict
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import spearmanr, fisher_exact

from pathlib import Path
import os
REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_COHORTS = REPO_ROOT / "data" / "cohorts"
DATA_PANELS = REPO_ROOT / "data" / "panels"
DATA_TOPICS = REPO_ROOT / "data" / "topics"
DATA_OPENALEX = REPO_ROOT / "data" / "openalex"
WOS_RAW_DIR = Path(os.environ.get("WOS_RAW_DIR", REPO_ROOT / "data" / "raw_wos"))
OUT = str(DATA_PANELS)
DATA_DIR = str(DATA_COHORTS)
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from utils.country_norm import parse_countries
from utils import wos_matching as gwm

# Reuse funding agency rules from build_covariates.py
AGENCY_RULES = [
    ("US (NIH/HHS/DoD)", ["national institutes of health", " niaid", "national institute of allergy",
                          "u.s. department of health", "department of defense", " dod"]),
    ("Brazil", ["cnpq", "conselho nacional de desenvolvimento", "fap", "fundacao de amparo",
                "capes", "coordenacao de aperfeicoamento"]),
    ("EU", ["european commission", "horizon", "european union"]),
    ("China", ["national key research", "national natural science", "nsfc",
               "major project for control"]),
    ("UK", ["wellcome", "medical research council", " mrc"]),
    ("France", ["ird", "cirad", "inserm", "agence nationale de la recherche", " anr",
                "institut national de la sante"]),
    ("Japan", ["japan society", "jps", "japan agency"]),
]
HIC_AGENCIES = {"US (NIH/HHS/DoD)", "EU", "UK", "France", "Japan"}

def classify_agencies(fp_text):
    low = (fp_text or "").lower()
    return {tag for tag, kws in AGENCY_RULES if any(k in low for k in kws)}

def to_int(s):
    try:
        return int(float(str(s).strip()))
    except Exception:
        return None

# 1. Decadal Share Shift (2010-2014 vs 2020-2024 on panel_whole & frac)
pw = pd.read_csv(os.path.join(str(DATA_PANELS), "chik_country_year_whole.csv"), index_col=0)
pf = pd.read_csv(os.path.join(str(DATA_PANELS), "chik_country_year_fractional.csv"), index_col=0)
pw.columns = pw.columns.astype(int)
pf.columns = pf.columns.astype(int)

def share_shift(panel, w1, w2):
    s1 = panel[w1].sum(axis=1)
    s2 = panel[w2].sum(axis=1)
    p1 = s1 / s1.sum() * 100.0
    p2 = s2 / s2.sum() * 100.0
    return pd.DataFrame({
        "cnt_early": s1, "cnt_late": s2,
        "share_early_pct": p1, "share_late_pct": p2,
        "shift_pp": p2 - p1,
    })

ss_w = share_shift(pw, list(range(2010, 2015)), list(range(2020, 2025)))
ss_f = share_shift(pf, list(range(2010, 2015)), list(range(2020, 2025)))
top15 = list(pw.sum(axis=1).sort_values(ascending=False).head(15).index)
ss_cmp = ss_w.loc[top15].join(ss_f.loc[top15], lsuffix="_whole", rsuffix="_frac")
ss_cmp.index.name = "ISO3"
ss_cmp.to_csv(os.path.join(OUT, "chik_share_shift_2010_2024.csv"))
print("=== 1. DECADAL SHARE SHIFT (2020-2024 vs 2010-2014, Top 15) ===")
print(ss_cmp[["share_early_pct_whole", "share_late_pct_whole", "shift_pp_whole", "shift_pp_frac"]].round(2).to_string())

# 2. Fractional-count HHI vs Whole-count HHI (2010-2024)
years_10_24 = list(range(2010, 2025))
hhi_rows = []
for y in years_10_24:
    sw = pw[y] / pw[y].sum()
    sf = pf[y] / pf[y].sum()
    hhi_w = float((sw ** 2).sum())
    hhi_f = float((sf ** 2).sum())
    hhi_rows.append({"year": y, "HHI_whole": round(hhi_w, 4), "HHI_frac": round(hhi_f, 4)})
hhi_df = pd.DataFrame(hhi_rows)
hhi_df.to_csv(os.path.join(OUT, "chik_hhi_whole_vs_frac.csv"), index=False)
print("\n=== 2. HHI (Whole vs Fractional, 2010-2024) ===")
print(hhi_df.iloc[[0, 5, 10, 11, 14]].to_string(index=False))

# 3. FIG1 OLS VIF & Fractional-count Sensitivity across 4 viruses
def compute_ols_and_vif(df, pubs_col="Publications"):
    d = pd.DataFrame({
        "cases": np.log10(df["Cases"] + 1),
        "pop": np.log10(df["Population"]),
        "gdp": np.log10(df["GDP_pc"]),
        "pubs": np.log10(df[pubs_col] + 1),
    }).dropna()
    Z = (d - d.mean()) / d.std(ddof=1)
    X_raw = Z[["cases", "pop", "gdp"]].values
    X = np.column_stack([np.ones(len(Z)), X_raw])
    y = Z["pubs"].values
    inv = np.linalg.inv(X.T @ X)
    beta = inv @ X.T @ y
    res = y - X @ beta
    n, k = X.shape
    s2 = (res @ res) / (n - k)
    se = np.sqrt(np.diag(s2 * inv))
    tv = beta / se
    p = 2 * (1 - stats.t.cdf(np.abs(tv), n - k))
    r2 = 1 - (res @ res) / ((y - y.mean()) @ (y - y.mean()))
    # VIF on standardized predictors
    corr_X = np.corrcoef(X_raw, rowvar=False)
    vifs = np.diag(np.linalg.inv(corr_X))
    # Unstandardized resid_mv
    Xu = np.column_stack([np.ones(len(d)), d[["cases", "pop", "gdp"]].values])
    bu, *_ = np.linalg.lstsq(Xu, d["pubs"].values, rcond=None)
    resid_mv = d["pubs"].values - (Xu @ bu)
    return {
        "n": n, "R2": r2,
        "beta_cases": beta[1], "p_cases": p[1],
        "beta_pop": beta[2], "p_pop": p[2],
        "beta_gdp": beta[3], "p_gdp": p[3],
        "VIF_cases": vifs[0], "VIF_pop": vifs[1], "VIF_gdp": vifs[2],
        "resid_mv": resid_mv,
    }

# Compute fractional country publications from the 4 raw WOS files using global_wos_matching rules
WOS_PATHS = {
    "Chikungunya": (str(WOS_RAW_DIR / "Merged_Chikungunya_WOS_Full.txt"), 2008, "master_chik_percapita.csv"),
    "Zika": (str(WOS_RAW_DIR / "Merged_ZIKA_WOS_Full.txt"), 2015, "master_zika_percapita.csv"),
    "Dengue": (str(WOS_RAW_DIR / "Merged_dengue_WOS_Full.txt"), 2008, "master_dengue_percapita.csv"),
    "West Nile": (str(WOS_RAW_DIR / "Merged_west_nile_WOS_Full.txt"), 2008, "master_wnv_percapita.csv"),
}
EU2_TO_3 = {
    "AT": "AUT", "BE": "BEL", "BG": "BGR", "CY": "CYP", "CZ": "CZE", "DE": "DEU", "DK": "DNK",
    "EE": "EST", "EL": "GRC", "ES": "ESP", "FI": "FIN", "FR": "FRA", "HR": "HRV", "HU": "HUN",
    "IE": "IRL", "IS": "ISL", "IT": "ITA", "LI": "LIE", "LT": "LTU", "LU": "LUX", "LV": "LVA",
    "MT": "MLT", "NL": "NLD", "NO": "NOR", "PL": "POL", "PT": "PRT", "RO": "ROU", "SE": "SWE",
    "SI": "SVN", "SK": "SVK"
}

sens_rows = []
for v, (wpath, ystart, mcsv) in WOS_PATHS.items():
    frac_pubs = Counter()
    whole_pubs = Counter()
    seen = set()
    with open(wpath, "r", encoding="utf-8-sig", errors="replace", newline="") as fh:
        rdr = csv.DictReader(fh, delimiter="\t")
        for rn, row in enumerate(rdr, 2):
            py = (row.get("PY") or "").strip()
            if not py.isdigit() or not (ystart <= int(py) <= 2024):
                continue
            uid = (row.get("UT") or "").strip() or f"ROW:{rn}"
            if uid in seen:
                continue
            seen.add(uid)
            codes = set()
            for blk in gwm.split_blocks(row.get("C1") or ""):
                code = gwm.block_country(blk)
                if code:
                    codes.add(EU2_TO_3.get(code, code))
            if codes:
                w = 1.0 / len(codes)
                for c in codes:
                    whole_pubs[c] += 1
                    frac_pubs[c] += w

    mdf = pd.read_csv(os.path.join(DATA_DIR, mcsv))
    mdf = mdf[mdf["Cases"] > 0].copy()
    mdf["Pubs_frac"] = mdf["ISO3"].map(lambda x: frac_pubs.get(x, 0.0))
    mdf["Pubs_whole_recheck"] = mdf["ISO3"].map(lambda x: whole_pubs.get(x, 0))

    ow = compute_ols_and_vif(mdf, "Publications")
    of = compute_ols_and_vif(mdf, "Pubs_frac")
    rho_w = spearmanr(np.log10(mdf["Cases"] + 1), np.log10(mdf["Publications"] + 1))
    rho_f = spearmanr(np.log10(mdf["Cases"] + 1), np.log10(mdf["Pubs_frac"] + 1))
    cmed = mdf["Cases"].median()
    gap_w = set(mdf.loc[(mdf["Cases"] > cmed) & (ow["resid_mv"] < -0.25), "ISO3"])
    gap_f = set(mdf.loc[(mdf["Cases"] > cmed) & (of["resid_mv"] < -0.25), "ISO3"])
    resid_corr = np.corrcoef(ow["resid_mv"], of["resid_mv"])[0, 1]

    sens_rows.append({
        "Virus": v, "n": ow["n"],
        "rho_whole": round(rho_w.statistic, 3), "p_whole": rho_w.pvalue,
        "rho_frac": round(rho_f.statistic, 3), "p_frac": rho_f.pvalue,
        "R2_whole": round(ow["R2"], 3), "R2_frac": round(of["R2"], 3),
        "beta_cases_w": round(ow["beta_cases"], 3), "beta_cases_f": round(of["beta_cases"], 3),
        "p_cases_f": round(of["p_cases"], 4),
        "beta_pop_w": round(ow["beta_pop"], 3), "beta_pop_f": round(of["beta_pop"], 3),
        "beta_gdp_w": round(ow["beta_gdp"], 3), "beta_gdp_f": round(of["beta_gdp"], 3),
        "VIF_cases": round(ow["VIF_cases"], 2), "VIF_pop": round(ow["VIF_pop"], 2), "VIF_gdp": round(ow["VIF_gdp"], 2),
        "resid_corr_w_vs_f": round(resid_corr, 3),
        "n_gap_whole": len(gap_w), "n_gap_frac": len(gap_f),
        "gap_overlap": len(gap_w & gap_f),
        "gap_frac_list": ";".join(sorted(gap_f)),
    })

sens_df = pd.DataFrame(sens_rows)
sens_df.to_csv(os.path.join(OUT, "regression_vif_fractional.csv"), index=False)
print("\n=== 3. VIF & FRACTIONAL SENSITIVITY ===")
print(sens_df.to_string(index=False))

# 4. Load deduplicated CHIKV corpus (UT-deduplicated) for Steps 4, 5, 6
chik_path = str(WOS_RAW_DIR / "Merged_Chikungunya_WOS_Full.txt")
df_chik = pd.read_csv(chik_path, sep="\t", dtype=str, encoding="utf-8-sig", low_memory=False, on_bad_lines="skip")
df_chik.columns = [c.strip() for c in df_chik.columns]
df_chik = df_chik[df_chik["DT"].fillna("").str.contains("Article|Review", case=False, regex=True)].copy()
df_chik = df_chik.drop_duplicates(subset=["UT"]).copy()
df_chik["PY_int"] = pd.to_numeric(df_chik["PY"], errors="coerce")
df_chik["TC_int"] = pd.to_numeric(df_chik["TC"], errors="coerce").fillna(0)
df_chik["isos"] = df_chik["C1"].fillna("").map(parse_countries)
df_chik["agencies"] = df_chik["FP"].fillna("").map(classify_agencies)
df_chik["funded"] = df_chik["FU"].fillna("").str.strip().ne("")
print(f"\nLoaded deduplicated CHIKV Art+Rev corpus: {len(df_chik)} unique UTs")

# 4. Year-Normalized Citation Impact (MNCS, 2015-2024) for Top 12 countries
sub_15_24 = df_chik[(df_chik["PY_int"] >= 2015) & (df_chik["PY_int"] <= 2024)].copy()
yr_mean_tc = sub_15_24.groupby("PY_int")["TC_int"].mean().to_dict()
sub_15_24["norm_cite"] = sub_15_24.apply(lambda r: r["TC_int"] / max(yr_mean_tc[r["PY_int"]], 1e-9), axis=1)

top12_iso = list(pw.sum(axis=1).sort_values(ascending=False).head(12).index)
mncs_rows = []
for iso in top12_iso:
    mask = sub_15_24["isos"].map(lambda s: iso in s)
    vals = sub_15_24.loc[mask, "norm_cite"].values
    raw_vals = sub_15_24.loc[mask, "TC_int"].values
    # Fractional weighted mean
    frac_sub = sub_15_24.loc[mask]
    weights = frac_sub["isos"].map(lambda s: 1.0 / len(s)).values
    mncs_f = float(np.average(vals, weights=weights)) if len(vals) else np.nan
    mncs_w = float(np.mean(vals)) if len(vals) else np.nan
    se_w = float(np.std(vals, ddof=1) / np.sqrt(len(vals))) if len(vals) > 1 else 0.0
    mncs_rows.append({
        "ISO3": iso,
        "n_papers_2015_2024": int(mask.sum()),
        "raw_mean_TC": round(float(np.mean(raw_vals)), 2),
        "MNCS_whole": round(mncs_w, 3),
        "MNCS_SE": round(se_w, 3),
        "MNCS_frac": round(mncs_f, 3),
        "Median_NormCite": round(float(np.median(vals)), 3),
        "IQR_NormCite": round(float(np.percentile(vals, 75) - np.percentile(vals, 25)), 3),
    })
mncs_df = pd.DataFrame(mncs_rows).sort_values("MNCS_whole", ascending=False)
mncs_df.to_csv(os.path.join(OUT, "chik_year_normalized_citations.csv"), index=False)
print("\n=== 4. YEAR-NORMALIZED CITATIONS (2015-2024, Top 12) ===")
print(mncs_df.to_string(index=False))

# 5. Collaboration & Funding Flow to Gap Countries (FIG4D)
chik_pc = pd.read_csv(os.path.join(DATA_DIR, "master_chik_percapita.csv"))
chik_pc = chik_pc[chik_pc["Cases"] > 0].copy()
cmed_chik = chik_pc["Cases"].median()
GAP_STRICT = set(chik_pc.loc[(chik_pc["Cases"] > cmed_chik) & (chik_pc["resid_mv"] < -0.25), "ISO3"])
UNDER_RES = set(chik_pc.loc[chik_pc["resid_mv"] < 0, "ISO3"])
HUB_TOP8 = set(pw.sum(axis=1).sort_values(ascending=False).head(8).index)

# Classify countries in CHIKV corpus by World Bank GDP_pc >= 14005 (HIC approx) vs LMIC
wb = pd.read_csv(os.path.join(DATA_DIR, "worldbank_pop_gdp.csv"))
wb = wb[wb["ISO3"].str.len() == 3].set_index("ISO3")
HIC_SET = set(wb.index[wb["GDP_pc"] >= 14000]) | {"TWN", "SGP", "FRA", "GBR", "USA", "DEU", "AUS", "ITA", "JPN", "CAN", "ESP", "NLD", "CHE", "BEL", "AUT", "SWE", "NOR", "DNK", "FIN", "IRL", "ISR", "KOR", "NZL", "PRT", "GRC", "SAU", "ARE", "QAT"}

def tag_paper_group(isos):
    if not isos:
        return "Unresolved"
    has_gap = bool(isos & GAP_STRICT)
    has_under = bool(isos & UNDER_RES)
    has_hic = bool(isos & HIC_SET)
    has_hub = bool(isos & HUB_TOP8)
    n_c = len(isos)
    return pd.Series([has_gap, has_under, has_hic, has_hub, n_c > 1])

# Record-level summary by country group (for papers 2010-2024 and all years)
sub_all = df_chik[df_chik["isos"].map(len) > 0].copy()
sub_all["is_intl"] = sub_all["isos"].map(lambda s: len(s) > 1)
sub_all["has_hic_fund"] = sub_all["agencies"].map(lambda s: bool(s & HIC_AGENCIES))
sub_all["has_hic_partner"] = sub_all["isos"].map(lambda s: bool(s & HIC_SET))
sub_all["has_gap_strict"] = sub_all["isos"].map(lambda s: bool(s & GAP_STRICT))
sub_all["has_under_res"] = sub_all["isos"].map(lambda s: bool(s & UNDER_RES))
sub_all["has_lmic"] = sub_all["isos"].map(lambda s: any(c not in HIC_SET for c in s))
sub_all["hic_only"] = sub_all["isos"].map(lambda s: all(c in HIC_SET for c in s))

group_stats = []
for gname, mask in [
    ("Gap priority (DOM,GTM,GUY,SLV)", sub_all["has_gap_strict"]),
    ("Under-researched Americas (resid_mv<0)", sub_all["has_under_res"]),
    ("LMIC endemic (non-HIC)", sub_all["has_lmic"]),
    ("Core Hub countries (Top 8)", sub_all["isos"].map(lambda s: bool(s & HUB_TOP8))),
    ("High-income only (no LMIC)", sub_all["hic_only"]),
]:
    d = sub_all[mask]
    n = len(d)
    group_stats.append({
        "Group": gname,
        "n_papers": n,
        "Intl_collab_pct": round(d["is_intl"].mean() * 100, 1) if n else 0,
        "Domestic_only_pct": round((~d["is_intl"]).mean() * 100, 1) if n else 0,
        "Any_funding_pct": round(d["funded"].mean() * 100, 1) if n else 0,
        "HIC_agency_funding_pct": round(d["has_hic_fund"].mean() * 100, 1) if n else 0,
        "HIC_coauthor_pct": round(d["has_hic_partner"].mean() * 100, 1) if n else 0,
    })
gs_df = pd.DataFrame(group_stats)
gs_df.to_csv(os.path.join(OUT, "chik_collab_funding_gap_flow.csv"), index=False)
print("\n=== 5. COLLABORATION & FUNDING FLOW BY COUNTRY GROUP ===")
print(gs_df.to_string(index=False))

# Yearly composition of international collaborations (2010-2024)
comp_rows = []
for y in years_10_24:
    dy = sub_all[sub_all["PY_int"] == y]
    intl_y = dy[dy["is_intl"]]
    n_tot = len(dy)
    n_intl = len(intl_y)
    n_hic_only_intl = int(intl_y["hic_only"].sum())
    n_lmic_incl_intl = int(intl_y["has_lmic"].sum())
    n_gap_incl_intl = int(intl_y["has_gap_strict"].sum())
    n_under_incl_intl = int(intl_y["has_under_res"].sum())
    comp_rows.append({
        "year": y,
        "total_papers": n_tot,
        "intl_papers": n_intl,
        "intl_pct": round(n_intl / max(n_tot, 1) * 100, 1),
        "HIC_only_of_intl_pct": round(n_hic_only_intl / max(n_intl, 1) * 100, 1),
        "LMIC_incl_of_intl_pct": round(n_lmic_incl_intl / max(n_intl, 1) * 100, 1),
        "UnderRes_Amer_of_intl_pct": round(n_under_incl_intl / max(n_intl, 1) * 100, 1),
        "Gap_strict_of_intl_pct": round(n_gap_incl_intl / max(n_intl, 1) * 100, 1),
        "n_gap_strict_intl": n_gap_incl_intl,
        "n_under_res_intl": n_under_incl_intl,
    })
comp_df = pd.DataFrame(comp_rows)
comp_df.to_csv(os.path.join(OUT, "chik_collab_composition_by_year.csv"), index=False)
print("\n=== COLLAB COMPOSITION (2010 vs 2024) ===")
print(comp_df.iloc[[0, 5, 10, 14]].to_string(index=False))

# 6. BERTopic 19-topic bias (HIC-only vs Endemic/LMIC)
LABELS = {
    0: "Molecular virology & host immunity",
    1: "Epidemics & outbreak surveillance",
    2: "Antiviral inhibitors & drug discovery",
    3: "Vector ecology: Aedes mosquitoes",
    4: "Dengue co-circulation & diagnosis",
    5: "Zika co-infection & pregnancy",
    6: "Mosquito-virus interaction / saliva",
    7: "Insecticide resistance",
    8: "RT-PCR / molecular diagnostics",
    9: "Plant-based larvicides (oils)",
    10: "Wolbachia biocontrol",
    11: "Spatial risk modelling & surveillance",
    12: "Climate change & public health",
    13: "Prevention & community response",
    14: "Climate & vector distribution",
    15: "Vector population genetics",
    16: "Transfusion & blood safety",
    17: "Mathematical modelling & R0",
    18: "Nanoparticle vector control",
}

bt_doc = pd.read_csv(DATA_TOPICS / "bertopic_chik_doc_topics.csv")
bt_m = bt_doc.merge(df_chik[["UT", "isos", "funded", "agencies"]], on="UT", how="inner")
bt_m = bt_m[bt_m["isos"].map(len) > 0].copy()
bt_m["hic_only"] = bt_m["isos"].map(lambda s: all(c in HIC_SET for c in s))
bt_m["has_lmic"] = bt_m["isos"].map(lambda s: any(c not in HIC_SET for c in s))
bt_m["has_under"] = bt_m["isos"].map(lambda s: bool(s & UNDER_RES))

n_hic = int(bt_m["hic_only"].sum())
n_lmic = int(bt_m["has_lmic"].sum())
print(f"\n=== 6. BERTopic TOPIC BIAS (matched UTs={len(bt_m)}, HIC-only={n_hic}, LMIC-participating={n_lmic}) ===")

topic_rows = []
for tid in range(19):
    c_hic = int(((bt_m["Topic"] == tid) & bt_m["hic_only"]).sum())
    c_lmic = int(((bt_m["Topic"] == tid) & bt_m["has_lmic"]).sum())
    c_under = int(((bt_m["Topic"] == tid) & bt_m["has_under"]).sum())
    s_hic = c_hic / n_hic * 100.0
    s_lmic = c_lmic / n_lmic * 100.0
    diff_pp = s_hic - s_lmic  # positive = HIC-enriched, negative = LMIC-enriched
    or_val, p_val = fisher_exact([[c_hic, n_hic - c_hic], [c_lmic, n_lmic - c_lmic]])
    topic_rows.append({
        "Topic": tid,
        "Label": LABELS[tid],
        "Total_matched": c_hic + c_lmic,
        "HIC_only_n": c_hic, "HIC_only_pct": round(s_hic, 2),
        "LMIC_n": c_lmic, "LMIC_pct": round(s_lmic, 2),
        "UnderRes_Amer_n": c_under,
        "Diff_pp_HIC_minus_LMIC": round(diff_pp, 2),
        "Odds_Ratio_HIC_vs_LMIC": round(or_val, 3),
        "p_fisher": p_val,
    })

t_df = pd.DataFrame(topic_rows)
# BH-FDR correction
m_tests = len(t_df)
order = np.argsort(t_df["p_fisher"].values)
ranks = np.empty(m_tests)
ranks[order] = np.arange(1, m_tests + 1)
fdr = np.minimum(1.0, t_df["p_fisher"].values * m_tests / ranks)
# Enforce monotonicity
fdr_sorted = fdr[order]
for i in range(m_tests - 2, -1, -1):
    fdr_sorted[i] = min(fdr_sorted[i], fdr_sorted[i + 1])
fdr[order] = fdr_sorted
t_df["FDR_q"] = fdr
t_df = t_df.sort_values("Diff_pp_HIC_minus_LMIC", ascending=False)
t_df.to_csv(str(DATA_TOPICS / "bertopic_chik_income_bias.csv"), index=False)
print(t_df[["Topic", "Label", "HIC_only_pct", "LMIC_pct", "Diff_pp_HIC_minus_LMIC", "Odds_Ratio_HIC_vs_LMIC", "FDR_q"]].to_string(index=False))

# 7. IHME GBD 2021 Dengue Modeled Incidence Sensitivity Check
gbd = pd.read_csv(str(DATA_COHORTS / "ihme_gbd2021_dengue_incidence.csv"))
gbd_08_21 = gbd[(gbd["year"] >= 2008) & (gbd["year"] <= 2021)].groupby("location_name")["val"].sum().reset_index()
gbd_08_21.columns = ["location_name", "GBD_Inc_2008_2021"]

# Map GBD location names -> ISO3 in master_dengue_percapita.csv
den_pc = pd.read_csv(os.path.join(DATA_DIR, "master_dengue_percapita.csv"))
den_pc = den_pc[den_pc["Cases"] > 0].copy()

# Build name normalization map
GBD_NAME_TO_ISO = {
    "United States of America": "USA", "Bolivia (Plurinational State of)": "BOL",
    "Venezuela (Bolivarian Republic of)": "VEN", "Viet Nam": "VNM",
    "United Republic of Tanzania": "TZA", "Taiwan (Province of China)": "TWN",
    "Lao People's Democratic Republic": "LAO", "Republic of Korea": "KOR",
    "Democratic People's Republic of Korea": "PRK", "Iran (Islamic Republic of)": "IRN",
    "Syrian Arab Republic": "SYR", "Russian Federation": "RUS",
    "Republic of Moldova": "MDA", "Democratic Republic of the Congo": "COD",
    "Congo": "COG", "Cabo Verde": "CPV", "Côte d'Ivoire": "CIV",
    "Cote d'Ivoire": "CIV", "Brunei Darussalam": "BRN", "Bahamas": "BHS",
    "Saint Lucia": "LCA", "Saint Vincent and the Grenadines": "VCT",
    "United Kingdom": "GBR", "Turkiye": "TUR", "Turkey": "TUR",
    "Micronesia (Federated States of)": "FSM", "Palestine": "PSE",
    "United States Virgin Islands": "VIR", "Saint Kitts and Nevis": "KNA",
    "Gambia": "GMB", "Egypt": "EGY", "Yemen": "YEM",
}
cname_to_iso = {str(r["Country"]).strip().upper(): r["ISO3"] for _, r in den_pc.iterrows()}
# Add worldbank country names too
for _, r in wb.reset_index().iterrows():
    cname_to_iso.setdefault(str(r["Country"]).strip().upper(), r["ISO3"])

def map_gbd_iso(loc):
    if loc in GBD_NAME_TO_ISO:
        return GBD_NAME_TO_ISO[loc]
    up = loc.strip().upper()
    if up in cname_to_iso:
        return cname_to_iso[up]
    # fuzzy clean
    up2 = re.sub(r"\s*\(.*?\)", "", up).strip()
    return cname_to_iso.get(up2)

gbd_08_21["ISO3"] = gbd_08_21["location_name"].map(map_gbd_iso)
gbd_iso = gbd_08_21.dropna(subset=["ISO3"]).groupby("ISO3", as_index=False)["GBD_Inc_2008_2021"].sum()

# Also compute Dengue WoS 2008-2021 pubs so time window matches GBD 2008-2021!
den_wos_path = str(WOS_RAW_DIR / "Merged_dengue_WOS_Full.txt")
pubs_08_21 = Counter()
seen_d = set()
with open(den_wos_path, "r", encoding="utf-8-sig", errors="replace", newline="") as fh:
    rdr = csv.DictReader(fh, delimiter="\t")
    for rn, row in enumerate(rdr, 2):
        py = (row.get("PY") or "").strip()
        if not py.isdigit() or not (2008 <= int(py) <= 2021):
            continue
        uid = (row.get("UT") or "").strip() or f"ROW:{rn}"
        if uid in seen_d:
            continue
        seen_d.add(uid)
        codes = set()
        for blk in gwm.split_blocks(row.get("C1") or ""):
            code = gwm.block_country(blk)
            if code:
                codes.add(EU2_TO_3.get(code, code))
        for c in codes:
            pubs_08_21[c] += 1

m_gbd = den_pc.merge(gbd_iso, on="ISO3", how="inner")
m_gbd = m_gbd[m_gbd["GBD_Inc_2008_2021"] > 0].copy()
m_gbd["Pubs_2008_2021"] = m_gbd["ISO3"].map(lambda x: pubs_08_21.get(x, 0))

# Compare OpenDengue (2008-2024) vs GBD 2021 Modeled Incidence (2008-2021)
df_od = m_gbd.copy()
df_gb = m_gbd.copy()
df_gb["Cases"] = df_gb["GBD_Inc_2008_2021"]
df_gb["Publications"] = df_gb["Pubs_2008_2021"]

o_od = compute_ols_and_vif(df_od, "Publications")
o_gb = compute_ols_and_vif(df_gb, "Publications")
rho_od = spearmanr(np.log10(df_od["Cases"] + 1), np.log10(df_od["Publications"] + 1))
rho_gb = spearmanr(np.log10(df_gb["Cases"] + 1), np.log10(df_gb["Publications"] + 1))
rho_burden = spearmanr(df_od["Cases"], df_gb["Cases"])

gap_od = set(df_od.loc[(df_od["Cases"] > df_od["Cases"].median()) & (o_od["resid_mv"] < -0.25), "ISO3"])
gap_gb = set(df_gb.loc[(df_gb["Cases"] > df_gb["Cases"].median()) & (o_gb["resid_mv"] < -0.25), "ISO3"])

m_gbd["resid_mv_OpenDengue"] = o_od["resid_mv"]
m_gbd["resid_mv_GBD2021"] = o_gb["resid_mv"]
m_gbd.to_csv(str(DATA_COHORTS / "dengue_gbd2021_sensitivity.csv"), index=False)

print("\n=== 7. DENGUE GBD 2021 MODELED INCIDENCE VS OPENDENGUE ===")
print(f"Matched countries: n={len(m_gbd)} (unmatched in den_pc: {set(den_pc['ISO3']) - set(m_gbd['ISO3'])})")
print(f"Burden correlation (OpenDengue cases vs GBD 2021 incidence): rho={rho_burden.statistic:.3f}, p={rho_burden.pvalue:.3e}")
print(f"OpenDengue (2008-2024): rho={rho_od.statistic:.3f} (p={rho_od.pvalue:.3e}), R2={o_od['R2']:.3f}, beta_cases={o_od['beta_cases']:.3f} (p={o_od['p_cases']:.4f}), beta_pop={o_od['beta_pop']:.3f}, beta_gdp={o_od['beta_gdp']:.3f}")
print(f"GBD 2021   (2008-2021): rho={rho_gb.statistic:.3f} (p={rho_gb.pvalue:.3e}), R2={o_gb['R2']:.3f}, beta_cases={o_gb['beta_cases']:.3f} (p={o_gb['p_cases']:.4f}), beta_pop={o_gb['beta_pop']:.3f}, beta_gdp={o_gb['beta_gdp']:.3f}")
print(f"Resid correlation (OpenDengue vs GBD 2021): r={np.corrcoef(o_od['resid_mv'], o_gb['resid_mv'])[0,1]:.3f}")
print(f"Gap overlap: {len(gap_od & gap_gb)} / {len(gap_od)} (OD) & {len(gap_gb)} (GBD)")
print("  OD gaps :", sorted(gap_od))
print("  GBD gaps:", sorted(gap_gb))
print("ALL STATS DONE!")
