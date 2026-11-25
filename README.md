# Arbovirus Burden and Research Capacity

Processed datasets, statistical analysis scripts, and supplementary tables accompanying the manuscript:

> *Unequal alignment between arbovirus disease burden and research capacity: cross-country evidence across four viruses and mechanistic insights from chikungunya, 1985–2026*

## Repository Structure

```text
.
├── CITATION.cff
├── LICENSE
├── README.md
├── requirements.txt
├── data/
│   ├── cohorts/          # Harmonized burden, population, GDP, and GBD 2021 sensitivity tables
│   ├── panels/           # Country-year whole/fractional panels, HHI, funding, and MNCS tables
│   ├── topics/           # BERTopic UT-to-topic assignments, stability metrics, and income-bias tests
│   ├── openalex/         # Stratified OpenAlex audit sample (n = 2,000) and WoS DOI list
│   └── raw_wos/          # Target directory for licensed Web of Science TSV exports
├── scripts/
│   ├── utils/
│   │   ├── country_norm.py
│   │   └── wos_matching.py
│   ├── 01_audit_corpus_and_gbd.py
│   └── 02_compute_regressions_and_inequity.py
└── supplementary/
    └── Supplementary_Tables_S1-S5.xlsx
```

## Data Availability and Licensing Scope

Raw bibliographic records retrieved from the Web of Science (WoS) Core Collection on 7 June 2026 cannot be redistributed under Clarivate licensing. All non-copyrighted processed outputs (country-level burden–output cohorts, whole- and fractional-count country–year panels, funding and year-normalized citation tables, BERTopic accession-number (`UT`) topic assignments with `Representative_Docs` abstracts removed, OpenAlex audit records, and DOI matching sets) are included in `data/`.

### WoS Corpora (Core Collection; Article & Review)

* **Chikungunya**: `TS=("Chikungunya" OR "CHIKV")` (11,569 initial records; 9,430 unique UTs, 1985–2026)
* **Dengue**: Parallel Topic-field search (30,623 unique UTs, 1902–2027)
* **Zika**: Parallel Topic-field search (11,395 unique UTs, 1987–2027)
* **West Nile virus**: Parallel Topic-field search (11,238 unique UTs, 1942–2026)

Public epidemiological and socioeconomic inputs derive from OpenDengue v1.3, PAHO ARCA, US CDC ArboNET, ECDC Surveillance Atlas, IHME Global Burden of Disease 2021, World Bank World Development Indicators, and OpenAlex.

## Usage

Tested with Python 3.13:

```bash
pip install -r requirements.txt
python scripts/01_audit_corpus_and_gbd.py
```

To regenerate all downstream tables from licensed WoS TSV exports, place the four tab-delimited exports inside `data/raw_wos/` (or set `WOS_RAW_DIR`) and run:

```bash
python scripts/02_compute_regressions_and_inequity.py
```

## License

* Python scripts in `scripts/`: MIT License.
* Processed summary tables in `data/` and `supplementary/`: CC BY 4.0.
