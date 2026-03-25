# Semantic and Context-Aware Security Analysis

A machine learning pipeline for network intrusion detection and semantic interpretation of cybersecurity alerts, built on the UNSW-NB15 dataset using XGBoost.

**Authors**
- Mehdi Ougadi
- Aziz Doghri

---

## Project Structure

```
├── data/
│   ├── UNSW_NB15_training-set.csv
│   ├── UNSW_NB15_testing-set.csv
│   └── DATA.md                        # Dataset setup instructions
├── results/
│   ├── binary/                        # Binary classification outputs
│   ├── multiclass/                    # Multi-class classification outputs
│   ├── feature_semantics/             # Feature grouping & distribution outputs
│   ├── attack_profiling/              # Attack behavior profiling outputs
│   ├── false_negatives/               # False negative analysis outputs
│   ├── false_positives/               # False positive analysis outputs
│   └── script.log                     # Full run log
├── src/
│   ├── data_loader.py                 # Dataset loading & preprocessing
│   ├── binary_classifier.py           # Binary intrusion detection
│   ├── multiclass_classifier.py       # Attack category classification
│   ├── feature_semantics.py           # Feature grouping & distributions
│   ├── attack_profiling.py            # Attack behavior profiling
│   ├── false_negatives.py             # False negative analysis
│   └── false_positives.py             # False positive analysis
├── main.py
├── report.pdf
├── requirements.txt
└── pyproject.toml
```

---

## Configuration

**Requirements:** Python 3.11+

Create and activate a virtual environment, then install dependencies:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Download the dataset and place the CSV files in `data/` — see `data/DATA.md` for instructions.

---

## Run

```bash
python main.py
```

All plots and CSVs are saved under `results/`. Execution logs are written to `results/script.log`.

---

## Results

| Module | Output |
|--------|--------|
| Binary Classification | Precision: ~0.82, Recall: ~0.99, FPR: ~0.27, FNR: ~0.01 — the model detects most attacks but produces a significant number of false alarms on normal traffic |
| Multi-Class Classification | Generic and Normal traffic are well classified (F1 ~0.98). Minority categories such as Analysis, Backdoor, and DoS show low recall (~0.07) due to class imbalance |
| Feature Semantics | Features grouped into 4 semantic categories: Network Context, Traffic Behavior, Temporal Context, and Connection Patterns — with distribution plots comparing normal vs attack traffic |
| Attack Profiling | DoS shows high source load with near-zero destination response. Exploits have higher packet exchange and longer duration. Reconnaissance has low load and short-lived connections |
| False Negatives | Missed attacks closely mimic normal traffic in duration, packet count, and load — making them inherently difficult to detect without deeper behavioral context |
| False Positives | High-load benign flows share statistical overlap with attack traffic, particularly in `sload` — contributing to alert fatigue in operational environments |