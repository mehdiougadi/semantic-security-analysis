# Data Setup

## Dataset: UNSW-NB15

The UNSW-NB15 dataset is **not included** in this repository due to its size.  
You must download it manually and place the files as described below.

---

## Download

Choose one of the following sources:

- **Official:** https://research.unsw.edu.au/projects/unsw-nb15-dataset
- **Google Drive Mirror:** https://drive.google.com/drive/folders/19fuKj2Ij9Z-kgmTo61TWyLX1JVyYrveO?usp=sharing

---

## Files Required

Download the following CSV files:

| File | Description |
|------|-------------|
| `UNSW_NB15_training-set.csv` | Training split |
| `UNSW_NB15_testing-set.csv` | Testing split |

---

## Expected Folder Structure

Once downloaded, place the files as follows:

```
data/
├── UNSW_NB15_training-set.csv
├── UNSW_NB15_testing-set.csv
└── DATA.md
```

---

## Notes

- Do **not** rename the files, as the scripts reference them by their original names.
- The dataset contains ~257,000 records across 49 features including `attack_cat` and `Label`.
- If you use a different split, update the file paths in `src/data_loader.py`.