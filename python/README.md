# HAND ML Pipeline — Phase 1: Dataset Parsing & Feature Extraction

## Project Structure

```
hand_ml/
├── README.md
├── requirements.txt
├── data/
│   ├── grabmyo/          ← download GRABMyo here (PhysioNet WFDB format)
│   └── pads/             ← download PADS here (PhysioNet JSON + txt format)
├── features/             ← output: feature CSVs produced by loaders
├── models/               ← output: saved SVM + scaler (Phase 2)
│
├── loaders/
│   ├── grabmyo_loader.py     ← parses GRABMyo WFDB files → raw EMG arrays
│   └── pads_loader.py        ← parses PADS JSON + timeseries → raw IMU arrays
│
├── feature_extraction/
│   ├── emg_features.py       ← RMS, median freq, tremor band power, EMD proxy
│   ├── imu_features.py       ← tremor power, tremor freq, orientation stability, etc.
│   └── pressure_features.py  ← MVC, RFD, fatigue index, force variability (for live use)
│
├── build_dataset.py          ← runs loaders + feature extraction → saves features/*.csv
└── train_svm.py              ← Phase 2: loads CSVs, trains SVM, saves model
```

## Download Instructions

### GRABMyo (Group 1 — Stroke/EMG baseline)
```
pip install wfdb
# Then download from PhysioNet:
wget -r -N -c -np https://physionet.org/files/grabmyo/1.1.0/ -P data/grabmyo/
```
Or manually from: https://physionet.org/content/grabmyo/1.1.0/

### PADS (Group 2 — Parkinson's/IMU)
```
wget -r -N -c -np https://physionet.org/files/parkinsons-disease-smartwatch/1.0.0/ -P data/pads/
```
Or manually from: https://physionet.org/content/parkinsons-disease-smartwatch/1.0.0/

## Label Mapping

### Group 1 (Stroke — EMG features)
- GRABMyo subjects = all healthy → label 0 (Healthy)
- When stroke patient data becomes available: label 1 (Mild) or 2 (Severe)

### Group 2 (Parkinson's — IMU features)
- PADS condition == "Healthy"       → label 0
- PADS condition == "Parkinson's"   → label 1 or 2 based on UPDRS score
  - UPDRS tremor score 0–1          → label 1 (Mild)
  - UPDRS tremor score 2–3          → label 2 (Severe)
- Differential diagnosis subjects are excluded (ambiguous label)

## Running Phase 1
```bash
pip install -r requirements.txt
python build_dataset.py
# Outputs: features/group1_features.csv, features/group2_features.csv
```
