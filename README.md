# Heart Disease Prediction System

Assignment 2 (Task 2) – Machine Learning Fundamentals (AIC354), COMSATS University Islamabad, Lahore Campus.

**Live app:** `PASTE STREAMLIT URL HERE AFTER DEPLOYMENT`

> ⚠️ Educational project only. Not a medical diagnosis tool.

## Overview
Binary classifier that predicts whether heart disease is present (1) or absent (0) from **four inputs**. It follows the instructor's Titanic workflow (load → understand → encode → train → test → apply → feedback → deploy) and uses the same algorithm, a **Support Vector Classifier (SVC)**.

## Dataset
UCI Heart Disease, **Cleveland** database (`processed.cleveland.data`, 303 patients, 14 attributes, `?` = missing):
https://archive.ics.uci.edu/dataset/45/heart+disease
Credit: Robert Detrano, M.D., Ph.D. (V.A. Medical Center, Long Beach and Cleveland Clinic Foundation); donor David W. Aha.
Target `num` (0–4) is converted to binary: 0 → no disease, 1–4 → disease present.

## Features used (exactly four)
| Feature | Meaning |
|---|---|
| `age` | Age in years |
| `sex` | 1 = male, 0 = female |
| `cp` | Chest pain: 1 typical angina, 2 atypical angina, 3 non-anginal, 4 asymptomatic |
| `thalach` | Maximum heart rate achieved |

## Methodology
Stratified 80/20 split (`random_state=42`) → scikit-learn `Pipeline` (StandardScaler for age/thalach, one-hot for `cp`, `sex` as is) → `SVC(random_state=0)`. Evaluated with accuracy, precision, recall, F1, confusion matrix and 5-fold cross-validation. Whole pipeline saved as `heart_disease_model.pkl`.

## Files
| File | Purpose |
|---|---|
| `Heart_Disease_Prediction.ipynb` | Full executed notebook |
| `heart_disease_model.pkl` | Trained pipeline (scikit-learn 1.7.2) |
| `app.py` | Streamlit app (4 inputs) |
| `requirements.txt`, `runtime.txt` | Dependencies, Python 3.11 |
| `processed.cleveland.data` | Dataset |

## Run locally
```bash
python -m venv venv
venv\Scripts\activate          # Windows  (macOS/Linux: source venv/bin/activate)
pip install -r requirements.txt
streamlit run app.py
```
To re-run the notebook: `pip install jupyter matplotlib seaborn`, then `jupyter notebook` and *Restart & Run All*.
Important: retrain with **scikit-learn 1.7.2** (the pinned version) so the saved pickle loads on Streamlit Cloud.

## Deploy (Streamlit Community Cloud)
1. Push these files to a public GitHub repo.
2. https://share.streamlit.io → *Create app* → *Deploy a public app from GitHub*.
3. Branch `main`, main file `app.py`, Advanced settings → Python **3.11** → Deploy.
4. Open the URL, test, and paste it at the top of the notebook and this README.

## Known limitations
Only four attributes; 303 patients from one hospital (patients referred for angiography); small test set (61) so results vary between splits; default SVC settings (no tuning); not clinically validated.
