# Fraud Detection Pipeline — Phase 1 (Local, No Cloud Needed Yet)

This is the very first stage of a much bigger project (a full real-time
fraud detection system on Azure). In this phase, everything runs on
**your own laptop** — no Azure account, no credit card, no cost. The
goal is just to get a real, working fraud-detection model end to end.

---

## 0. What you need before starting

1. **Python 3.10 or newer** installed on your computer.
   - Check by opening a terminal (Command Prompt / PowerShell on
     Windows, Terminal on Mac/Linux) and typing:
     ```
     python --version
     ```
     If you see `Python 3.10.x` or higher, you're good. If not, download
     it from https://www.python.org/downloads/ and install it (on
     Windows, tick "Add Python to PATH" during install).

2. **VS Code** (recommended, but any editor works):
   https://code.visualstudio.com/

3. **This project folder**, unzipped somewhere easy to find, like
   `Documents/fraud-detection-pipeline`.

---

## 1. Open a terminal INSIDE the project folder

- In VS Code: `File > Open Folder...` → select `fraud-detection-pipeline`
  → then open a terminal with `` Ctrl+` `` (backtick).
- Or manually: open a terminal and run `cd path/to/fraud-detection-pipeline`.

Every command below assumes you're standing inside this folder.

---

## 2. Create a "virtual environment" (venv)

**What is this and why do I need it?** A virtual environment is a
private, isolated box of Python packages just for this project. Without
it, every Python project on your computer shares the same packages, and
they can start conflicting with each other (Project A needs version 1
of a library, Project B needs version 2 — chaos). A venv gives each
project its own clean box.

Create it:
```
python -m venv venv
```

Activate it (you must do this every time you open a new terminal to
work on this project):

- **Windows (PowerShell):**
  ```
  venv\Scripts\activate
  ```
- **Mac/Linux:**
  ```
  source venv/bin/activate
  ```

You'll know it worked because your terminal prompt now starts with
`(venv)`.

---

## 3. Install the project's dependencies

**What's a dependency?** Code other people already wrote that we
reuse instead of rebuilding from scratch (e.g. `scikit-learn` gives us
ready-made machine learning algorithms).

```
pip install -r requirements.txt
```

This reads `requirements.txt` and installs everything listed there,
inside your venv only.

---

## 4. Generate fake test data (so you can try everything right now)

The REAL dataset (284,807 real-ish anonymized transactions) has to be
downloaded from Kaggle, which needs a free account. So we can start
testing immediately, this script builds a small FAKE dataset with the
exact same column names and a similarly rare fraud rate.

```
python -m src.data.make_sample_data
```

You should see something like:
```
Fake sample data created at: .../data/raw/creditcard.csv
   Total rows: 20000
   Fraud rows: 34 (0.170%)
```

---

## 5. Look at how the data gets split

```
python -m src.data.load_data
```

This shows you how many rows land in the train / validation / test
sets. Notice it's split by TIME, not randomly — see the comment at the
top of `src/data/load_data.py` for why that matters.

---

## 6. Train your first model

```
python -m src.models.train_baseline
```

This trains a Logistic Regression model (a simple, fast, well-understood
algorithm — the standard "first model" for any classification problem)
and saves it to `outputs/models/baseline_logreg.joblib`.

---

## 7. Check how good the model is

```
python -m src.models.evaluate
```

You'll see something like:
```
PR-AUC:                 0.83
ROC-AUC:                0.95
Recall @ 90% precision: 0.71
Confusion matrix ([[TN, FP], [FN, TP]]):
[[738   2]
 [  3   7]]
```

(Exact numbers will differ since the fake data is random.) See the
comment at the top of `src/models/evaluate.py` for what each metric
means and why we don't just use "accuracy."

---

## 8. Run the automated tests

**What's a test?** Code that checks OTHER code is doing what it's
supposed to — automatically, so you don't have to manually re-check by
eye every time you change something.

```
pytest
```

You should see `2 passed`.

---

## 9. Phase 2 — a smarter model, and a fair comparison

Logistic Regression is fine as a starting point, but it's a fairly
simple model. Now we train **XGBoost**, which is far more powerful at
finding complex patterns — but power alone isn't enough here, because
fraud is *extremely* rare. If we trained XGBoost the normal way, it
could get 99.8%+ "accuracy" just by never predicting fraud at all. So
we teach it to care about the rare class in two different ways (see the
big comment at the top of `src/models/train_xgboost.py` for the full
explanation):

- **`scale_pos_weight`** — tell XGBoost directly that a missed fraud
  row is much more costly than a missed normal row.
- **SMOTE** — generate synthetic fraud examples so the training set
  looks more balanced before the model ever sees it.

Train both versions:
```
python -m src.models.train_xgboost
```

Then put every model you've trained (baseline + both XGBoost versions)
on the same test set, side by side:
```
python -m src.models.compare_models
```

You'll get a table like:
```
Model                                    PR-AUC  ROC-AUC  Recall@90P
------------------------------------------------------------------------
Baseline (Logistic Regression)           1.0000   1.0000      1.0000
XGBoost + scale_pos_weight               1.0000   1.0000      1.0000
XGBoost + SMOTE                          1.0000   1.0000      1.0000
```

**Important beginner lesson:** on the FAKE data, you'll likely see every
model score a perfect (or near-perfect) 1.0000. That's not a bug — it
just means our fake fraud pattern is too easy (we made it obvious on
purpose so you could test things quickly). It does NOT mean all three
models are equally good. Real differences between a simple model and a
well-tuned XGBoost only show up on real, messy data — which is exactly
why step 10 matters.

---

## 10. Swap in the REAL dataset (once you're ready)

1. Go to https://www.kaggle.com/ and create a free account.
2. Search for **"Credit Card Fraud Detection" (ULB / Worldline /
   MLG-ULB)** and download `creditcard.csv`.
3. Put that file at `data/raw/creditcard.csv`, **replacing** the fake
   one (same filename, so nothing else needs to change).
4. Re-run everything from step 6 onward (`train_baseline`,
   `train_xgboost`, `compare_models`). Now you're training on 284,807
   real transactions instead of 20,000 fake ones — and this time the
   PR-AUC differences between models will actually mean something.

---

## What's next (future phases — we'll build these together)

1. ~~Local setup~~ ✅ / ~~Baseline model~~ ✅
2. ~~Advanced model: XGBoost with imbalance handling, compared fairly
   against the baseline~~ ✅ *(you are here)*
3. Wrap everything into a clean, reusable pipeline + add MLflow so
   every experiment's results get tracked automatically.
4. Move training into Azure ML (cloud), with the dataset registered as
   a Data Asset.
5. Automate it: GitHub Actions runs your tests and retrains the model
   every time you push code (CI/CD).
6. Deploy the model behind a real-time scoring API (Azure Managed
   Online Endpoint) with a <150ms response target.
7. Add monitoring: watch for data drift and automatically trigger
   retraining when fraud patterns shift.
8. Wrap the cloud infrastructure itself in Terraform, so the whole
   system can be rebuilt from scratch with one command.

Come back and say "let's do phase 3" whenever you're ready to keep
going.

## Project folder map

```
fraud-detection-pipeline/
├── data/
│   ├── raw/              <- creditcard.csv goes here (fake or real)
│   └── processed/
├── outputs/
│   └── models/           <- trained model files get saved here
├── src/
│   ├── data/
│   │   ├── make_sample_data.py   <- builds the fake test dataset
│   │   └── load_data.py          <- loads CSV + time-based split
│   ├── features/
│   │   └── build_features.py     <- turns raw columns into features
│   ├── models/
│   │   ├── train_baseline.py     <- trains Logistic Regression
│   │   ├── train_xgboost.py      <- trains XGBoost (2 imbalance strategies)
│   │   ├── compare_models.py     <- champion vs challenger comparison
│   │   └── evaluate.py           <- scores a model on test data
│   └── utils/
│       └── config.py             <- all file paths & constants
├── tests/
│   └── test_features.py          <- automated checks for our code
├── requirements.txt              <- list of packages to install
└── README.md                     <- this file
```
