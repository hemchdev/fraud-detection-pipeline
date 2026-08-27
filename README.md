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

## 11. Phase 3 — one command that runs everything, and remembers every experiment

Right now, running three separate scripts and eyeballing the printed
numbers works, but it doesn't scale — you'll forget what settings gave
you which score after your 10th experiment. Phase 3 fixes that two
ways:

**A) A validation step, so broken data never quietly reaches training.**
```
python -m src.pipeline.validate_data
```
This runs sanity checks (no missing values, no negative amounts, the
answer column only contains 0/1, etc.) and stops loudly if anything
looks wrong — see `src/pipeline/validate_data.py` for the full list and
why each check exists.

**B) MLflow — automatic experiment tracking.**
**What is MLflow, really?** Every time you train a model, it records a
"run": what settings you used and what scores came out — automatically,
so you never have to remember or write it down yourself.

Run the whole pipeline (validate → preprocess → train all 3 models →
evaluate → log to MLflow → promote a champion) with:
```
python -m src.pipeline.train_pipeline
```

Then open a browser dashboard of every run you've ever done:
```
mlflow ui
```
and visit **http://localhost:5000**. Click "fraud-detection" (our
experiment name) and you'll see every run listed with its metrics —
tick two runs' checkboxes and click "Compare" to see them side by side.

**What's this "champion" thing?** After every pipeline run, the model
with the best PR-AUC gets checked against whoever the CURRENT champion
is (saved in `outputs/models/registry/champion_metadata.json`). It only
gets promoted if it's at least as good — so a bad run can never
accidentally replace a good model. This is the same logic real MLOps
teams use before anything reaches production.

---

## 13. Phase 4 — moving training to Azure ML (the cloud)

**Heads up before we start:** everything up to now, I actually ran
myself and watched it work. From here on, I can't — I don't have
network access to Azure from where I run code, so I can't create your
Azure account, click your buttons, or submit a real cloud job on your
behalf. What I *did* do: verified every Azure SDK import in the code
below against the real installed library (not just my memory), and ran
the actual training script's logic locally as a stand-in test — it
works. The only genuinely untested part is the real Azure connection
itself. **If any command below errors, paste me the exact error message
and I'll help you fix it.**

**Also — this phase can cost real money if left running.** Read the
cost note at the end of this section before you start.

### What are we actually building here, in plain words?

Right now your training happens on your laptop. Azure ML lets the
*same kind of training* happen on a rented computer in Microsoft's data
center instead — useful because cloud machines can be much bigger than
your laptop, can run unattended on a schedule, and every run gets
automatically tracked and versioned. Five new Azure concepts, explained
simply:

- **Resource group** — a folder that holds every Azure resource for
  this project together, so you can delete the whole project in one
  click later.
- **Workspace** — the "home base" for all your ML work in Azure —
  datasets, models, and job history all live inside it.
- **Compute cluster** — virtual machines Azure spins up *only when a
  job needs them*, then shuts back down. `min-instances 0` means you
  pay nothing while it's idle.
- **Environment** — a recipe listing exactly which Python packages
  need to be installed in the container that runs your code (the cloud
  version of `requirements.txt`).
- **Data Asset** — a named, versioned pointer to your dataset, stored
  in Azure, that jobs reference by name instead of a hardcoded path.

### Step-by-step setup

**1. Create a free Azure account.**
Go to https://azure.microsoft.com/free — you get $200 of credit for 30
days plus some always-free services for 12 months. Needs a credit card
for identity verification, but you won't be charged unless you go over
the free credit.

**2. Install the Azure CLI** (the `az` command).
Follow the instructions for your OS: https://learn.microsoft.com/cli/azure/install-azure-cli
Then add the machine learning extension:
```
az extension add -n ml
```

**3. Log in:**
```
az login
```
This opens a browser window for you to sign in.

**4. Create a resource group** (pick any Azure region near you, e.g.
`eastus`, `centralindia`, `southeastasia`):
```
az group create --name rg-fraud-detection-sea --location southeastasia
```

**5. Create the ML workspace:**
```
az ml workspace create --name mlw-fraud-detection --resource-group rg-fraud-detection-sea --location southeastasia
```
This takes a couple of minutes — it's also quietly creating a storage
account, a key vault, and an insights resource for you, all bundled
inside the resource group.

**6. Save yourself repetitive typing:**
```
az configure --defaults group=rg-fraud-detection-sea workspace=mlw-fraud-detection
```

**7. Create the compute cluster** (the VM size below is a small, cheap
one — fine for this dataset size):
```
az ml compute create --name cpu-cluster --type AmlCompute --size Standard_DS2_v2 --min-instances 0 --max-instances 1
```

**8. Register the training environment** (the "recipe" of Python
packages, defined in `aml/aml-environment.yml`):
```
az ml environment create --file aml/aml-environment.yml
```

**9. Install the Python libraries that let YOUR laptop talk to Azure**
(already added to `requirements.txt`, so just re-run):
```
pip install -r requirements.txt
```

**10. Register the dataset as a Data Asset.** First, find your
subscription ID:
```
az account show --query id -o tsv
```
Then set it as an environment variable and run the registration script
(`src/cloud/register_data_asset.py`):
- **Mac/Linux:**
  ```
  export AZURE_SUBSCRIPTION_ID=<paste the id from above>
  python -m src.cloud.register_data_asset
  ```
- **Windows (PowerShell):**
  ```
  $env:AZURE_SUBSCRIPTION_ID="<paste the id from above>"
  python -m src.cloud.register_data_asset
  ```

**11. Submit the actual training job to the cloud:**
```
az ml job create --file aml/job.yml --web
```
`--web` opens the job's live page in Azure ML Studio, where you can
watch logs stream in real time and see the PR-AUC / ROC-AUC metrics
appear once it finishes (usually a few minutes).

**12. Download the trained model back to your laptop** once the job
shows "Completed" (replace `<job-name>` with the name shown in Studio
or printed by step 11):
```
az ml job download --name <job-name> --output-name model_dir --download-path ./outputs/from_azure
```

### Cost note — please read

The compute cluster only costs money while a job is actually running
on it (`min-instances 0` means it scales to zero and costs nothing
while idle). Still:
- Check **Cost Management** in the Azure Portal occasionally.
- When you're done experimenting for the day, you can stop worrying
  about it since idle compute is free — but if you want to be
  extra safe, delete the compute with
  `az ml compute delete --name cpu-cluster` and recreate it (step 7)
  next time.
- When you're completely done with this project, delete everything at
  once with `az group delete --name rg-fraud-detection` — this removes
  every resource we created, guaranteed zero ongoing cost.

---

## 14. Swap in the REAL dataset (once you're ready)

1. Go to https://www.kaggle.com/ and create a free account.
2. Search for **"Credit Card Fraud Detection" (ULB / Worldline /
   MLG-ULB)** and download `creditcard.csv`.
3. Put that file at `data/raw/creditcard.csv`, **replacing** the fake
   one (same filename, so nothing else needs to change).
4. Re-run everything from step 6 onward (`train_baseline`,
   `train_xgboost`, `compare_models`), or just re-run the whole pipeline
   with `python -m src.pipeline.train_pipeline`. Now you're training on
   284,807 real transactions instead of 20,000 fake ones — and this
   time the PR-AUC differences between models will actually mean
   something.

---

## What's next (future phases — we'll build these together)

1. ~~Local setup~~ ✅ / ~~Baseline model~~ ✅
2. ~~Advanced model: XGBoost with imbalance handling, compared fairly
   against the baseline~~ ✅
3. ~~Clean, reusable pipeline + MLflow experiment tracking + a
   champion/challenger registry~~ ✅
4. ~~Move training into Azure ML (cloud), with the dataset registered
   as a Data Asset~~ ✅ *(you are here)*
5. Automate it: GitHub Actions runs your tests and retrains the model
   every time you push code (CI/CD).
6. Deploy the model behind a real-time scoring API (Azure Managed
   Online Endpoint) with a <150ms response target.
7. Add monitoring: watch for data drift and automatically trigger
   retraining when fraud patterns shift.
8. Wrap the cloud infrastructure itself in Terraform, so the whole
   system can be rebuilt from scratch with one command.

Come back and say "let's do phase 5" whenever you're ready to keep
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
│   ├── pipeline/
│   │   ├── validate_data.py      <- data sanity checks (stage 1)
│   │   └── train_pipeline.py     <- runs every stage + MLflow + registry
│   ├── cloud/
│   │   ├── register_data_asset.py <- uploads dataset to Azure ML
│   │   └── train_job_entry.py     <- the script that runs INSIDE Azure
│   └── utils/
│       └── config.py             <- all file paths & constants
├── aml/
│   ├── environment.yml           <- conda recipe for the cloud job
│   ├── aml-environment.yml       <- registers that recipe with Azure
│   └── job.yml                   <- describes one cloud training run
├── tests/
│   ├── test_features.py          <- automated checks for our code
│   └── test_validate_data.py     <- automated checks for validation
├── requirements.txt              <- list of packages to install
└── README.md                     <- this file
```
