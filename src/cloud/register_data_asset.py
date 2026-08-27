"""
Uploads our dataset to Azure and registers it as a "Data Asset" — a
named, versioned pointer to that file that any cloud job can reference
by name instead of a hardcoded local file path.

WHY BOTHER, IF WE COULD JUST UPLOAD THE CSV DIRECTLY?
A Data Asset is versioned. Every time you register a new version of
creditcard.csv, Azure keeps every old version too. A job can pin
itself to an exact version ("train on v3"), so your results stay
reproducible even after the underlying data changes later — you can
always go back and ask "what did the model see when it got this
score?"

BEFORE RUNNING THIS, YOU NEED (see README Phase 4 for the full walkthrough):
  1. A free Azure account: https://azure.microsoft.com/free
  2. Azure CLI installed and logged in (`az login`)
  3. An Azure ML workspace already created
  4. pip install azure-ai-ml azure-identity   (already in requirements.txt)
  5. These three environment variables set:
       AZURE_SUBSCRIPTION_ID, AZURE_RESOURCE_GROUP, AZURE_ML_WORKSPACE

Run it with:
    python -m src.cloud.register_data_asset
"""
import os

from azure.ai.ml import MLClient
from azure.ai.ml.entities import Data
from azure.ai.ml.constants import AssetTypes
from azure.identity import DefaultAzureCredential

from src.utils.config import RAW_DATA_PATH

DATA_ASSET_NAME = "fraud-creditcard-data"


def register():
    subscription_id = os.environ["AZURE_SUBSCRIPTION_ID"]
    resource_group = os.environ.get("AZURE_RESOURCE_GROUP", "rg-fraud-detection-sea")
    workspace_name = os.environ.get("AZURE_ML_WORKSPACE", "mlw-fraud-detection")

    # DefaultAzureCredential automatically uses whatever you're already
    # logged in with via `az login` — no separate password needed here.
    ml_client = MLClient(
        DefaultAzureCredential(),
        subscription_id,
        resource_group,
        workspace_name,
    )

    data_asset = Data(
        name=DATA_ASSET_NAME,
        description="Kaggle credit card fraud dataset (ULB/Worldline), or the local practice sample.",
        path=RAW_DATA_PATH,
        type=AssetTypes.URI_FILE,
    )

    registered = ml_client.data.create_or_update(data_asset)
    print(f"Registered data asset: {registered.name}, version: {registered.version}")
    print(f"Jobs can reference it as: azureml:{registered.name}@latest")


if __name__ == "__main__":
    register()
