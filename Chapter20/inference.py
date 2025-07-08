"""inference.py – Calling the Managed Online Endpoint
====================================================
This utility script accompanies *Chapter XX – Deploying Models* and shows
how to perform an **online inference** against the Azure ML managed
endpoint that hosts our SimpleNN-MNIST model.

Sections
--------
1. Workspace connection – authenticates with Azure via `DefaultAzureCredential`
2. Endpoint metadata    – fetches scoring URI and access key (Bearer token)
3. Test payload         – creates a random 28×28 tensor serialised as a list
4. HTTP invocation      – sends JSON to the `/score` route and prints results

Key Concepts
------------
• *Scoring URI* – HTTPS address of the endpoint’s inference route.
• *Primary/secondary key* – simple authorization tokens returned by
  `ml_client.online_endpoints.get_keys()`.
• *Payload schema* – The scoring script expects `{ "data": [784 floats] }`
  for a single image or a batch of images.
• *Why not Azure ML SDK invoke?* – We purposely use plain `requests` to show
  the underlying REST call that tools (SDK/CLI) perform under the hood.

Run the script:
    python inference.py                       # quick smoke-test
    python inference.py --batch 8             # send 8 random digits

Set the following environment variables to avoid hard-coding secrets:
    AZ_SUBSCRIPTION_ID, AZ_RESOURCE_GROUP, AZ_WORKSPACE_NAME, AZ_ENDPOINT,
    AZ_ML_PRIMARY_KEY  (optional – will be fetched if missing)
"""

import os
import json
import logging
from typing import List

import requests
import torch
from azure.identity import DefaultAzureCredential  # or AzureCliCredential
from azure.ai.ml import MLClient

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration – pulled from env vars with sane defaults for the book demo.
# ---------------------------------------------------------------------------
SUBSCRIPTION = os.getenv("AZ_SUBSCRIPTION_ID", "287ddee1-542c-4ae3-9c98-ed8d88dd64bc")
RESOURCE_GRP = os.getenv("AZ_RESOURCE_GROUP", "pytorchbook")
WORKSPACE    = os.getenv("AZ_WORKSPACE_NAME", "pytorchbook")
ENDPOINT     = os.getenv("AZ_ENDPOINT", "simplenn-mnist-ep")
BATCH        = int(os.getenv("BATCH", "1"))   # number of random images


def build_random_payload(batch: int = 1) -> dict:
    """Return a JSON-serialisable dict with *batch* random MNIST images."""
    tensor = torch.randn(batch, 1, 28, 28)
    if batch == 1:
        data: List[float] = tensor.view(-1).tolist()
    else:
        data = [img.view(-1).tolist() for img in tensor]
    return {"data": data}


def main():
    log.info("Connecting to workspace …")
    ml_client = MLClient(DefaultAzureCredential(), SUBSCRIPTION, RESOURCE_GRP, WORKSPACE)

    log.info("Fetching endpoint metadata …")
    endpoint = ml_client.online_endpoints.get(ENDPOINT)
    scoring_uri = endpoint.scoring_uri

    # Allow user to supply the key via env to avoid RBAC requirement.
    key = os.getenv("AZ_ML_PRIMARY_KEY") or ml_client.online_endpoints.get_keys(ENDPOINT).primary_key

    log.info("Scoring URI: %s", scoring_uri)

    payload = build_random_payload(BATCH)
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {key}",
    }

    log.info("Sending request with batch=%d …", BATCH)
    response = requests.post(scoring_uri, headers=headers, data=json.dumps(payload))

    log.info("Status: %s", response.status_code)
    try:
        log.info("Response JSON: %s", response.json())
    except ValueError:
        log.error("Non-JSON response: %s", response.text[:300])


if __name__ == "__main__":
    main()