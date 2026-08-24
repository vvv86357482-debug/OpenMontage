#!/usr/bin/env bash
# Push a kernel dir (sana_sprint | omnivoice) to Kaggle.
# Auth: KAGGLE_API_TOKEN (kaggle CLI >= 2.x reads it straight from env).
# Substitutes __KAGGLE_USERNAME__ into kernel-metadata.json id.
set -euo pipefail

dir="tools/kaggle/$1"
: "${KAGGLE_USERNAME:?KAGGLE_USERNAME not set - add it as a Codespace secret, then stop+start this codespace}"
: "${KAGGLE_API_TOKEN:?KAGGLE_API_TOKEN not set - add it as a Codespace secret, then stop+start this codespace}"

sed -i "s/__KAGGLE_USERNAME__/${KAGGLE_USERNAME}/g" "$dir/kernel-metadata.json"
kaggle kernels push -p "$dir"
