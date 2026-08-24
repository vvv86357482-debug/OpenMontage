#!/usr/bin/env bash
# Push a kernel dir (sana_sprint | omnivoice) to Kaggle.
# Auth (either): KAGGLE_API_TOKEN env var OR ~/.kaggle/access_token file.
# Substitutes __KAGGLE_USERNAME__ into kernel-metadata.json id when needed.
set -euo pipefail

dir="tools/kaggle/$1"
meta="$dir/kernel-metadata.json"

if grep -q "__KAGGLE_USERNAME__" "$meta"; then
  : "${KAGGLE_USERNAME:?KAGGLE_USERNAME not set and metadata still has placeholder}"
  sed -i "s/__KAGGLE_USERNAME__/${KAGGLE_USERNAME}/g" "$meta"
fi

if [[ -z "${KAGGLE_API_TOKEN:-}" && ! -f "$HOME/.kaggle/access_token" ]]; then
  echo "No KAGGLE_API_TOKEN env and no ~/.kaggle/access_token - cannot authenticate" >&2
  exit 1
fi

kaggle kernels push -p "$dir"
