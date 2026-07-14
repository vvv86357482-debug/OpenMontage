"""Kaggle batch generation tools.

Architecture: Codespace writes job specifications → submits ONE Kaggle kernel
push per project → kernel loads model once, generates ALL assets for that
project in one session → Codespace polls until complete → downloads.

This package provides:
- kaggle_image: image generation via Kaggle kernels (SANA, FLUX)
- kaggle_video: video generation via Kaggle kernels
- kernel: helper utilities for kernel push/poll/download lifecycle
"""
