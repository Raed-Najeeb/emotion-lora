# Emotion Classification with LoRA

## Objective

Investigate whether LoRA fine-tuning improves a pretrained language model's ability to classify emotions in text.

## Dataset

- **Source:** [Hugging Face — dair-ai/emotion](https://huggingface.co/datasets/dair-ai/emotion)
- **Task:** Six-class text classification
- **Labels:** sadness, joy, love, anger, fear, surprise

## Experimental Plan

1. Inspect and validate the dataset.
2. Establish a pretrained-model baseline.
3. Fine-tune the model using LoRA.
4. Compare baseline and fine-tuned performance on the same held-out test examples.
5. Analyze classification errors and document limitations.

## Current Status

Phase 2 — Dataset inspection.

Dataset statistics and data-quality findings will be added after running `src/inspect_data.py`.

## Reproducibility

Python dependencies will be documented as the project develops. Dataset source and evaluation procedures will be recorded in this repository.
