
import json
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

MODEL_NAME = "distilbert/distilbert-base-uncased"
OUTPUT_DIR = Path("models/baseline")
RESULTS_DIR = Path("results")
MAX_LENGTH = 128

LABEL_NAMES = ["sadness", "joy", "love", "anger", "fear", "surprise"]


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        predictions,
        average="macro",
        zero_division=0,
    )

    return {
        "accuracy": accuracy_score(labels, predictions),
        "macro_precision": precision,
        "macro_recall": recall,
        "macro_f1": f1,
    }


def main():
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable. Check your PyTorch GPU installation."
        )

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Loading dataset and tokenizer: {MODEL_NAME}")

    dataset = load_dataset("dair-ai/emotion")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=MAX_LENGTH,
        )

    tokenized = dataset.map(tokenize, batched=True)
    tokenized = tokenized.rename_column("label", "labels")

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=6,
        id2label={i: name for i, name in enumerate(LABEL_NAMES)},
        label2id={name: i for i, name in enumerate(LABEL_NAMES)},
    )

    args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        num_train_epochs=3,
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_macro_f1",
        greater_is_better=True,
        fp16=True,
        logging_steps=100,
        save_total_limit=2,
        report_to="none",
        seed=42,
        data_seed=42,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
    )

    print("\nStarting full fine-tuning baseline...")
    trainer.train()

    print("\nEvaluating on the held-out test set...")
    test_metrics = trainer.evaluate(
        eval_dataset=tokenized["test"],
        metric_key_prefix="test",
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    trainer.save_model(str(OUTPUT_DIR))
    tokenizer.save_pretrained(str(OUTPUT_DIR))

    clean_metrics = {
        key: float(value)
        for key, value in test_metrics.items()
        if isinstance(value, (int, float, np.number))
    }

    with open(RESULTS_DIR / "baseline_metrics.json", "w") as f:
        json.dump(clean_metrics, f, indent=2)

    print("\nBaseline test metrics:")
    print(json.dumps(clean_metrics, indent=2))
    print(f"\nModel saved to: {OUTPUT_DIR}")
    print(f"Metrics saved to: {RESULTS_DIR / 'baseline_metrics.json'}")


if __name__ == "__main__":
    main()
