
import argparse

import torch
from peft import PeftModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer

BASE_MODEL = "distilbert/distilbert-base-uncased"

LABEL_NAMES = ["sadness", "joy", "love", "anger", "fear", "surprise"]


def load_model(model_type, device):
    if model_type == "baseline":
        model_path = "models/baseline"

        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path)

    else:
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

        base_model = AutoModelForSequenceClassification.from_pretrained(
            BASE_MODEL,
            num_labels=len(LABEL_NAMES),
            id2label={i: name for i, name in enumerate(LABEL_NAMES)},
            label2id={name: i for i, name in enumerate(LABEL_NAMES)},
        )

        model = PeftModel.from_pretrained(base_model, "models/lora")

    model.to(device)
    model.eval()

    return model, tokenizer


def predict(text, model, tokenizer, device):
    if not text.strip():
        raise ValueError("Input text cannot be empty.")

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=128,
    )

    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.inference_mode():
        logits = model(**inputs).logits
        probabilities = torch.softmax(logits, dim=-1)[0]

    predicted_id = int(torch.argmax(probabilities).item())

    return LABEL_NAMES[predicted_id], probabilities.cpu().tolist()


def main():
    parser = argparse.ArgumentParser(
        description="Predict emotions using the fine-tuned DistilBERT models."
    )
    parser.add_argument(
        "--model",
        choices=["baseline", "lora"],
        default="lora",
        help="Choose the model: full fine-tuning baseline or LoRA.",
    )
    parser.add_argument(
        "--text",
        type=str,
        help="Text to classify. Omit this argument for interactive mode.",
    )
    args = parser.parse_args()

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print(f"Device: {device}")
    print(f"Loading {args.model} model...")

    model, tokenizer = load_model(args.model, device)

    def show_prediction(text):
        emotion, probabilities = predict(text, model, tokenizer, device)

        print(f"\nText: {text}")
        print(f"Predicted emotion: {emotion}")
        print("Class probabilities:")

        for label, probability in zip(LABEL_NAMES, probabilities):
            print(f"  {label:10s} {probability:.2%}")

    if args.text is not None:
        show_prediction(args.text)
        return

    print("\nInteractive emotion classifier")
    print("Enter a sentence, or type 'quit' to exit.")

    while True:
        try:
            text = input("\nText: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if text.lower() in {"quit", "exit"}:
            break

        if not text:
            print("Please enter a non-empty sentence.")
            continue

        show_prediction(text)


if __name__ == "__main__":
    main()
