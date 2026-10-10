
from datasets import load_dataset


def main():
    dataset = load_dataset("dair-ai/emotion")

    print("\n=== DATASET OVERVIEW ===")
    print(dataset)
    print("\nColumns:", dataset["train"].column_names)

    label_names = dataset["train"].features["label"].names
    print("\nLabel mapping:")
    for label_id, label_name in enumerate(label_names):
        print(f"  {label_id}: {label_name}")

    print("\n=== SPLIT SIZES ===")
    for split_name, split in dataset.items():
        print(f"{split_name}: {len(split)} examples")

    print("\n=== LABEL DISTRIBUTION ===")
    for split_name, split in dataset.items():
        counts = {name: 0 for name in label_names}

        for label_id in split["label"]:
            counts[label_names[label_id]] += 1

        print(f"\n{split_name}:")
        for label_name, count in counts.items():
            print(f"  {label_name}: {count}")

    print("\n=== SAMPLE EXAMPLES ===")
    for example in dataset["train"].select(range(5)):
        label = label_names[example["label"]]
        print(f"\nText: {example['text']}")
        print(f"Label: {label}")

    print("\n=== DATA QUALITY CHECKS ===")
    for split_name, split in dataset.items():
        texts = split["text"]
        missing = sum(
            not isinstance(text, str) or not text.strip()
            for text in texts
        )
        duplicates = len(texts) - len(set(texts))

        print(f"\n{split_name}:")
        print(f"  Missing/blank text: {missing}")
        print(f"  Repeated text entries: {duplicates}")

    print("\n=== CROSS-SPLIT OVERLAP ===")
    split_names = list(dataset.keys())

    for i in range(len(split_names)):
        for j in range(i + 1, len(split_names)):
            first_name = split_names[i]
            second_name = split_names[j]

            first_texts = set(dataset[first_name]["text"])
            second_texts = set(dataset[second_name]["text"])
            overlap = first_texts & second_texts

            print(
                f"{first_name} vs {second_name}: "
                f"{len(overlap)} shared unique texts"
            )

    print("\nInspection complete.")


if __name__ == "__main__":
    main()
