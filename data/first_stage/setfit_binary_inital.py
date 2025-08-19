import argparse
import json
from setfit import SetFitModel, SetFitTrainer
from datasets import Dataset
from sentence_transformers.losses import CosineSimilarityLoss

def main():
    parser = argparse.ArgumentParser(
        description="Train one SetFit model per label (or per-name) and save each separately."
    )
    parser.add_argument(
        "-n", "--model-names",
        nargs="+",
        help=(
            "Optional list of model names, in the same order as the sorted integer labels. "
            "If provided and length matches, models will be saved under these names "
            "instead of the numeric labels."
        )
    )
    args = parser.parse_args()
    with open("data/first_stage/labeled.jsonl", "r", encoding="utf-8") as f:
        examples = [json.loads(line) for line in f]

    from data.preprocessor import PreProcessor
    processor = PreProcessor()

    keys = {int(ex.get("label")) for ex in examples}
    cleaned_examples = {label: [] for label in keys}
    for ex in examples:
        label = ex.get("label")
        if label in keys and ex.get("content"):
            cleaned_text = processor.preprocess(ex["content"])["clean_text"]
            if label != 0:
                cleaned_examples[label].append({
                    "text": cleaned_text,
                    "label": int(ex["label"])
                })
            else: 
                for key in keys:
                    if key == 0:
                        continue
                    cleaned_examples[key].append({
                    "text": cleaned_text,
                    "label": int(ex["label"])
                })
        
    keys.remove(0)
    datasets = {label: Dataset.from_list(cleaned_examples[label]).shuffle(seed=42) for label in keys}

    splits = {label: datasets[label].train_test_split(test_size=0.2)  for label in datasets}

    models = {label: SetFitModel.from_pretrained("sentence-transformers/paraphrase-mpnet-base-v2")  for label in keys}

    trainers = {}
    for label, model in models.items():
        trainers[label] = SetFitTrainer(
                            model=model,
                            train_dataset=splits[label]["train"],
                            eval_dataset=splits[label]["test"],
                            loss_class=CosineSimilarityLoss,
                            batch_size=8,
                            num_iterations=8,
                            column_mapping={"text": "text", "label": "label"},
                        )

    if len(args.model_names) == len(models):
        model_name = {label:args.model_names[label-1] for label in trainers}
    else:
        model_name = {label:label for label in trainers}
        print("number of classes and names don't match, default naming applied")

    for label, trainer in trainers.items():
        trainer.train()
        acc = trainer.evaluate()["accuracy"]
        name = model_name[label]
        print(f"{name} → ✅ Accuracy: {acc:.2f}")
        trainer.model.save_pretrained(f"data/first_stage/{name}-initial-model")


if __name__ == "__main__":
    main()

