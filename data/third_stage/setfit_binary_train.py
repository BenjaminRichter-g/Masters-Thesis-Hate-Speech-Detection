import json
from datasets import Dataset
from setfit import SetFitModel, SetFitTrainer
from sentence_transformers.losses import CosineSimilarityLoss
from data.preprocessor import PreProcessor


def train_model(MODEL_DIR, data_origin, wait: bool = True):
    """
    Train & evaluate a SetFit gender discrimination model.
    If wait=True: yields the (accuracy: float), then expects a boolean .send()
      indicating whether to save the model.
    If wait=False: saves unconditionally before returning.
    """

    with open(data_origin, "r", encoding="utf-8") as f:
        examples = [json.loads(line) for line in f]

    pp = PreProcessor()
    cleaned = []
    for ex in examples:
        if ex.get("label") in (0,1) and ex.get("content"):
            txt = pp.preprocess(ex["content"])["clean_text"]
            cleaned.append({"text": txt, "label": ex["label"]})

    dataset = Dataset.from_list(cleaned).shuffle(seed=42)
    split   = dataset.train_test_split(test_size=0.2)
    train_ds, test_ds = split["train"], split["test"]

    model = SetFitModel.from_pretrained("sentence-transformers/paraphrase-mpnet-base-v2")
    trainer = SetFitTrainer(
        model=model,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        loss_class=CosineSimilarityLoss,
        batch_size=16,
        num_iterations=32,
        column_mapping={"text": "text", "label": "label"},
    )

    print("Starting trainer.train()")
    trainer.train()
    print("trainer.train() complete")
    acc = trainer.evaluate()["accuracy"]
    print(f"Trained. Eval accuracy = {acc:.2f}")

    if wait:
        should_save = yield acc
    else:
        should_save = True

    if should_save:
        trainer.model.save_pretrained(MODEL_DIR)
        print(f"Model saved to {MODEL_DIR}")
    else:
        print("Model not saved.")

    return acc


if __name__ == "__main__":
    gen = train_model(wait=False)

    try:
        next(gen)
    except StopIteration:
        pass
