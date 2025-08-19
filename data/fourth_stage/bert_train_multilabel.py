import os, argparse, multiprocessing, json
import numpy as np
import torch
from datasets import load_dataset
from transformers import (
    AutoConfig,
    BertTokenizerFast,
    BertForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)
from sklearn.metrics import (
    f1_score, precision_score, recall_score, accuracy_score,
    hamming_loss, classification_report
)

LABELS = ["race", "gender", "religion"]  

def compute_metrics_builder(threshold: float):
    def compute_metrics(eval_pred):
        logits, labels = eval_pred 
        probs = 1 / (1 + np.exp(-logits))
        preds = (probs >= threshold).astype(int)

        micro_p = precision_score(labels, preds, average="micro", zero_division=0)
        micro_r = recall_score(labels, preds, average="micro", zero_division=0)
        micro_f1 = f1_score(labels, preds, average="micro", zero_division=0)

        macro_p = precision_score(labels, preds, average="macro", zero_division=0)
        macro_r = recall_score(labels, preds, average="macro", zero_division=0)
        macro_f1 = f1_score(labels, preds, average="macro", zero_division=0)

        exact = (preds == labels).all(axis=1).mean()
        ham = hamming_loss(labels, preds)
        none_acc = ((labels.sum(axis=1) == 0) == (preds.sum(axis=1) == 0)).mean()

        return {
            "precision_micro": micro_p, "recall_micro": micro_r, "f1_micro": micro_f1,
            "precision_macro": macro_p, "recall_macro": macro_r, "f1_macro": macro_f1,
            "exact_match": exact, "hamming_loss": ham, "none_accuracy": none_acc
        }
    return compute_metrics

class WeightedTrainer(Trainer):
    """Adds optional pos_weight for BCEWithLogitsLoss."""
    def __init__(self, *args, pos_weight=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.pos_weight = pos_weight

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        if self.pos_weight is not None:
            pw = torch.tensor(self.pos_weight, device=logits.device, dtype=logits.dtype)
            loss_fct = torch.nn.BCEWithLogitsLoss(pos_weight=pw)
        else:
            loss_fct = torch.nn.BCEWithLogitsLoss()

        loss = loss_fct(logits, labels.type_as(logits))
        return (loss, outputs) if return_outputs else loss
    

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_csv", required=True, help="CSV with columns: text,race,gender,religion (0/1 each)")
    ap.add_argument("--out_dir", required=True, help="Where to store checkpoints + metrics")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--train_bs", type=int, default=32)
    ap.add_argument("--eval_bs", type=int, default=64)
    ap.add_argument("--max_len", type=int, default=256)
    ap.add_argument("--threshold", type=float, default=0.5, help="Eval threshold on sigmoid outputs")
    ap.add_argument("--num_workers", type=int, default=6)
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--pos_weight", type=float, nargs=3, default=None,
                    help="Optional BCE pos_weight per class, e.g. --pos_weight 3.0 2.0 4.0 for [race, gender, religion]")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    logs_dir = os.path.join(args.out_dir, "logs"); os.makedirs(logs_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

    ds_all = load_dataset("csv", data_files={"train": args.data_csv})["train"]

    def to_labels(batch):
        batch["labels"] = [
            int(batch["race"]), int(batch["gender"]), int(batch["religion"])
        ]
        return batch

    def tok(batch):
        return tokenizer(batch["text"], truncation=True, max_length=args.max_len)

    ds_all = ds_all.map(to_labels)
    ds_all = ds_all.map(tok, batched=True)

    ds = ds_all.train_test_split(test_size=0.2, seed=42)
    keep = {"input_ids", "attention_mask", "labels"}
    ds = ds.remove_columns([c for c in ds["train"].column_names if c not in keep])

    collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8 if args.fp16 else None)

    config = AutoConfig.from_pretrained("bert-base-uncased", num_labels=3)
    config.problem_type = "multi_label_classification"
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", config=config).to(device)

    training_args = TrainingArguments(
        output_dir=args.out_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",       
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        learning_rate=2e-5,
        per_device_train_batch_size=args.train_bs,
        per_device_eval_batch_size=args.eval_bs,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        logging_dir=logs_dir,
        logging_steps=50,
        save_total_limit=2,
        dataloader_num_workers=args.num_workers,
        dataloader_pin_memory=True,
        fp16=args.fp16,
        report_to="none",                
        no_cuda=not torch.cuda.is_available(),
        seed=42,
    )

    compute_metrics = compute_metrics_builder(args.threshold)

    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=ds["train"],
        eval_dataset=ds["test"],
        processing_class=tokenizer,      
        data_collator=collator,
        compute_metrics=compute_metrics,
        pos_weight=args.pos_weight,
    )

    train_result = trainer.train()
    trainer.log_metrics("train", train_result.metrics)
    trainer.save_metrics("train", train_result.metrics)
    trainer.save_state()

    eval_metrics = trainer.evaluate()
    trainer.log_metrics("eval", eval_metrics)
    trainer.save_metrics("eval", eval_metrics)

    best_dir = os.path.join(args.out_dir, "best")
    trainer.save_model(best_dir)
    tokenizer.save_pretrained(best_dir)

    preds = trainer.predict(ds["test"])
    y_true = preds.label_ids
    probs = 1 / (1 + np.exp(-preds.predictions))
    y_pred = (probs >= args.threshold).astype(int)

    with open(os.path.join(args.out_dir, "classification_report.txt"), "w", encoding="utf-8") as f:
        f.write(classification_report(y_true, y_pred, target_names=LABELS, digits=4, zero_division=0))

    core = {
        "f1_macro": float(eval_metrics.get("eval_f1_macro", 0.0)),
        "f1_micro": float(eval_metrics.get("eval_f1_micro", 0.0)),
        "exact_match": float(eval_metrics.get("eval_exact_match", 0.0)),
        "hamming_loss": float(eval_metrics.get("eval_hamming_loss", 0.0)),
        "threshold": args.threshold,
        "best_model_checkpoint": trainer.state.best_model_checkpoint,
    }
    with open(os.path.join(args.out_dir, "eval_summary.json"), "w", encoding="utf-8") as f:
        json.dump(core, f, indent=2)

    train_loss_by_epoch, eval_loss_by_epoch = {}, {}
    for rec in trainer.state.log_history:
        if "epoch" in rec:
            ep = int(round(float(rec["epoch"])))
            if "loss" in rec:
                train_loss_by_epoch[ep] = float(rec["loss"])
            if "eval_loss" in rec:
                eval_loss_by_epoch[ep] = float(rec["eval_loss"])

    csv_path = os.path.join(args.out_dir, "loss_history.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("epoch,train_loss,eval_loss\n")
        for ep in sorted(set(list(train_loss_by_epoch.keys()) + list(eval_loss_by_epoch.keys()))):
            t = train_loss_by_epoch.get(ep, "")
            v = eval_loss_by_epoch.get(ep, "")
            f.write(f"{ep},{t},{v}\n")
    print(f"[i] Saved per-epoch loss CSV → {csv_path}")

if __name__ == "__main__":
    multiprocessing.freeze_support()  
    main()
