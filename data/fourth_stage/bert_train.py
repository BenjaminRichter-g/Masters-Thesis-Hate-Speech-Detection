import os, argparse, multiprocessing, json
import numpy as np
import torch
from datasets import load_dataset
from transformers import (
    BertTokenizerFast,
    BertForSequenceClassification,
    Trainer,
    TrainingArguments,
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix

LABEL_NAMES = ["none", "race", "gender", "religion"]  

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    acc = accuracy_score(labels, preds)
    p_m, r_m, f1_m, _ = precision_recall_fscore_support(labels, preds, average="macro", zero_division=0)
    p_w, r_w, f1_w, _ = precision_recall_fscore_support(labels, preds, average="weighted", zero_division=0)
    return {
        "accuracy": acc,
        "f1_macro": f1_m,
        "precision_macro": p_m,
        "recall_macro": r_m,
        "f1_weighted": f1_w,
        "precision_weighted": p_w,
        "recall_weighted": r_w,
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_csv", required=True, help="CSV with columns: text,label")
    ap.add_argument("--out_dir", required=True, help="Directory to store checkpoints + metrics")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--train_bs", type=int, default=128)
    ap.add_argument("--eval_bs", type=int, default=128)
    ap.add_argument("--max_len", type=int, default=256)
    ap.add_argument("--num_workers", type=int, default=6)
    ap.add_argument("--fp16", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    logs_dir = os.path.join(args.out_dir, "logs")
    os.makedirs(logs_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

    ds = load_dataset("csv", data_files={"train": args.data_csv})["train"]

    def preprocess(batch):
        return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=args.max_len)

    ds = ds.map(preprocess, batched=True)
    ds = ds.train_test_split(test_size=0.2, seed=42)

    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=4).to(device)

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
        save_total_limit=2,
        dataloader_num_workers=args.num_workers,
        dataloader_pin_memory=True,
        fp16=args.fp16,
        report_to=["tensorboard"],   
        no_cuda=not torch.cuda.is_available(),
        seed=42,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds["train"],
        eval_dataset=ds["test"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
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
    y_pred = np.argmax(preds.predictions, axis=1)

    with open(os.path.join(args.out_dir, "classification_report.txt"), "w", encoding="utf-8") as f:
        f.write(classification_report(y_true, y_pred, target_names=LABEL_NAMES, digits=4))

    cm = confusion_matrix(y_true, y_pred)
    np.savetxt(os.path.join(args.out_dir, "confusion_matrix.csv"), cm, fmt="%d", delimiter=",")

    core = {
        "accuracy": float(eval_metrics.get("eval_accuracy", 0.0)),
        "f1_macro": float(eval_metrics.get("eval_f1_macro", 0.0)),
        "f1_weighted": float(eval_metrics.get("eval_f1_weighted", 0.0)),
        "best_model_checkpoint": trainer.state.best_model_checkpoint,
    }
    with open(os.path.join(args.out_dir, "eval_summary.json"), "w", encoding="utf-8") as f:
        json.dump(core, f, indent=2)

    train_loss_by_epoch = {}
    eval_loss_by_epoch = {}

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

    print(f"Saved per-epoch loss CSV → {csv_path}")

if __name__ == "__main__":
    multiprocessing.freeze_support()  
    main()
