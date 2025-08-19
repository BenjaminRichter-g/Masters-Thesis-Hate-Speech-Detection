import argparse
import numpy as np
import torch
from datasets import load_dataset
from transformers import BertTokenizerFast, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix

LABEL_NAMES = ["none", "race", "gender", "religion"]  # 0..3

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    acc = accuracy_score(labels, preds)
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(labels, preds, average="macro", zero_division=0)
    p_w, r_w, f1_w, _ = precision_recall_fscore_support(labels, preds, average="weighted", zero_division=0)
    return {
        "accuracy": acc,
        "precision_macro": p_macro,
        "recall_macro": r_macro,
        "f1_macro": f1_macro,
        "precision_weighted": p_w,
        "recall_weighted": r_w,
        "f1_weighted": f1_w,
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True, help="Path to fine-tuned model (e.g., ./bert-discrimination/checkpoint-XXXX)")
    ap.add_argument("--eval_csv",   required=True, help="CSV file with columns: text,label")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--max_length", type=int, default=128)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Evaluating on device:", device)

    tokenizer = BertTokenizerFast.from_pretrained(args.model_dir)
    model = BertForSequenceClassification.from_pretrained(args.model_dir, num_labels=4).to(device)

    ds = load_dataset("csv", data_files={"eval": args.eval_csv})
    def preprocess(ex):
        return tokenizer(ex["text"], truncation=True, padding="max_length", max_length=args.max_length)
    ds = ds.map(preprocess, batched=True)
    eval_ds = ds["eval"].remove_columns([c for c in ds["eval"].column_names if c not in {"input_ids","attention_mask","label"}])
    eval_ds.set_format(type="torch")

    training_args = TrainingArguments(
        output_dir="./eval_tmp",
        per_device_eval_batch_size=args.batch_size,
        dataloader_num_workers=0,  
        pin_memory=True,
        no_cuda=not torch.cuda.is_available(),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        eval_dataset=eval_ds,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    metrics = trainer.evaluate()
    print("\n=== Eval Metrics ===")
    for k, v in metrics.items():
        if k.startswith("eval_"):
            print(f"{k}: {v}")

    preds_out = trainer.predict(eval_ds)
    y_true = preds_out.label_ids
    y_pred = np.argmax(preds_out.predictions, axis=1)

    print("\n=== Classification Report ===")
    print(classification_report(y_true, y_pred, target_names=LABEL_NAMES, digits=4))

    print("=== Confusion Matrix (rows=true, cols=pred) ===")
    print(confusion_matrix(y_true, y_pred))

if __name__ == "__main__":
    main()
