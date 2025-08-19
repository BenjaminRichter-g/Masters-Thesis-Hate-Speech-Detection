import subprocess
import sys
import shutil
from pathlib import Path
import argparse
from data.third_stage.setfit_binary_train import train_model

THRESHOLD         = 30
RANDOMIZE_LABELS  = True

def count_lines(path: Path) -> int:
    return path.exists() and sum(1 for _ in path.open("r", encoding="utf-8")) or 0

def run(cmd):
    print(f"\n {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

def read_best_acc() -> float:
    if BEST_ACC_FILE.exists():
        try:
            return float(BEST_ACC_FILE.read_text().strip())
        except:
            return 0.0
    return 0.0

def write_best_acc(acc: float):
    BEST_ACC_FILE.parent.mkdir(exist_ok=True, parents=True)
    BEST_ACC_FILE.write_text(f"{acc:.4f}")

def main(confidence):
    PROBABILITY_BINS  = False
    confidence = confidence
    number_labeled = count_lines(MASTER_FILE)
    if number_labeled > 30 and not PROBABILITY_BINS:
        ans = input(f"{number_labeled} labeled sampled, do you want to switch to probability interval bins to prevent overfitting [y/N]:").strip().lower()
        if ans == "y":
            PROBABILITY_BINS = True
            confidence = 0.5
    while True:
        new_count = count_lines(NEW_FILE)
        print(f"\n🔎 {new_count} new labels (threshold = {THRESHOLD})")
        ans = input(f"Do you want to force a retrain? [y/N]: ").strip().lower()
        if ans == "y":
            new_count = THRESHOLD

        if new_count >= THRESHOLD:
            ans = input(f"You’ve labeled ≥{THRESHOLD} items. Retrain now? [y/N]: ").strip().lower()
            if ans == "y":
                tmp = MASTER_FILE.parent / "master_tmp.jsonl"
                if tmp.exists():
                    tmp.unlink()

                with open(tmp, "wb") as out_f:
                    if MASTER_FILE.exists():
                        out_f.write(MASTER_FILE.read_bytes())
                    if NEW_FILE.exists():
                        out_f.write(NEW_FILE.read_bytes())

                run([
                    sys.executable,
                    "-m", "data.third_stage.unique_json",
                    str(tmp),        
                    str(MASTER_FILE) 
                ])

                
                gen = train_model(MODEL_DIR, MASTER_FILE, wait=True)
                new_acc = next(gen) 
                print(f"New model eval accuracy = {new_acc:.2f}")

                run([
                    sys.executable, "-m", "data.third_stage.setfit_binary",
                    "--model_dir", str(MODEL_DIR.parent),
                    "--input",     str(FULL_INPUT),
                    "--output",    str(FULL_OUTPUT_DIR),
                    "--confidence", str(confidence)
                ])

                run([
                    sys.executable, "-m", "data.third_stage.remove_dups",
                    "--source", str(LABEL_HATE),
                    "--filter",     str(MASTER_FILE),
                ])

                if NEW_FILE.exists():
                    NEW_FILE.unlink()
                NEW_FILE.write_text("")
                print("\nRetraining cycle complete. Back to labeling.")
            else:
                print("⏸ Skipping retrain.")

        else:
            print(f"…not enough new labels yet (<{THRESHOLD}).")

        if not PROBABILITY_BINS:
            cmd = [
                sys.executable, "-m", "data.third_stage.manual_labelling",
                "--clean",  str(LABEL_CLEAN),
                "--hate",   str(LABEL_HATE),
                "--output", str(LABEL_OUT),
                "--backup", str(LABEL_BACKUP),
                "--master", str(MASTER_FILE)
            ]
        else:
            cmd = [
                sys.executable, "-m", "data.third_stage.manual_labelling",
                "--mode", "bins",
                "--per_bin", str(10),
                "--round_robin",
                "--clean",  str(LABEL_CLEAN),
                "--hate",   str(LABEL_HATE),
                "--output", str(LABEL_OUT),
                "--backup", str(LABEL_BACKUP),
                "--master", str(MASTER_FILE)
            ]


        if RANDOMIZE_LABELS and not PROBABILITY_BINS:
            cmd.append("--randomize")
        run(cmd)

        print("\nLooping back…\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Automatically updating system mixing labelling and retraining"
    )
    p.add_argument("--master_file",     required=True, help="path to clean posts JSONL")
    p.add_argument("--new_file",      required=True, help="path to discriminatory posts JSONL")
    p.add_argument("--model_dir",    required=True, help="where to append your new labels")
    p.add_argument("--full_input",    required=True, help="where to store a backup of existing output")
    p.add_argument("--full_output_dir",    required=True, help="where to store a backup of existing output")
    p.add_argument("--label_clean",    required=True, help="where to store a backup of existing output")
    p.add_argument("--label_hate",    required=True, help="where to store a backup of existing output")
    p.add_argument("--label_out",    required=True, help="where to store a backup of existing output")
    p.add_argument("--label_backup",    required=True, help="where to store a backup of existing output")
    p.add_argument("--confidence", help = "If the early binary models overfit, you can lower the conf threshold to accept more items into the class")
    args = p.parse_args()
        
    MASTER_FILE       = Path(args.master_file)
    NEW_FILE          = Path(args.new_file)
    MODEL_DIR         = Path(args.model_dir)
    BEST_ACC_FILE     = MODEL_DIR.parent / "best_accuracy.txt"

    FULL_INPUT        = Path(args.full_input)
    FULL_OUTPUT_DIR   = Path(args.full_output_dir)

    LABEL_CLEAN       = Path(args.label_clean)
    LABEL_HATE        = Path(args.label_hate)
    LABEL_OUT         = Path(args.label_out)
    LABEL_BACKUP      = Path(args.label_backup)

    if args.confidence is None:
        CONF = "0.95"
    else:
        CONF = args.confidence

    if not LABEL_CLEAN.is_file():

        gen = train_model(MODEL_DIR, MASTER_FILE, wait=True)
        new_acc = next(gen) 
        print(f"New model eval accuracy = {new_acc:.2f}")

        print("Generating first classification based off of labels from step 2")
        
        run([
            sys.executable, "-m", "data.third_stage.setfit_binary",
            "--model_dir", str(MODEL_DIR.parent),
            "--input",     str(FULL_INPUT),
            "--output",    str(FULL_OUTPUT_DIR)
        ])

    main(CONF)