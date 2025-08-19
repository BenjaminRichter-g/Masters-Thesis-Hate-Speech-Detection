import argparse
import os
import json
from data.preprocessor import PreProcessor
from setfit import SetFitModel
from tqdm import tqdm

def classify_with_cutoffs(model, text, pos_cutoff=0.95, neg_cutoff=0.99):
    """
    Returns (predicted_label, prob_pos)
      predicted_label ∈ {1, 0, None}
        - 1 if p >= pos_cutoff              (confident positive)
        - 0 if p <= 1 - neg_cutoff          (confident negative, neg_cutoff is on (1-p))
        - None otherwise (abstain)
    """
    probs = model.predict_proba([text])[0]
    p_pos = float(probs[1])
    if p_pos >= pos_cutoff:
        return 1, p_pos
    if p_pos <= (1.0 - float(neg_cutoff)):
        return 0, p_pos
    return None, p_pos

def get_model_path(models_dir):
    """
    Returns dict of {model_name: model_path}
    model_name is derived from subfolder name (e.g. race-setfit-model → race)
    """
    model_paths = [
        os.path.join(models_dir, d)
        for d in os.listdir(models_dir)
        if d != "__pycache__" and os.path.isdir(os.path.join(models_dir, d))
    ]
    model_names = [os.path.basename(p).split('-')[0] for p in model_paths]  
    return dict(zip(model_names, model_paths))

def _infer_name_from_tau_json(obj):
    if "label_key" in obj and obj["label_key"]:
        return str(obj["label_key"]).strip()
    md = obj.get("model_dir")
    if md:
        base = os.path.basename(md.rstrip("/\\"))
        return base.split("-")[0]
    raise ValueError("Cannot infer model name from thresholds JSON; include 'label_key' or 'model_dir'.")

def load_per_model_cutoffs(tau_files):
    """
    Reads one or more JSON files (outputs of your threshold picker) and returns:
      { model_name: (tau_pos, tau_neg) }
    """
    mapping = {}
    if not tau_files:
        return mapping
    for path in tau_files:
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        name = _infer_name_from_tau_json(obj)
        tau_pos = float(obj["tau_pos"])
        tau_neg = float(obj["tau_neg"])
        mapping[name] = (tau_pos, tau_neg)
    return mapping

def run_all_models(models, input_file, output_file, default_pos=0.95, default_neg=0.99, per_model_cutoffs=None):
    per_model_cutoffs = per_model_cutoffs or {}
    processor = PreProcessor()
    count = 0
    skipped_language = 0
    errored = 0
    abstain_counts = {name: 0 for name in models}

    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'w', encoding='utf-8') as outfile:
        for line in tqdm(infile, desc="Processing"):
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                errored += 1
                continue
            if record.get("language") != "en":
                skipped_language += 1
                continue
            text = (record.get("clean_text") or "").strip()
            if not text:
                try:
                    cleaned = processor.preprocess(record.get("content", ""))
                    text = cleaned.get("clean_text", "") or ""
                except Exception:
                    errored += 1
                    continue
            if not text:
                errored += 1
                continue

            count += 1
            for model_name, model in models.items():
                pos_c, neg_c = per_model_cutoffs.get(model_name, (default_pos, default_neg))
                pred, conf = classify_with_cutoffs(model, text, pos_c, neg_c)
                record[f"{model_name}_prediction"] = pred  
                record[f"{model_name}_confidence"] = conf    
                if pred is None:
                    abstain_counts[model_name] += 1
            outfile.write(json.dumps(record) + "\n")

            if count % 400 == 0:
                print(f"Processed: {count} | Skipped (non-en): {skipped_language} | Errors: {errored}")
                print("Abstains per model:", abstain_counts)

def main():
    parser = argparse.ArgumentParser(
        description="Run SetFit models from specified model folders on the same input JSONL, "
                    "using per-model positive/negative cutoffs (τ_pos / τ_neg)."
    )
    parser.add_argument('--model_dirs', nargs='+', required=True,
                        help="List of model directories (e.g. data/third_stage/religion data/third_stage/gender)")
    parser.add_argument('--input', required=True, help="Input JSONL file")
    parser.add_argument('--output', required=True, help="Output JSONL file")
    parser.add_argument('--tau_files', nargs='*', default=[],
                        help="One or more JSON files each containing tau_pos/tau_neg and label_key/model_dir to map to a model")
    parser.add_argument('--pos_cutoff', type=float, default=0.95,
                        help="Fallback τ_pos (predict 1 if p >= τ_pos) for models without a tau JSON")
    parser.add_argument('--neg_cutoff', type=float, default=0.99,
                        help="Fallback τ_neg on (1-p); i.e., predict 0 if p <= 1-τ_neg for models without a tau JSON")

    args = parser.parse_args()

    all_models = {}
    for model_dir in args.model_dirs:
        model_paths = get_model_path(model_dir)  
        for name, path in model_paths.items():
            all_models[name] = SetFitModel.from_pretrained(path)

    per_model_cutoffs = load_per_model_cutoffs(args.tau_files)

    run_all_models(
        all_models,
        args.input,
        args.output,
        default_pos=args.pos_cutoff,
        default_neg=args.neg_cutoff,
        per_model_cutoffs=per_model_cutoffs
    )

if __name__ == "__main__":
    main()
