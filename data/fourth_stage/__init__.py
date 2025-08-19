import argparse
import json
from setfit import SetFitModel
from tqdm import tqdm

def classify_with_confidence(model, text, threshold=0.95):
    probs = model.predict_proba([text])[0]
    pred = int(probs[1] >= probs[0] and probs[1] > threshold)
    return pred, float(probs[1])  # return label and confidence for positive class

def load_models(model_dirs):
    models = {}
    for name, path in model_dirs.items():
        models[name] = SetFitModel.from_pretrained(path)
    return models

def run_all_models(models, input_file, output_file, threshold=0.95):
    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'w', encoding='utf-8') as outfile:

        for line in tqdm(infile, desc="Processing"):
            try:
                record = json.loads(line)
                text = record.get("text", "")

                for model_name, model in models.items():
                    pred, conf = classify_with_confidence(model, text, threshold)
                    record[f"{model_name}_prediction"] = pred
                    record[f"{model_name}_confidence"] = conf

                outfile.write(json.dumps(record) + "\n")
            except json.JSONDecodeError:
                continue

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--religious_model", required=True)
    parser.add_argument("--gender_model", required=True)
    parser.add_argument("--racial_model", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--confidence", type=float, default=0.95)

    args = parser.parse_args()

    model_dirs = {
        "religious": args.religious_model,
        "gender": args.gender_model,
        "racial": args.racial_model,
    }

    models = load_models(model_dirs)
    run_all_models(models, args.input, args.output, threshold=args.confidence)
