import argparse
from setfit import SetFitModel
import json
from data.classification_filter import classification_filter
import os
    
def classify_with_confidence(model, text: str, conf: float = 0.95):
    """
    Returns (predicted_label:int, confidence:float),
    where confidence is the probability of the chosen class.
    Start with extremely high confidence to filter out any amibigious posts 
    """
    probs = model.predict_proba([text])[0]
    if probs[1] >= probs[0] and probs[1] > conf:
        return 1, float(probs[0]), float(probs[1])
    else:
        return 0, float(probs[0]), float(probs[1])

    
def get_model_path(models_dir="data/first_stage", output="data/second_stage"):
    model_paths = [
    os.path.join(models_dir, d)
    for d in os.listdir(models_dir)
    if d != "__pycache__" and os.path.isdir(os.path.join(models_dir, d))
    ]
    model_names = [os.path.basename(p) for p in model_paths]
    model_outputs_discrim = [f'{output}/{name}_discriminatory.jsonl' for name in model_names]
    model_outputs_non_discrim =  [f'{output}/{name}_non_discriminatory.jsonl' for name in model_names]

    return model_names, model_paths, list(zip(model_outputs_discrim, model_outputs_non_discrim))
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Filter JSONL posts using a SetFit binary classifier for discriminatory."
    )
    
    parser.add_argument('--model_dir', required=True, help="Path to your trained SetFit model directory")
    parser.add_argument('--input', required=True, help="Input JSONL file")
    parser.add_argument('--output', required=True, help="Output directory for JSONL for labelled posts")
    parser.add_argument('--confidence', required=False, help="Level of confidence at which it gets labeled as true")
    args = parser.parse_args()
    
    model_names, model_paths, output_paths = get_model_path(args.model_dir, args.output)

    if args.confidence is None:
        confidence = 0.95
    else:
        confidence = float(args.confidence)

    final_data = []

    for i, model_path in enumerate(model_paths):
        model = SetFitModel.from_pretrained(model_path)
        values = classification_filter(args.input, output_paths[i][0],output_paths[i][1], model, classify_with_confidence, confidence)
        values = list(values)
        values.append(model_path)
        final_data.append(values)

    for count, nb_disc, nb_non_disc, model_path in final_data:
        print(f"""For the model: {model_path}\nTotal posts treated: {count}\nNb posts discriminatory {nb_disc}\nNb posts non-disc {nb_non_disc}""")

