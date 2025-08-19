import json
from data.preprocessor import PreProcessor
from tqdm import tqdm

def classification_filter(input_path: str, out_discriminatory: str, out_non_discriminatory: str, model, is_discriminatory_fn, confidence):
    
    processor = PreProcessor()

    count = 0
    discriminatory = 0
    nondiscriminatory = 0
    errored_return = 0

    with open(input_path, 'r', encoding='utf-8') as infile, \
         open(out_discriminatory, 'w', encoding='utf-8') as discriminatory_out, \
         open(out_non_discriminatory, 'w', encoding='utf-8') as non_discriminatory_out:
        for line in tqdm(infile, desc="Filtering posts"):

            data = json.loads(line)
            if data.get('language') != 'en':
                continue

            cleaned = processor.preprocess(data['content'])

            try:
                content = cleaned['clean_text']
            except Exception as e:
                continue
            count+=1

            discrimination, proba_0, proba_1 = is_discriminatory_fn(model, content, confidence)
            

            if discrimination is None:
                errored_return+=1
            else:
                discrimination = bool(discrimination)
                data["pred_proba_0"] = proba_0
                data["pred_proba_1"] = proba_1
                if discrimination:
                    discriminatory_out.write(json.dumps(data) + '\n')
                    discriminatory+=1
                else:
                    non_discriminatory_out.write(json.dumps(data) + '\n')
                    nondiscriminatory+=1
            
            if count % 400 == 0:
                print(f"Total treated: {count}\nTotal discriminatorys: {discriminatory}\nTotal non-discriminatorys: {nondiscriminatory}")
    return count, discriminatory, nondiscriminatory
