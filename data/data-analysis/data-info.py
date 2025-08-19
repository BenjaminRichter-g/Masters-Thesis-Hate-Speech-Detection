import os
import json
from data.preprocessor import PreProcessor


path = "data/first_stage/output_timeline.jsonl"
pp = PreProcessor()
    
if not os.path.exists(path):
    print(f"[!] {path} not found.")

with open(path, 'r', encoding='utf-8') as f:
    lines = f.readlines()
    non_empty_count = 0
    processed = 0

    for line in lines:
        data = json.loads(line)
        text = pp.preprocess(data['content'])
        try:
            text = text['clean_text']
            if text != "" and data.get("language")=="en":
                non_empty_count+=1
        except Exception as e:
            non_empty_count+=1
        
        processed+=1

        if processed%1000==0:
            print(f"{processed}/100 000")
        

print(f"There are {non_empty_count} usable post with text content")
