from data.preprocessor  import PreProcessor
from tqdm import tqdm
import json
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Returns json with nothing but the content to facilitate reading"
    )
    parser.add_argument('--input', required=True, help="Input JSONL file")
    parser.add_argument('--output', required=True, help="Output JSONL file containing only content")
    args = parser.parse_args()

    processor = PreProcessor()

    with open(args.input, 'r', encoding='utf-8') as infile, \
            open(args.output, 'w', encoding='utf-8') as racist_out:
        for line in tqdm(infile, desc="Filtering posts"):
            data = json.loads(line)
            if data.get('language') != 'en':
                continue

            cleaned = processor.preprocess(data['content'])
            cleaned['id'] = data['id']

            racist_out.write(json.dumps(cleaned) + '\n')


