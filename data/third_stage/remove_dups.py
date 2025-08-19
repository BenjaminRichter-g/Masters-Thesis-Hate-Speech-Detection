"""
Script: remove_jsonl_duplicates.py

Reads two JSONL files and removes from the source file any entries whose IDs appear
in the filter file, overwriting the source file.
"""
import json
import argparse
import os
import tempfile

def parse_args():
    parser = argparse.ArgumentParser(
        description='Remove entries from a source JSONL whose IDs appear in the filter JSONL.'
    )
    parser.add_argument(
        '--source', '-s',
        required=True,
        help='Path to the source JSONL file to be filtered (will be overwritten).'
    )
    parser.add_argument(
        '--filter', '-f',
        required=True,
        help='Path to the JSONL file containing IDs to filter out.'
    )
    return parser.parse_args()

def main():
    args = parse_args()

    filter_ids = set()
    with open(args.filter, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if 'id' in record:
                filter_ids.add(record['id'])

    dirpath = os.path.dirname(os.path.abspath(args.source)) or '.'
    fd, temp_path = tempfile.mkstemp(dir=dirpath)
    os.close(fd)

    kept_lines = 0
    with open(args.source, 'r', encoding='utf-8') as src, open(temp_path, 'w', encoding='utf-8') as out_f:
        for line in src:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                out_f.write(line)
                continue

            if record.get('id') not in filter_ids:
                kept_lines+=1
                out_f.write(line)

    os.replace(temp_path, args.source)
    print(f"Filtered JSONL has overwritten: {args.source}")
    print(f"{kept_lines} number of lines kept in file after filtering")

if __name__ == '__main__':
    main()
