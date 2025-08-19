import json
import os
import argparse

def dedupe_jsonl(input_path: str, output_path: str, backup: bool = True):
    """
    Read JSONL from input_path, remove duplicate records based on 'id',
    and write deduped lines to output_path. Keeps first occurrence.
    If backup is True and output_path exists, renames it to output_path+".bak".
    """
    seen_ids = set()
    lines_out = []

    with open(input_path, 'r', encoding='utf-8') as f:
        for ln in f:
            try:
                obj = json.loads(ln)
            except json.JSONDecodeError:
                continue
            rec_id = obj.get("id")
            if rec_id is None:
                lines_out.append(ln)
            elif rec_id not in seen_ids:
                seen_ids.add(rec_id)
                lines_out.append(ln)

    if backup and os.path.exists(output_path):
        bak_path = output_path + ".bak"
        os.replace(output_path, bak_path)
        print(f"Backed up old output to {bak_path}")

    with open(output_path, 'w', encoding='utf-8') as f:
        for ln in lines_out:
            f.write(ln)

    print(f"Wrote {len(lines_out):,} unique records to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Remove duplicate entries from a JSONL file based on 'id'."
    )
    parser.add_argument("input",  help="path to the labeled JSONL file")
    parser.add_argument("output", help="path to write deduplicated JSONL")
    parser.add_argument(
        "--no-backup",
        action="store_false",
        dest="backup",
        help="don't backup the existing output file"
    )
    args = parser.parse_args()

    dedupe_jsonl(args.input, args.output, backup=args.backup)
