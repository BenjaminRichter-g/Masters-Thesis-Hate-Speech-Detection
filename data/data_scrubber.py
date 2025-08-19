import argparse, json, os, tempfile

REMOVE_KEYS = {
    "in_reply_to_id", "in_reply_to_account_id", "sensitive", "spoiler_text",
    "visibility", "uri", "url", "replies_count", "reblogs_count",
    "favourites_count", "edited_at", "reblog", "account", "media_attachments",
    "mentions", "tags", "emojis", "card", "poll"
}

def iter_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            s = ln.strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
                yield obj
            except json.JSONDecodeError:
                continue

def main():
    ap = argparse.ArgumentParser(description="In-place scrub JSONL: remove specific top-level fields.")
    ap.add_argument("--input", required=True, help="Path to the JSONL file to scrub (overwritten in place).")
    args = ap.parse_args()

    in_path = os.path.abspath(args.input)
    in_dir  = os.path.dirname(in_path) or "."

    total = 0
    skipped = 0
    keys_removed = 0

    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=in_dir, suffix=".tmp", delete=False) as tmp:
        tmp_path = tmp.name
        for obj in iter_jsonl(in_path):
            total += 1
            if not isinstance(obj, dict):
                skipped += 1
                continue
            for k in REMOVE_KEYS:
                if k in obj:
                    obj.pop(k, None)
                    keys_removed += 1
            tmp.write(json.dumps(obj, ensure_ascii=False) + "\n")

    os.replace(tmp_path, in_path)

    print(f"Scrubbed in place → {in_path}")
    print(f"Total lines read: {total:,}")
    print(f"Non-dict/invalid skipped: {skipped:,}")
    print(f"Keys removed (total occurrences): {keys_removed:,}")

if __name__ == "__main__":
    main()
