
To extract from the labeled dataset and create a subset for interannotor kappa score do the following:
```
python -m data.eval.kappa.sample_labeled `
  --input data/silver_data/silver_labeled.jsonl `
  --output data/eval/kappa/1st_annotator.jsonl `
  --per_class 40 `
  --seed 42
```

then label them:

```
python -m data.gold_silver_labeler `
  --input data/eval/kappa/1st_annotator.jsonl `
  --output data/eval/kappa/2nd_annotator.jsonl `
  --backup data/eval/kappa/2nd_backup.jsonl
```

then calculate cohen kappa score:

```
python -m data.eval.kappa.multi_single_cohen `
  --ann1 data/eval/kappa/1st_annotator.jsonl `
  --ann2 data/eval/kappa/2nd_annotator.jsonl `
  --out_json data/eval/kappa/reliability_summary.json `
  --boot_iters 3000
```