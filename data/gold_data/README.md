# Testing file to obtain performance metrics of the models

## First step

Extract fresh data to ensure no contamination, execute this within the folder you want the data in:

```
mastodoner instance --instance-url gameliberty.club --timeline output_timeline.jsonl --limit 10000
```

10k was picked as with the natural distribution of data, the gold dataset will most likely be used more for false positives testing as
there is less than 1% of hate speech.

## Run then run this script to run the preprocessor and randomly sample datapoints for a true gold set

```
python -m data.gold_data.sample_gold `
  --input data/gold_data/output_timeline.jsonl `
  --output data/gold_data/gold_sample.jsonl `
  --n 600 `
  --seed 42
```

## Run this to label the previously obtained dataset
```
python -m data.gold_silver_labeler `
  --input data/gold_data/gold_sample.jsonl `
  --output data/gold_data/gold_labeled.jsonl `
  --backup data/gold_data/backup.jsonl
```
