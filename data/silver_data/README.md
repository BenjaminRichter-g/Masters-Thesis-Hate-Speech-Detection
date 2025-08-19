# Testing file to obtain performance metrics of the models

## First step

Extract fresh data to ensure no contamination, execute this within the folder you want the data in:

```
mastodoner instance --instance-url gameliberty.club --timeline output_silver.jsonl --limit 100000
```

10k was picked as with the natural distribution of data, the gold dataset will most likely be used more for false positives testing as
there is less than 1% of hate speech.

## Run then run this script to run the preprocessor

This will also apply each binary model on the data in order to find actual hate speech.
This is due to the low distribution of hate speech in the posts. In order to create as 
fair of a test as possible, an n amount of posts will be selected from mutliple confidence bins.
In practise this means that after the classifiers are applied to the dataset, an n amount of posts
will be selected from each classifiers in the 0-20, 20-40, 40-60, 60-80 and 80-100% confidence bins.
This will prevent an overfitting in the testing as the final bert models are trained on these binary classifiers.

```
python -m data.silver_data.sample_silver `
  --model_dirs data/third_stage/religion data/third_stage/gender data/third_stage/race `
  --input data/silver_data/output_silver.jsonl `
  --out_file data/silver_data/sample_merged.jsonl `
  --stats_out data/silver_data/silver_sample_stats.json `
  --per_bin 40 `
  --seed 42
```

## Run this to label the previously obtained dataset
```
python -m data.gold_silver_labeler `
  --input data/silver_data/sample_merged.jsonl `
  --output data/silver_data/silver_labeled.jsonl `
  --backup data/silver_data/backup.jsonl
```


