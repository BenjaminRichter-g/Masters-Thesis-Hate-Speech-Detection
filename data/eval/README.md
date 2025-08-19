## To execute the eval of the bert model

### Gold dataset

```
python -m data.eval.eval_single_bert `
  --model_path data/fourth_stage/bert_output/best `
  --input_jsonl data/gold_data/gold_labeled.jsonl `
  --out_dir eval_results/single_on_gold `
  --batch_size 128 --max_len 256
```

### Silver dataset

```
python -m data.eval.eval_single_bert `
  --model_path data/fourth_stage/bert_output/best `
  --input_jsonl data/silver_data/silver_labeled.jsonl `
  --out_dir eval_results/single_on_silver `
  --batch_size 128 --max_len 256
```

## To execute the eval of the multilabel bert model

### Gold dataset

```
python -m data.eval.eval_multi_bert `
  --model_path data/fourth_stage/bert_multilabel_output/best `
  --input_jsonl data/gold_data/gold_labeled.jsonl `
  --out_dir eval_results/mutli_on_gold `
  --batch_size 128 --max_len 256
```

### Silver dataset

```
python -m data.eval.eval_multi_bert `
  --model_path data/fourth_stage/bert_multilabel_output/best `
  --input_jsonl data/silver_data/silver_labeled.jsonl `
  --out_dir eval_results/mutli_on_silver `
  --batch_size 128 --max_len 256
```