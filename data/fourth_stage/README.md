## Run the script to label the dataset with the three models:

### IMPORTANT NOTE
Whilst its possible to run without the threshold sweeps with the default cutoff from the command below, 
its extremely recommended to run the threshold sweep scripts. Please refer to data/eval/metrics/README.md to generate them

```
python -m data.fourth_stage.setfit_multi_model `
  --model_dirs data/third_stage/race data/third_stage/gender data/third_stage/religion `
  --input data/first_stage/output_timeline.jsonl `
  --output data/fourth_stage/predicted.jsonl `
  --tau_files data/eval/metrics/race/t_race.json data/eval/metrics/gender/t_gender.json data/eval/metrics/religion/t_religion.json `
  --pos_cutoff 0.95 --neg_cutoff 0.99
```


## SINGLE LABEL

## convert the prediction to csv for bert training
```
python -m data.fourth_stage.csv_converter
```

## Train bert model
```
python -m data.fourth_stage.bert_train `
  --data_csv data/fourth_stage/train_discrimination_with_none.csv `
  --out_dir data/fourth_stage/bert_output `
  --epochs 4 `
  --train_bs 128 --eval_bs 128 `
  --num_workers 6 `
  --fp16
```

## MULTI LABEL

## convert to csv

```
python -m data.fourth_stage.csv_converter_multi_label
```

## execute mutlilabel bert model with
note: do not forget to update the pos_weight values which are returned by the previous scripts execution.
```
python -m data.fourth_stage.bert_train_multilabel `
  --data_csv data/fourth_stage/multilabel_train.csv `
  --out_dir data/fourth_stage/bert_multilabel_output `
  --epochs 4 `
  --train_bs 128 --eval_bs 128 `
  --max_len 256 `
  --num_workers 6 `
  --fp16 `
  --threshold 0.5 `
  --pos_weight 91.49 68.009 88.698
```

