This script allows you to calculate the AUPRC and pick the most appropirate threshold to
train the silver dataset for the bert models

## For race 

```
python -m data.eval.metrics.setfit_pick_threshold `
   --model_dir data/third_stage/race/race-setfit-model `
   --dev_jsonl data/silver_data/silver_labeled.jsonl `
   --label_key race `
   --text_key clean_text `
   --pos_floor 0.7 --neg_floor 0.9 `
   --out data/eval/metrics/race/t_race.json `
   --pr_pos_json data/eval/metrics/race/pr_race_pos.json `
   --pr_neg_json data/eval/metrics/race/pr_race_neg.json
```

## For gender

```
python -m data.eval.metrics.setfit_pick_threshold `
   --model_dir data/third_stage/gender/gender-setfit-model `
   --dev_jsonl data/silver_data/silver_labeled.jsonl `
   --label_key gender `
   --text_key clean_text `
   --pos_floor 0.55 --neg_floor 0.9 `
   --out data/eval/metrics/gender/t_gender.json `
   --pr_pos_json data/eval/metrics/gender/pr_gender_pos.json `
   --pr_neg_json data/eval/metrics/gender/pr_gender_neg.json
```

## For religion

```
python -m data.eval.metrics.setfit_pick_threshold `
   --model_dir data/third_stage/religion/religion-setfit-model `
   --dev_jsonl data/silver_data/silver_labeled.jsonl `
   --label_key religion `
   --text_key clean_text `
   --pos_floor 0.3 --neg_floor 0.9 `
   --out data/eval/metrics/religion/t_religion.json `
   --pr_pos_json data/eval/metrics/religion/pr_religion_pos.json `
   --pr_neg_json data/eval/metrics/religion/pr_religion_neg.json
```

## plot pr for positive and negative

```
python -m data.eval.metrics.plot_pr `
  --dirs data/eval/metrics/race data/eval/metrics/gender data/eval/metrics/religion `
  --title_pos "Positive PR, race/gender/religion" `
  --title_neg "Negative PR, race/gender/religion" `
  --out_pos data/eval/metrics/plots/pos_overlay.png `
  --out_neg data/eval/metrics/plots/neg_overlay.png `
  --step
```

