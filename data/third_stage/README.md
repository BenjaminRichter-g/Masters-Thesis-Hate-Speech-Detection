## Before you start

make sure to create a folder per hate category and import the discriminatory, non discriminatory and labeled data file from the second stage.

## How to run the auto train loop

Instead of doing it all manually, put into one script, simply run the auto_train_loop which will
and itll automatically:
1. consolidate new labelled dataset
2. train model on new data
3. apply model to dataset
4. ask you to keep labelling
5. repeat from step 1 to keep incrementally improving model

Latest version for all subcategories:

Note: you can add --confidence float
where float [0, 1] to change the threshold for confidence, 0.9 by default:

## How to run for different models

GENDER

```
python -m data.third_stage.auto_train_loop `
    --master_file data/third_stage/gender/labeled_gender.jsonl `
    --new_file  data/third_stage/gender/labeled_gender_new.jsonl `
    --model_dir data/third_stage/gender/gender-setfit-model `
    --full_input data/first_stage/output_timeline.jsonl `
    --full_output_dir data/third_stage/gender `
    --label_clean data/third_stage/gender/gender-setfit-model_non_discriminatory.jsonl `
    --label_hate  data/third_stage/gender/gender-setfit-model_discriminatory.jsonl `
    --label_out   data/third_stage/gender/labeled_gender_new.jsonl `
    --label_backup data/third_stage/gender/labeled_gender_backup.jsonl
```

RACE

```
python -m data.third_stage.auto_train_loop     --master_file data/third_stage/race/labeled_race.jsonl `
                    --new_file data/third_stage/race/labeled_race_new.jsonl `
                    --model_dir data/third_stage/race/race-setfit-model `
                    --full_input data/first_stage/output_timeline.jsonl `
                    --full_output_dir data/third_stage/race `
                    --label_clean data/third_stage/race/race-setfit-model_non_discriminatory.jsonl `
                    --label_hate data/third_stage/race/race-setfit-model_discriminatory.jsonl `
                    --label_out data/third_stage/race/labeled_race_new.jsonl `
                    --label_backup data/third_stage/race/labeled_race_backup.jsonl 
```

RELIGION 

```
python -m data.third_stage.auto_train_loop    --master_file data/third_stage/religion/labeled_religion.jsonl `
                    --new_file data/third_stage/religion/labeled_religion_new.jsonl `
                    --model_dir data/third_stage/religion/religion-setfit-model `
                    --full_input data/first_stage/output_timeline.jsonl `
                    --full_output_dir data/third_stage/religion `
                    --label_clean data/third_stage/religion/religion-setfit-model_non_discriminatory.jsonl `
                    --label_hate data/third_stage/religion/religion-setfit-model_discriminatory.jsonl `
                    --label_out data/third_stage/religion/labeled_religion_new.jsonl `
                    --label_backup data/third_stage/religion/labeled_religion_backup.jsonl 
```


