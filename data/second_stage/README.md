# How to run

## 1) Manual Labelling

Now that an initial filtering is done, you'll be able to manually label some data quickly 
with the use of the manual_labelling.py script. This is due to the fact that the first stage allowed us 
to filter out the revelevant data preventing the need to parse the entire dataset.
The script thus shuffles the clean_posts and discriminatory_posts and servers the data for you to label with the simple click of a button.
You can interrupt this process and pick it up at a later date as the script ensure that no duplicates will appear in the labelled data.

To run it simply execute:

For gender:
```
python -m data.second_stage.manual_labelling --clean data/second_stage/gender-initial-model_non_discriminatory.jsonl --hate data/second_stage/gender-initial-model_discriminatory.jsonl --output data/second_stage/labeled_gender.jsonl --backup data/second_stage/labeled_gender_backup.jsonl
```

For race:
```
python -m data.second_stage.manual_labelling --clean data/second_stage/race-initial-model_non_discriminatory.jsonl --hate data/second_stage/race-initial-model_discriminatory.jsonl --output data/second_stage/labeled_race.jsonl --backup data/second_stage/labeled_race_backup.jsonl
```

For religion:
```
python -m data.second_stage.manual_labelling --clean data/second_stage/religion-initial-model_non_discriminatory.jsonl --hate data/second_stage/religion-initial-model_discriminatory.jsonl --output data/second_stage/labeled_religion.jsonl --backup data/second_stage/labeled_religion_backup.jsonl
```

Once the data is labelled, execute the following to train based off of that model.
To run the setfit_binary from root of project execute:

```
python -m data.racist_binary_classif.setfit_binary_train
```

note:
Make sure to have a venv created and activated!

## Performance

Inital training achieves:

with labels:
9 racists
72 non-racists
0.88 accuracy

11 racists
72 non-racists
0.94 accuracy

51 racists
72 non-racists
Accuracy: 0.88