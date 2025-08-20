## How the first stage works

The first stage makes use of the data in labeled.jsonl. This is the file containing the synthetic seed which will be replaced
by the annotation in the second stage. This contains 27 instance of racism, 27 instances of gender and religious discrimintaion.
You will need to generate these with the help of an LLM such as deepseek or chatGPT, or use the existing ones if the classes are the same.
## How to run

To obtain new data simply run:

```
mastodoner instance --instance-url LINK_TO_INSTANCE --timeline output_timeline.jsonl --limit 100000
```
in our case we used
```
mastodoner instance --instance-url gameliberty.club --timeline output_timeline.jsonl --limit 100000
```

Scrub any personal data with:
```
python -m data.data_scrubber --input data/first_stage/output_timeline.jsonl
```

A lof of Mastodon posts language is marked as null when the content is in English.
To resolve this run this script to run a py3langid language identification and label the null posts to english.
```
python -m data.first_stage.relabel_null_to_english data/first_stage/output_timeline.jsonl
```
if you'd like a higher accuracy but longer processing (especially without GPU) you can run:
```
python -m data.first_stage.relabel_null_to_english_transformer data/first_stage/output_timeline.jsonl
```

To run the setfit_binary from root of project execute, this will train n number of binary classifiers as provided in the labelled.jsonl, aka if 3 classes are definied such as race, religion and gender,
the script will return 3 binary classifiers. 
```
python -m data.first_stage.setfit_binary_inital -n race religion gender
```

After the training is done, execute the following to classify the dataset you extracted.
This will automatically split the data into discriminatory_posts.jsonl and clean_posts.jsonl into the second_stage folder:
```
python -m data.first_stage.setfit_binary --model_dir data/first_stage --input data/first_stage/output_timeline.jsonl --output data/second_stage
```

note:
Make sure to have a venv created and activated!

This stage does not train a model on actual extracted data, it simply uses pre-extracted data to train a basic initial model, allowing for the second stage to find relevant data to train the actual discrimination binary classification.

Results for gender, race, religion with 0.8> confidence threshold:

Filtering posts: 99923it [05:31, 301.20it/s]
For the model: <setfit.modeling.SetFitModel object at 0x00000253A0864AC0>
Total posts treated: 26689
Nb posts discriminatory 532
Nb posts non-disc 26157
For the model: <setfit.modeling.SetFitModel object at 0x00000253A0864AC0>
Total posts treated: 26689
Nb posts discriminatory 537
For the model: <setfit.modeling.SetFitModel object at 0x00000253A0864AC0>
Total posts treated: 26689
Nb posts discriminatory 792
Nb posts non-disc 25897

Results with 0.95> confidence threshold:
For the model: data/first_stage\gender-initial-model
Total posts treated: 26689
Nb posts discriminatory 292
Nb posts non-disc 26397
For the model: data/first_stage\race-initial-model
Total posts treated: 26689
Nb posts discriminatory 267
Nb posts non-disc 26422
For the model: data/first_stage\religion-initial-model
Total posts treated: 26689
Nb posts discriminatory 460
Nb posts non-disc 26229
