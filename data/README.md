## Clean data for visual inspection

To use the content viewer, execute the following command:

```
python -m data.content_viewer --input data/second_stage_classification/racist_posts.jsonl --output data/second_stage_classification/clean_posts.jsonl
```
you can replace the input with any jsonl file and get only the content back to facilitate reading.

## Important! This is the personal data scrubber

To run the DATA SCRUBBER, run the following script:
```
python -m data.data_scrubber --input input_path 
```
in this way the file will be overwritten and the data anonymized

