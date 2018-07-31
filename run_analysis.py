""" Script file to run HMM NER example and save results to S3. """
import boto3
import json

import pandas as pd

from hmm import fit_validate_hmm

if __name__ == "__main__":
    # load data
    with open('dataset_05-22-2018.txt', errors='ignore') as f:
        df = pd.read_csv(f, delimiter="\t", index_col=0,
                         dtype={'Sentence #': int})

    # remove space in col name
    df.rename(columns={'Sentence #': 'Sentence_#'}, inplace=True)

    # convert data to strings
    df['Word'] = df.Word.apply(str)
    df['POS'] = df.POS.apply(str)

    # fit model, obtain results
    results = fit_validate_hmm(df,
                               y_col='Tag',
                               seq_id_col='Sentence_#',
                               feature_cols=['Word', 'POS'])

    # save scores to json file
    with open('scores.json', 'w') as f:
        json.dump(results['k_scores'], f)

    # save file to s3
    s3 = boto3.resource('s3')
    bucket = s3.Bucket('awspot-instance')

    with open('scores.json', 'rb') as data:
        bucket.put_object(Key='scores.json', Body=data)
