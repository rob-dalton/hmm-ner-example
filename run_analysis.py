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

    # get word positions
    positions = []
    pos = 0
    current_sent = 1
    for sent_num in df['Sentence_#']:
        if current_sent == sent_num:
            pos += 1
        else:
            pos = 1
            current_sent = sent_num
        positions.append(pos)

    df['Position'] = positions

    # add capitalized feature
    df['Capitalized'] = df.apply(lambda x: x['Word'].istitle() and x['Position'] != 1, axis=1)

    # fit model, obtain results
    models = {
                'unigram': ['Word', 'POS'],
                'unigram_capitalized': ['Word', 'POS', 'Capitalized'],
                'unigram_capitalized_position': ['Word', 'POS', 'Capitalized',
                                                 'Position']
             }
    results = {}
    for model, features in models.items():
        results[model] = fit_validate_hmm(df,
                                          y_col='Tag',
                                          seq_id_col='Sentence_#',
                                          feature_cols=features)

    # save scores to json file
    scores = {k: v['k_scores'] for k,v in results.items()}
    with open('scores.json', 'w') as f:
        json.dump(scores, f)

    # save file to s3
    s3 = boto3.resource('s3')
    bucket = s3.Bucket('awspot-instances')

    with open('scores.json', 'rb') as data:
        bucket.put_object(Key='hmm_example/scores.json', Body=data)
