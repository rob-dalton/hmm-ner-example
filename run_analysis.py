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

    # drop Tag prefixes
    df['Tag'] = df.Tag.apply(lambda tag: tag[2:] if '-' in tag else tag)

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
    # NOTE: All feature values must be same type for vector encoding.
    #       This is why Capitalized is a str, not a bool.
    def is_capitalized(row):
        if row['Word'].istitle() and row['Position'] != 1:
            return 'y'
        return 'n'
    df['Capitalized'] = df.apply(is_capitalized, axis=1)

    # convert Position to string
    df['Position'] = df['Position'].apply(str)

    # fit model, obtain results
    models = {
                'unigram': ['Word', 'POS'],
                'unigram_capitalized': ['Word', 'POS', 'Capitalized'],
                'unigram_capitalized_position': ['Word', 'POS', 'Capitalized',
                                                 'Position']
             }

    scores = {}
    for model, features in models.items():
        scores[model] = {}
        for alpha in [0.01, 0.05, 0.1, 0.15, 0.2, 0.25]:
            results = fit_validate_hmm(df,
                                       y_col='Tag',
                                       seq_id_col='Sentence_#',
                                       feature_cols=features,
                                       alpha=alpha)
            scores[alpha][model] = results['k_scores']['avg']

    # save scores to json file
    with open('scores.json', 'w') as f:
        json.dump(scores, f)

    # save file to s3
    s3 = boto3.resource('s3')
    bucket = s3.Bucket('awspot-instances')

    with open('scores.json', 'rb') as data:
        bucket.put_object(Key='hmm_example/scores.json', Body=data)
