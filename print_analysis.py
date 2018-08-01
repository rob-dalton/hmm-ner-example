""" Script file to read k-scores results from S3 and print. """
import boto3
import json

from metrics import print_scores


if __name__ == "__main__":

    # load data from s3
    s3 = boto3.resource('s3')
    obj = s3.Object('awspot-instances', 'hmm_example/scores.json').get()
    contents = obj['Body'].read().decode('utf-8')

    scores = json.loads(contents)

    print_scores(scores['avg'])
