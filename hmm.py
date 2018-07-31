import typing
from typing import List

import numpy as np
import pandas as pd

from seqlearn.hmm import MultinomialHMM
from sklearn.model_selection import KFold

from sklearn import preprocessing as pp

from metrics import get_accuracy_metrics, avg_k_scores
from etc.types import DataFrame, Series

def add_one_hot_vectors(df_tokens: DataFrame, features: List[str]) -> None:
    '''One hot encode features from df. Add column for encoded vectors.'''

    # create one hot encoder
    vector_encoder = pp.MultiLabelBinarizer()
    unique_features = (df_tokens[f].unique() for f in features)
    vector_encoder.fit(unique_features)

    # get matrix of encoded values, split into vectors
    feature_values = zip(*(df_tokens[f] for f in features))
    vectors = np.vsplit(vector_encoder.fit_transform(feature_values), df_tokens.shape[0])
    df_tokens['one_hot_vector'] = vectors

    # transform 2d vectors to 1d
    df_tokens['one_hot_vector'] = df_tokens.one_hot_vector.apply(lambda x: x[0])


def get_sequence_lengths(df_tokens: DataFrame, seq_id_col: str) -> DataFrame:
    ''' Create DataFrame of sequence lengths '''

    # get non-sequence id column name for count
    count_cols = set(df_tokens.columns.values)
    count_cols.remove(seq_id_col)
    count_col = count_cols.pop()

    return df_tokens[[seq_id_col, count_col]].groupby(seq_id_col)\
                                             .agg('count')\
                                             .rename(columns={count_col: 'length'})\
                                             .copy()

def get_validated_scores(model, df_tokens: DataFrame, y_col: str,
                         seq_id_col: str, df_lengths: DataFrame, k: int) -> dict:
    '''Use k-fold cross validation to fit model on k folds of data. Collect and average scores.'''

    # create fold splittersdf 3
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    k_scores = {'scores': []}
    sequences = df_tokens[seq_id_col].unique()

    # iterate over folds
    # NOTE: kf.split returns indices, not values
    for test_indices, train_indices in kf.split(sequences):

        # get test, train sequences
        test_sequences = sequences[test_indices]
        train_sequences = sequences[train_indices]

        # split df into train, test
        df_test = df_tokens[df_tokens[seq_id_col].isin(test_sequences)]
        df_train = df_tokens[df_tokens[seq_id_col].isin(train_sequences)]
        df_test.reset_index(inplace=True)
        df_train.reset_index(inplace=True)

        # fit HMM model
        X_train = np.vstack(df_train.one_hot_vector.values)
        y_train = df_train[y_col]
        l_train = df_lengths.length[train_sequences]

        model.fit(X_train, y_train, l_train)

        # get test predictions
        X_test = np.vstack(df_test.one_hot_vector.values)
        y_test = df_test[y_col]
        l_test = df_lengths.length[test_sequences]

        preds = pd.Series(model.predict(X_test, l_test))

        # get accuracy metrics
        scores = get_accuracy_metrics(y_test, preds)

        # append scores
        k_scores['scores'].append(scores)

    # average scores
    k_scores['avg'] = avg_k_scores(k_scores['scores'])

    return k_scores

def fit_validate_hmm(df: DataFrame, y_col: str, seq_id_col: str, feature_cols: List[str],
                     k: int = 10) -> dict:
    ''' Fit HMM using features on sequence of tokens df_tokens from sequences seq_id_col. '''
    df_tokens = df.copy()

    # add one hot vectors encoded from features
    add_one_hot_vectors(df_tokens, feature_cols)

    # get sequence lengths
    df_lengths = get_sequence_lengths(df_tokens, seq_id_col)

    # cross validate model
    model = MultinomialHMM()

    # get cross validated scores
    k_scores = get_validated_scores(model, df_tokens, y_col, seq_id_col, df_lengths, k)

    # fit model on entire dataset
    X = np.vstack(df_tokens.one_hot_vector.values)
    y = df_tokens[y_col]
    l = df_lengths.length

    model.fit(X, y, l)

    # return model, data, metrics
    return {'model': model,
            'k_scores': k_scores}
