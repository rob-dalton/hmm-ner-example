from collections import defaultdict
from sklearn.metrics import f1_score, precision_score, recall_score

def get_accuracy_metrics(y_test, preds):
    '''get f1, precision, recall scores (weighted, unweighted, and by class)'''

    # setup class labels and score functions
    labels = y_test.unique()
    scores = defaultdict(dict)
    score_funcs = {'f1': f1_score,
                   'precision': precision_score,
                   'recall': recall_score}

    # get scores by type, weight
    for s_type, s_func in score_funcs.items():
        class_scores = s_func(y_test, preds, average=None, labels=labels)
        scores[s_type]['by_class'] = {label: score for label, score in zip(labels, class_scores)}
        scores[s_type]['weighted'] = s_func(y_test, preds, average='weighted')
        scores[s_type]['macro'] = s_func(y_test, preds, average='macro')
        scores[s_type]['micro'] = s_func(y_test, preds, average='micro')

    return scores


def avg_k_scores(k_scores):
    '''average k scoring metrics'''
    avg_scores = {}
    k = len(k_scores)

    # get scores by type, weight
    for s_type, scores in k_scores.items():
        class_scores = s_func(y_test, preds, average=None, labels=labels)
        scores[s_type]['by_class'] = [*zip(labels, class_scores)]
        scores[s_type]['weighted'] = s_func(y_test, preds, average='weighted')
        scores[s_type]['macro'] = s_func(y_test, preds, average='macro')
        scores[s_type]['micro'] = s_func(y_test, preds, average='micro')

    return scores


def zero_dict(d):
    output = {}
    for k,v in d.items():
        if isinstance(v, dict):
            output[k] = zero_dict(v)
        else:
            output[k] = 0

    return output


def add_dicts(d1, d2):
    output = {}

    for k, v in zip(d1.keys(), zip(d1.values(), d2.values())):
        if all(isinstance(el, dict) or isinstance(el, defaultdict) for el in v):
            output[k] = add_dicts(v[0], v[1])
        else:
            output[k] = v[0] + v[1]

    return output


def divide_dict(d, denom):
    output = {}
    for k,v in d.items():
        if isinstance(v, dict):
            output[k] = divide_dict(v, denom)
        else:
            output[k] = v / denom

    return output

def avg_k_scores(k_scores):
    k_scores_sum = k_scores[0]

    for k_score in k_scores[1:]:
        k_scores_sum = add_dicts(k_scores_sum, k_score)

    return divide_dict(k_scores_sum, len(k_scores))

def print_scores(scores):
    for k in scores.keys():
        print(f"{k} scores".upper())
        print(f"Weighted:\t{scores[k]['weighted']}")
        print(f"Micro:\t\t{scores[k]['micro']}")
        print(f"Macro:\t\t{scores[k]['macro']}\n")
