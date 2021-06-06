import sys
from sklearn import metrics
import math
import os
from sklearn.metrics import auc
from copy import deepcopy
import numpy as np
import warnings
import time
warnings.filterwarnings(action='ignore', category=DeprecationWarning)
warnings.filterwarnings(action='ignore', category=RuntimeWarning)

def ranking_precision_score(Y_true, Y_score, k=10):
    """Precision at rank k
    Parameters
    ----------
    y_true : array-like, shape = [n_samples]
        Ground truth (true relevance labels).
    y_score : array-like, shape = [n_samples]
        Predicted scores.
    k : int
        Rank.
    Returns
    -------
    precision @k : float
    """
    sum_prec = 0.
    n = len(Y_true)

    unique_Y = np.unique(Y_true)

    if len(unique_Y) > 2:
        raise ValueError("Only supported for two relevance levels.")

    pos_label = unique_Y[1]
    n_pos = np.sum(Y_true == pos_label, axis=1)
    order = np.argsort(Y_score, axis=1)[:, ::-1]
    Y_true = np.array([x[y] for x, y in zip(Y_true, order[:, :k])])

    n_relevant = np.sum(Y_true == pos_label, axis=1)

    cnt = k
    prec = np.divide(n_relevant.astype(float), cnt)
    return np.average(prec)


def subset_accuracy(true_targets, predictions, per_sample=False, axis=0):
    result = np.all(true_targets == predictions, axis=axis)
    if not per_sample:
        result = np.mean(result)
    return result


def hamming_loss(true_targets, predictions, per_sample=False, axis=0):
    result = np.mean(np.logical_xor(true_targets, predictions), axis=axis)
    if not per_sample:
        result = np.mean(result)
    return result


def compute_tp_fp_fn(true_targets, predictions, axis=0):
    tp = np.sum(true_targets * predictions, axis=axis).astype('float32')
    fp = np.sum(np.logical_not(true_targets) * predictions, 
                   axis=axis).astype('float32')
    fn = np.sum(true_targets * np.logical_not(predictions), 
                   axis=axis).astype('float32')
    return (tp, fp, fn)


def example_f1_score(true_targets, predictions, per_sample=False, axis=0):
    tp, fp, fn = compute_tp_fp_fn(true_targets, predictions, axis=axis)

    numerator = 2*tp
    denominator = (np.sum(true_targets, axis=axis).astype('float32') + np.sum(predictions, axis=axis).astype('float32'))

    zeros = np.where(denominator == 0)[0]

    denominator = np.delete(denominator, zeros)
    numerator = np.delete(numerator, zeros)

    example_f1 = numerator/denominator

    if per_sample:
        f1 = example_f1
    else:
        f1 = np.mean(example_f1)

    return f1


def f1_score_from_stats(tp, fp, fn, average='micro'):
    assert len(tp) == len(fp)
    assert len(fp) == len(fn)

    if average not in set(['micro', 'macro']):
        raise ValueError("Specify micro or macro")

    if average == 'micro':
        f1 = 2*np.sum(tp) / \
            float(2*np.sum(tp) + np.sum(fp) + np.sum(fn))

    elif average == 'macro':

        def safe_div(a, b):
            """ ignore / 0, div0( [-1, 0, 1], 0 ) -> [0, 0, 0] """
            with np.errstate(divide='ignore', invalid='ignore'):
                c = np.true_divide(a, b)
            return c[np.isfinite(c)]
        tmp = safe_div(2*tp, 2*tp + fp + fn + 1e-6)
        #print(tmp)
        f1 = np.mean(safe_div(2*tp, 2*tp + fp + fn + 1e-6))

    return f1


def f1_score(true_targets, predictions, average='micro', axis=0):
    """
        average: str
            'micro' or 'macro'
        axis: 0 or 1
            label axis
    """
    if average not in set(['micro', 'macro']):
        raise ValueError("Specify micro or macro")

    tp, fp, fn = compute_tp_fp_fn(true_targets, predictions, axis=axis)
    f1 = f1_score_from_stats(tp, fp, fn, average=average)
    return f1


def compute_fdr(all_targets, all_predictions, fdr_cutoff=0.5):
    fdr_array = []
    for i in range(all_targets.shape[1]):
        try:
            precision, recall, thresholds = metrics.precision_recall_curve(all_targets[:, i], all_predictions[:, i], pos_label=1)
            fdr = 1- precision
            cutoff_index = next(i for i, x in enumerate(fdr) if x <= fdr_cutoff)
            fdr_at_cutoff = recall[cutoff_index]
            if not math.isnan(fdr_at_cutoff):
                fdr_array.append(np.nan_to_num(fdr_at_cutoff))
        except: 
            pass
    
    fdr_array = np.array(fdr_array)
    mean_fdr = np.mean(fdr_array)
    median_fdr = np.median(fdr_array)
    var_fdr = np.var(fdr_array)
    return mean_fdr, median_fdr, var_fdr, fdr_array


def compute_aupr(all_targets, all_predictions):
    aupr_array = []
    for i in range(all_targets.shape[1]):
        precision, recall, thresholds = metrics.precision_recall_curve(all_targets[:, i], all_predictions[:, i], pos_label=1)
        auPR = metrics.auc(recall, precision)
        if not math.isnan(auPR):
            aupr_array.append(np.nan_to_num(auPR))
    aupr_array = np.array(aupr_array)
    mean_aupr = np.mean(aupr_array)
    median_aupr = np.median(aupr_array)
    var_aupr = np.var(aupr_array)
    return mean_aupr, median_aupr, var_aupr, aupr_array


def compute_auc(all_targets, all_predictions):
    auc_array = []
    for i in range(all_targets.shape[1]):
        try:  
            auROC = metrics.roc_auc_score(all_targets[:, i], all_predictions[:, i])
            auc_array.append(auROC)
        except ValueError:
            pass
    auc_array = np.array(auc_array)
    mean_auc = np.mean(auc_array)
    median_auc = np.median(auc_array)
    var_auc = np.var(auc_array)
    return mean_auc, median_auc, var_auc, auc_array


def compute_metrics(predictions, targets, threshold, all_metrics=True):
    all_targets = deepcopy(targets)
    all_predictions = deepcopy(predictions)

    if all_metrics:
        meanAUC, medianAUC, varAUC, allAUC = compute_auc(all_targets, all_predictions)
        meanAUPR, medianAUPR, varAUPR, allAUPR = compute_aupr(all_targets, all_predictions)
        meanFDR, medianFDR, varFDR, allFDR = compute_fdr(all_targets, all_predictions)
    else:
        meanAUC, medianAUC, varAUC, allAUC = 0, 0, 0, 0
        meanAUPR, medianAUPR, varAUPR, allAUPR = 0, 0, 0, 0
        meanFDR, medianFDR, varFDR, allFDR = 0, 0, 0, 0

    p_at_1 = 0.
    p_at_3 = 0.
    p_at_5 = 0.
    
    p_at_1 = ranking_precision_score(Y_true=all_targets, Y_score=all_predictions, k=1)
    p_at_3 = ranking_precision_score(Y_true=all_targets, Y_score=all_predictions, k=3)
    p_at_5 = ranking_precision_score(Y_true=all_targets, Y_score=all_predictions, k=5)
    
    optimal_threshold = threshold
    
    all_predictions[all_predictions < optimal_threshold] = 0
    all_predictions[all_predictions >= optimal_threshold] = 1

    
    acc_ = list(subset_accuracy(all_targets, all_predictions, axis=1, per_sample=True))
    hl_ = list(hamming_loss(all_targets, all_predictions, axis=1, per_sample=True))
    exf1_ = list(example_f1_score(all_targets, all_predictions, axis=1, per_sample=True))        
    ACC = np.mean(acc_)
    hl = np.mean(hl_)
    HA = 1 - hl
    ebF1 = np.mean(exf1_)
    tp, fp, fn = compute_tp_fp_fn(all_targets, all_predictions, axis=0)

    miF1 = f1_score_from_stats(tp, fp, fn, average='micro')
    maF1 = f1_score_from_stats(tp, fp, fn, average='macro')

    metrics_dict = {}
    metrics_dict['ACC'] = ACC
    metrics_dict['HA'] = HA
    metrics_dict['ebF1'] = ebF1
    metrics_dict['miF1'] = miF1
    metrics_dict['maF1'] = maF1
    metrics_dict['meanAUC'] = meanAUC
    metrics_dict['medianAUC'] = medianAUC
    metrics_dict['varAUC'] = varAUC
    metrics_dict['allAUC'] = allAUC
    metrics_dict['meanAUPR'] = meanAUPR
    metrics_dict['medianAUPR'] = medianAUPR
    metrics_dict['varAUPR'] = varAUPR
    metrics_dict['allAUPR'] = allAUPR
    metrics_dict['meanFDR'] = meanFDR
    metrics_dict['medianFDR'] = medianFDR
    metrics_dict['varFDR'] = varFDR
    metrics_dict['allFDR'] = allFDR
    metrics_dict['p_at_1'] = p_at_1
    metrics_dict['p_at_3'] = p_at_3
    metrics_dict['p_at_5'] = p_at_5

    return metrics_dict

