import numpy as np
from collections import namedtuple

ConfusionTuple = namedtuple(
    'ConfusionTuple', [
        'true_positives',
        'false_positives',
        'false_negatives',
        'true_negatives',
    ]
)

def do_confusion(ground_truth, test):
    '''
    Compare the ground truth and test values and return the confusion values, tp, fp, fn, tn.

    Args:
        ground_truth (np.ndarray): The ground truth. Shape (*batch_shape)
        test (np.ndarray): The test values. Shape (*param_batch_dims, *batch_shape)

    Returns:
        true_positives (np.ndarray): The true positives. Shape (*param_batch_dims)
        false_positives (np.ndarray): The false positives. Shape (*param_batch_dims)
        false_negatives (np.ndarray): The false negatives. Shape (*param_batch_dims)
        true_negatives (np.ndarray): The true negatives. Shape (*param_batch
    '''

    batch_shape = ground_truth.shape
    param_batch_shape = test.shape[:-len(batch_shape)]

    gt = ground_truth.reshape((1,)*(len(param_batch_shape)+2) + batch_shape).astype(bool)
    tt = test.reshape((1,)*2 + test.shape).astype(bool)
    gt = np.concatenate([gt, ~gt], axis=1)
    tt = np.concatenate([tt, ~tt], axis=0)
    confusion = np.logical_and(gt, tt).sum(axis=tuple(np.arange(-len(batch_shape), 0, dtype=int))).reshape((4,) + param_batch_shape)
    return ConfusionTuple(*confusion)

def calculate_precision_recall(true_positives, false_positives, false_negatives):
    '''
    Calculate the precision and recall from the confusion values.

    Args:
        true_positives (np.ndarray): The true positives. Shape (*param_batch_dims)
        false_positives (np.ndarray): The false positives. Shape (*param_batch_dims)
        false_negatives (np.ndarray): The false negatives. Shape (*param_batch_dims)

    Returns:
        precision (np.ndarray): The precision. Shape (*param_batch_dims)
        recall (np.ndarray): The recall. Shape (*param_batch_dims)
    '''
    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)
    return precision, recall