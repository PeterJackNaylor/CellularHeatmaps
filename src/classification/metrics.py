
from __future__ import division
import tensorflow as tf

def binary_confusion(lbl, pred):
    """
    Returns tp, tn, fp, fn
    """
    true_p = tf.count_nonzero(pred * lbl)
    true_n = tf.count_nonzero((pred - 1) * (lbl - 1))
    false_p = tf.count_nonzero(pred * (lbl - 1))
    false_n = tf.count_nonzero((pred - 1) * lbl)
    return true_p, true_n, false_p, false_n

def recall(y_true, y_pred):
    y_true = tf.argmax(y_true, axis=-1)
    y_pred = tf.argmax(y_pred, axis=-1)
    tp, _, _, fn = binary_confusion(y_true, y_pred)
    recall_ = tf.divide(tp, tf.add(tp, fn))
    recall_ = tf.where(tf.is_nan(recall_), tf.zeros_like(recall_), recall_)
    return recall_

def precision(y_true, y_pred):
    y_true = tf.argmax(y_true, axis=-1)
    y_pred = tf.argmax(y_pred, axis=-1)
    tp, _, fp, _ = binary_confusion(y_true, y_pred)
    precision_ = tf.divide(tp, tf.add(tp, fp))
    precision_ = tf.where(tf.is_nan(precision_), tf.zeros_like(precision_), precision_)
    return precision_

def f1(y_true, y_pred):
    y_true = tf.argmax(y_true, axis=-1)
    y_pred = tf.argmax(y_pred, axis=-1)
    true_p, _, false_p, false_n = binary_confusion(y_true, y_pred)
    precision_ = tf.divide(true_p, tf.add(true_p, false_p))
    recall_ = tf.divide(true_p, tf.add(true_p, false_n))
    num = tf.multiply(precision_, recall_)
    dem = tf.add(precision_, recall_)
    f1_ = tf.scalar_mul(2, tf.divide(num, dem))
    f1_ = tf.where(tf.is_nan(f1_), tf.zeros_like(f1_), f1_)
    return f1_

def auc_roc(y_true, y_pred):
    # any tensorflow metric
    value, update_op = tf.contrib.metrics.streaming_auc(y_pred, y_true)

    # find all variables created for this metric
    metric_vars = [i for i in tf.local_variables() if 'auc_roc' in i.name.split('/')[1]]

    # Add metric variables to GLOBAL_VARIABLES collection.
    # They will be initialized for new session.
    for v in metric_vars:
        tf.add_to_collection(tf.GraphKeys.GLOBAL_VARIABLES, v)

    # force to update metric values
    with tf.control_dependencies([update_op]):
        value = tf.identity(value)
        return value

def import_metrics(loss="categorical_crossentropy", classes=2):
    if classes > 2:
        return ["accuracy"]
    elif loss == "categorical_crossentropy":
        return ["accuracy", recall, precision, f1, auc_roc]
    else:
        return ["mse"]